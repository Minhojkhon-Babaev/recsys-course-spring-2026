import json
from collections import defaultdict

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from .recommender import Recommender


class SessionGraphMF(Recommender):
    """Session fold-in поверх SVD-эмбеддингов item-item графа.

    Граф собирается из SASRec-I2I и LightFM-I2I. Вектор пользователя —
    recency/time-взвешенное среднее эмбеддингов прослушанных треков.
    Кандидаты: I2I якоря, 2-hop и треки того же артиста.
    """

    def __init__(
        self,
        listen_history_redis,
        catalog,
        sasrec_path,
        lightfm_path,
        fallback,
        dim=64,
    ):
        self.listen_history_redis = listen_history_redis
        self.catalog = catalog
        self.fallback = fallback
        self.dim = dim
        self.i2i = defaultdict(list)
        self.i2i_set = defaultdict(set)
        self.emb = {}
        self.track_to_artist = {}
        self.artist_to_tracks = defaultdict(list)
        self._index_catalog()
        self._build(sasrec_path, lightfm_path)

    def _index_catalog(self):
        for track in getattr(self.catalog, "tracks", []):
            tid = int(track.track)
            artist = track.artist
            self.track_to_artist[tid] = artist
            self.artist_to_tracks[artist].append(tid)

    def _read_i2i(self, path, weight):
        edges = []
        with open(path) as fh:
            for line in fh:
                row = json.loads(line)
                src = int(row.get("item_id", row.get("track", row.get("user"))))
                recs = row.get("recommendations", row.get("tracks", []))
                for rank, dst in enumerate(recs):
                    dst = int(dst)
                    edges.append((src, dst, weight / float(rank + 1)))
                    if dst not in self.i2i_set[src]:
                        self.i2i[src].append(dst)
                        self.i2i_set[src].add(dst)
        return edges

    def _build(self, sasrec_path, lightfm_path):
        edges = []
        edges.extend(self._read_i2i(sasrec_path, 1.2))
        edges.extend(self._read_i2i(lightfm_path, 1.0))
        if not edges:
            return

        item_ids = sorted({src for src, _, _ in edges} | {dst for _, dst, _ in edges})
        index = {item: i for i, item in enumerate(item_ids)}
        n = len(item_ids)
        rows, cols, data = [], [], []
        for src, dst, weight in edges:
            rows.extend([index[src], index[dst]])
            cols.extend([index[dst], index[src]])
            data.extend([weight, 0.35 * weight])

        matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        n_comp = min(self.dim, max(n - 1, 1))
        factors = TruncatedSVD(n_components=n_comp, random_state=13).fit_transform(matrix)
        if n_comp < self.dim:
            factors = np.pad(factors, ((0, 0), (0, self.dim - n_comp)))
        norms = np.linalg.norm(factors, axis=1, keepdims=True)
        factors = factors / np.clip(norms, 1e-8, None)
        for item, idx in index.items():
            self.emb[item] = factors[idx].astype(np.float32)

    def _history(self, user):
        raw = self.listen_history_redis.lrange("user:{}:listens".format(user), 0, -1)
        history = []
        for item in raw or []:
            if isinstance(item, bytes):
                item = item.decode("utf-8")
            row = json.loads(item)
            history.append((int(row["track"]), float(row["time"])))
        return history

    def _user_vector(self, history):
        vec = np.zeros(self.dim, dtype=np.float32)
        weight_sum = 0.0
        for pos, (track, listened) in enumerate(history):
            emb = self.emb.get(track)
            if emb is None:
                continue
            recency = 0.72 ** pos
            weight = recency * (0.15 + max(listened, 0.0))
            vec += weight * emb
            weight_sum += weight
        if weight_sum > 0:
            vec /= weight_sum
            norm = float(np.linalg.norm(vec))
            if norm > 1e-8:
                vec /= norm
        return vec

    def _candidates(self, history, prev_track):
        pool = []
        anchors = [prev_track] + [track for track, _ in history[:4]]
        for anchor in anchors:
            first_hop = self.i2i.get(anchor, [])[:16]
            pool.extend(first_hop)
            for mid in first_hop[:6]:
                pool.extend(self.i2i.get(mid, [])[:6])
        artist = self.track_to_artist.get(prev_track)
        if artist is not None:
            pool.extend(self.artist_to_tracks.get(artist, [])[:24])
        return pool

    def recommend_next(self, user, prev_track, prev_track_time):
        history = self._history(user)
        if not history:
            history = [(prev_track, prev_track_time)]
        seen = {track for track, _ in history}
        seen.add(prev_track)
        user_vec = self._user_vector(history)
        prev_artist = self.track_to_artist.get(prev_track)
        prev_neighbors = self.i2i_set.get(prev_track, set())

        best_track = None
        best_score = -1e9
        for cand in self._candidates(history, prev_track):
            if cand in seen:
                continue
            emb = self.emb.get(cand)
            if emb is None:
                continue
            score = float(np.dot(user_vec, emb))
            if cand in prev_neighbors:
                score += 0.10
            if prev_artist is not None and self.track_to_artist.get(cand) == prev_artist:
                score += 0.16
            if score > best_score:
                best_score = score
                best_track = cand

        if best_track is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)
        return int(best_track)
