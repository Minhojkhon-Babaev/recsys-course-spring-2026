import json
from collections import defaultdict

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from .recommender import Recommender


class SessionGraphMF(Recommender):
    """SVD fold-in + HSTU/I2I кандидаты с штрафом за повтор артиста.

    В симуляторе повтор артиста режет listen time, поэтому same-artist
    не бустим, а наоборот понижаем. Персонализация — HSTU и recency fold-in.
    """

    def __init__(
        self,
        listen_history_redis,
        catalog,
        sasrec_path,
        lightfm_path,
        fallback,
        hstu_path=None,
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
        self.user_hstu = {}
        self._index_catalog()
        self._build(sasrec_path, lightfm_path)
        if hstu_path:
            self._load_hstu(hstu_path)

    def _index_catalog(self):
        for track in getattr(self.catalog, "tracks", []):
            self.track_to_artist[int(track.track)] = track.artist

    def _load_hstu(self, path):
        with open(path) as fh:
            for line in fh:
                row = json.loads(line)
                user = int(row.get("user", row.get("item_id", -1)))
                recs = [int(x) for x in row.get("tracks", row.get("recommendations", []))]
                self.user_hstu[user] = recs

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
            recency = 0.55 ** pos
            weight = recency * (0.2 + max(listened, 0.0))
            vec += weight * emb
            weight_sum += weight
        if weight_sum > 0:
            vec /= weight_sum
            norm = float(np.linalg.norm(vec))
            if norm > 1e-8:
                vec /= norm
        return vec

    def _candidates(self, user, history, prev_track):
        pool = []
        pool.extend(self.user_hstu.get(user, [])[:40])
        pool.extend(self.i2i.get(prev_track, [])[:12])
        for track, listened in history[:3]:
            if listened >= 0.35:
                pool.extend(self.i2i.get(track, [])[:8])
        return pool

    def recommend_next(self, user, prev_track, prev_track_time):
        history = self._history(user)
        if not history:
            history = [(prev_track, prev_track_time)]
        seen = {track for track, _ in history}
        seen.add(prev_track)
        user_vec = self._user_vector(history)
        recent_artists = set()
        for track, _ in history[:4]:
            artist = self.track_to_artist.get(track)
            if artist is not None:
                recent_artists.add(artist)
        prev_neighbors = self.i2i_set.get(prev_track, set())
        hstu_set = set(self.user_hstu.get(user, [])[:40])

        best_track = None
        best_score = -1e9
        for cand in self._candidates(user, history, prev_track):
            if cand in seen:
                continue
            emb = self.emb.get(cand)
            if emb is None:
                continue
            score = float(np.dot(user_vec, emb))
            if cand in prev_neighbors:
                score += 0.08
            if cand in hstu_set:
                score += 0.10
            artist = self.track_to_artist.get(cand)
            if artist is not None and artist in recent_artists:
                score -= 0.40
            if score > best_score:
                best_score = score
                best_track = cand

        if best_track is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)
        return int(best_track)
