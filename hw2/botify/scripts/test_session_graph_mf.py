#!/usr/bin/env python3
"""Локальный смоук SessionGraphMF без Redis/Docker."""

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from botify.recommenders.session_graph_mf import SessionGraphMF  # noqa: E402


class FakeRedis(object):
    def __init__(self, history):
        self.history = history

    def lrange(self, key, start, end):
        return [json.dumps({"track": t, "time": tm}) for t, tm in self.history]


class FakeFallback(object):
    def recommend_next(self, user, prev_track, prev_track_time):
        return 999


def main():
    tracks = [
        SimpleNamespace(track=1, artist="A", title="a"),
        SimpleNamespace(track=2, artist="A", title="b"),
        SimpleNamespace(track=3, artist="B", title="c"),
        SimpleNamespace(track=4, artist="B", title="d"),
        SimpleNamespace(track=5, artist="A", title="e"),
    ]
    catalog = SimpleNamespace(tracks=tracks)
    with tempfile.TemporaryDirectory() as tmp:
        sasrec = Path(tmp) / "sasrec.jsonl"
        lightfm = Path(tmp) / "lightfm.jsonl"
        sasrec.write_text(
            json.dumps({"item_id": 1, "recommendations": [2, 5, 3]}) + "\n"
            + json.dumps({"item_id": 2, "recommendations": [5, 1]}) + "\n"
        )
        lightfm.write_text(
            json.dumps({"item_id": 1, "recommendations": [5, 2]}) + "\n"
            + json.dumps({"item_id": 3, "recommendations": [4]}) + "\n"
        )
        model = SessionGraphMF(
            FakeRedis([(1, 0.9)]),
            catalog,
            str(sasrec),
            str(lightfm),
            FakeFallback(),
            dim=8,
        )
        nxt = model.recommend_next(7, 1, 0.9)
        assert nxt in {2, 5, 3, 4}, nxt
        assert nxt != 1
        print("session_graph_mf smoke ok, next=", nxt)


if __name__ == "__main__":
    main()
