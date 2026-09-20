#!/usr/bin/env python3
"""Локальный A/B без Docker: тот же сплит и те же рекомендеры, что в botify."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "botify"))
sys.path.insert(0, str(ROOT / "sim"))

from botify.experiment import Experiments, Treatment  # noqa: E402
from botify.recommenders.i2i import I2IRecommender  # noqa: E402
from botify.recommenders.random import Random  # noqa: E402
from botify.recommenders.session_graph_mf import SessionGraphMF  # noqa: E402
from botify.track import Track  # noqa: E402
from sim.envs import RecEnv  # noqa: E402
from sim.envs.config import RecEnvConfigSchema  # noqa: E402


class MemoryStore:
    def __init__(self):
        self.kv = {}
        self.lists = defaultdict(list)

    def set(self, key, value):
        self.kv[key] = value

    def get(self, key):
        if key in self.kv:
            return self.kv[key]
        return self.kv.get(str(key))

    def lpush(self, key, value):
        self.lists[key].insert(0, value)

    def lrange(self, key, start, end):
        data = self.lists.get(key, [])
        if end == -1:
            return data[start:]
        return data[start : end + 1]

    def ltrim(self, key, start, end):
        self.lists[key] = self.lists[key][start : end + 1]

    def randomkey(self):
        return random.choice(list(self.kv.keys()))


def load_catalog(path):
    tracks = []
    with open(path) as fh:
        for line in fh:
            row = json.loads(line)
            tracks.append(
                Track(int(row["track"]), row["artist"], row.get("title", ""), row.get("recommendations", []))
            )
    return SimpleNamespace(tracks=tracks)


def load_i2i(store, path):
    with open(path) as fh:
        for line in fh:
            row = json.loads(line)
            item = int(row["item_id"])
            recs = [int(x) for x in row["recommendations"]]
            store.set(item, pickle.dumps(recs))


def persist(store, user, track, listened):
    key = "user:{}:listens".format(user)
    store.lpush(key, json.dumps({"track": int(track), "time": float(listened)}))
    store.ltrim(key, 0, 9)


def run(episodes, seed, data_dir):
    random.seed(seed)
    np.random.seed(seed)

    botify_data = ROOT / "botify" / "data"
    catalog = load_catalog(botify_data / "tracks.json")
    history = MemoryStore()
    tracks_redis = MemoryStore()
    sasrec_store = MemoryStore()
    for track in catalog.tracks:
        tracks_redis.set(track.track, pickle.dumps(track))
    load_i2i(sasrec_store, botify_data / "sasrec_i2i.jsonl")

    fallback = Random(tracks_redis)
    control = I2IRecommender(history, sasrec_store, fallback)
    treatment = SessionGraphMF(
        history,
        catalog,
        str(botify_data / "sasrec_i2i.jsonl"),
        str(botify_data / "lightfm_i2i.jsonl"),
        fallback,
        str(botify_data / "hstu_recommendations.json"),
    )

    out_dir = Path(data_dir).resolve() / "local"
    os.chdir(ROOT / "sim")
    sim_cfg = RecEnvConfigSchema().load(yaml.full_load(open("config/env.yml")))
    env = RecEnv(sim_cfg)
    env.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "data.json"
    experiment = Experiments.SESSION_GRAPH_MF

    with open(log_path, "w") as log:
        for episode in range(episodes):
            observation, _ = env.reset()
            reward = 1.0
            done = False
            while not done:
                user = int(observation["user"])
                prev = int(observation["track"])
                persist(history, user, prev, reward)
                started = time.time()
                arm = experiment.assign(user)
                if arm == Treatment.C:
                    nxt = control.recommend_next(user, prev, reward)
                else:
                    nxt = treatment.recommend_next(user, prev, reward)
                latency = time.time() - started
                row = {
                    "message": "next",
                    "timestamp": int(datetime.now().timestamp() * 1000),
                    "user": user,
                    "track": prev,
                    "time": float(reward),
                    "latency": latency,
                    "recommendation": int(nxt),
                    "experiments": {experiment.name: arm.name},
                }
                log.write(json.dumps(row) + "\n")
                observation, reward, terminated, truncated, _ = env.step(int(nxt))
                done = terminated or truncated

            user = int(observation["user"])
            prev = int(observation["track"])
            persist(history, user, prev, reward)
            log.write(
                json.dumps(
                    {
                        "message": "last",
                        "timestamp": int(datetime.now().timestamp() * 1000),
                        "user": user,
                        "track": prev,
                        "time": float(reward),
                        "latency": 0.0,
                        "recommendation": None,
                        "experiments": {experiment.name: experiment.assign(user).name},
                    }
                )
                + "\n"
            )
            if (episode + 1) % 500 == 0:
                print("episodes {}/{}".format(episode + 1, episodes), flush=True)

    print("wrote", log_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=31312)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    run(args.episodes, args.seed, args.data_dir)


if __name__ == "__main__":
    main()
