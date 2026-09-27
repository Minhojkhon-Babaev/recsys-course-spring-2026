#!/usr/bin/env python3
"""ДЗ1: скоринг пар (user, track) и сабмит для All Cups.

Ожидаемые файлы:
  data/train.csv  — колонки user, track, time
  data/test.csv   — колонки user, track

Результат:
  data/submit.csv — колонки user, track, score
  data/val_metrics.json — локальная NDCG на hold-out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder

try:
    from implicit.als import AlternatingLeastSquares
except ImportError:
    AlternatingLeastSquares = None

try:
    from lightfm import LightFM
    from lightfm.data import Dataset as LFMDataset
except ImportError:
    LightFM = None
    LFMDataset = None


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def ndcg_user(times: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    gains = np.maximum(times[order], 0.0)
    if gains.sum() <= 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.dot(gains, discounts))
    ideal = np.sort(gains)[::-1]
    idcg = float(np.dot(ideal, discounts))
    return dcg / idcg if idcg > 0 else 0.0


def mean_ndcg(df: pd.DataFrame, score_col: str = "score", rel_col: str = "time") -> float:
    values = []
    for _, grp in df.groupby("user", sort=False):
        if len(grp) < 2:
            continue
        values.append(ndcg_user(grp[rel_col].to_numpy(), grp[score_col].to_numpy()))
    return float(np.mean(values)) if values else 0.0


def random_holdout(train: pd.DataFrame, min_user_events: int = 3, extra: int = 4, seed: int = 42):
    """time — доля дослушивания, не таймстемп, поэтому hold-out случайный."""
    rng = np.random.default_rng(seed)
    train = train.reset_index(drop=True)
    sizes = train.groupby("user", sort=False)["track"].transform("size")
    eligible = sizes >= min_user_events
    pick = []
    for _, idx in train.loc[eligible].groupby("user", sort=False).groups.items():
        pick.append(int(rng.choice(np.asarray(idx))))
    valid_pos = train.loc[pick].copy()
    fit = train.drop(index=pick).copy()
    extra_rows = (
        fit.groupby("user", sort=False)
        .sample(n=extra, replace=True, random_state=seed)
        .reset_index(drop=True)
    )
    valid = pd.concat([valid_pos, extra_rows], ignore_index=True)
    return fit, valid


def build_encoders(train: pd.DataFrame, test: pd.DataFrame):
    users = pd.Index(pd.concat([train["user"], test["user"]]).unique())
    items = pd.Index(pd.concat([train["track"], test["track"]]).unique())
    user_enc = LabelEncoder().fit(users)
    item_enc = LabelEncoder().fit(items)
    return user_enc, item_enc


def interaction_matrix(df: pd.DataFrame, user_enc, item_enc, n_users: int, n_items: int):
    rows = user_enc.transform(df["user"])
    cols = item_enc.transform(df["track"])
    data = np.clip(df["time"].to_numpy(dtype=np.float32), 0.05, None)
    return sparse.coo_matrix((data, (rows, cols)), shape=(n_users, n_items)).tocsr()


def fit_svd(matrix: sparse.csr_matrix, dim: int = 64):
    n_comp = min(dim, matrix.shape[1] - 1, matrix.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    user_emb = svd.fit_transform(matrix)
    item_emb = svd.components_.T
    if n_comp < dim:
        user_emb = np.pad(user_emb, ((0, 0), (0, dim - n_comp)))
        item_emb = np.pad(item_emb, ((0, 0), (0, dim - n_comp)))
    return user_emb.astype(np.float32), item_emb.astype(np.float32)


def fit_als(matrix: sparse.csr_matrix, dim: int = 64, iterations: int = 18):
    if AlternatingLeastSquares is None:
        return None, None
    model = AlternatingLeastSquares(
        factors=dim,
        regularization=0.08,
        iterations=iterations,
        random_state=42,
        calculate_training_loss=False,
    )
    # implicit 0.7+: fit(user_items)
    conf = matrix.astype(np.float32)
    conf.data = 1.0 + 25.0 * conf.data
    model.fit(conf)
    return model.user_factors.astype(np.float32), model.item_factors.astype(np.float32)


def fit_lightfm(train: pd.DataFrame, user_enc, item_enc, n_users: int, n_items: int):
    if LightFM is None:
        return None
    dataset = LFMDataset()
    dataset.fit(np.arange(n_users), np.arange(n_items))
    interactions, weights = dataset.build_interactions(
        zip(
            user_enc.transform(train["user"]),
            item_enc.transform(train["track"]),
            np.clip(train["time"].to_numpy(), 0.05, None),
        )
    )
    model = LightFM(
        no_components=64,
        loss="warp",
        learning_schedule="adagrad",
        user_alpha=1e-6,
        item_alpha=1e-6,
        random_state=42,
    )
    model.fit(interactions, sample_weight=weights, epochs=16, num_threads=4, verbose=True)
    return model


def lightfm_scores(model, users_idx: np.ndarray, items_idx: np.ndarray) -> np.ndarray:
    if model is None:
        return np.zeros(len(users_idx), dtype=np.float32)
    return model.predict(users_idx, items_idx, num_threads=4).astype(np.float32)


def pair_dots(user_emb, item_emb, users_idx, items_idx) -> np.ndarray:
    if user_emb is None:
        return np.zeros(len(users_idx), dtype=np.float32)
    return np.einsum("ij,ij->i", user_emb[users_idx], item_emb[items_idx])


def stat_features(train: pd.DataFrame) -> dict:
    user_stats = train.groupby("user")["time"].agg(["mean", "count", "sum"])
    item_stats = train.groupby("track")["time"].agg(["mean", "count", "sum"])
    return {
        "user": user_stats,
        "item": item_stats,
        "global_mean": float(train["time"].mean()),
    }


def attach_stats(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    out = df.copy()
    user_stats = stats["user"]
    item_stats = stats["item"]
    g = stats["global_mean"]
    out["user_mean"] = out["user"].map(user_stats["mean"]).fillna(g)
    out["user_count"] = np.log1p(out["user"].map(user_stats["count"]).fillna(0))
    out["user_sum"] = out["user"].map(user_stats["sum"]).fillna(0)
    out["item_mean"] = out["track"].map(item_stats["mean"]).fillna(g)
    out["item_count"] = np.log1p(out["track"].map(item_stats["count"]).fillna(0))
    out["item_sum"] = out["track"].map(item_stats["sum"]).fillna(0)
    return out


STAT_COLS = ["user_mean", "user_count", "user_sum", "item_mean", "item_count", "item_sum"]


def make_features(df, user_enc, item_enc, svd_u, svd_i, als_u, als_i, lfm, stats):
    df = attach_stats(df, stats)
    users_idx = user_enc.transform(df["user"])
    items_idx = item_enc.transform(df["track"])
    parts = [
        df[STAT_COLS].to_numpy(dtype=np.float32),
        pair_dots(svd_u, svd_i, users_idx, items_idx).reshape(-1, 1),
        pair_dots(als_u, als_i, users_idx, items_idx).reshape(-1, 1),
        lightfm_scores(lfm, users_idx, items_idx).reshape(-1, 1),
    ]
    return np.hstack(parts).astype(np.float32)


def fit_stacker(X: np.ndarray, y: np.ndarray) -> Ridge:
    model = Ridge(alpha=2.0, random_state=42)
    model.fit(X, y)
    return model


def run(train_path: Path, test_path: Path, submit_path: Path, metrics_path: Path) -> None:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    for col in ("user", "track"):
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
    train["time"] = train["time"].astype(float)

    fit, valid = random_holdout(train)
    user_enc, item_enc = build_encoders(train, test)
    n_users, n_items = len(user_enc.classes_), len(item_enc.classes_)
    print(
        "users={} items={} train={} valid={} test={}".format(
            n_users, n_items, len(fit), len(valid), len(test)
        )
    )

    matrix = interaction_matrix(fit, user_enc, item_enc, n_users, n_items)
    print("fit SVD")
    svd_u, svd_i = fit_svd(matrix, dim=64)
    print("fit ALS")
    als_u, als_i = fit_als(matrix, dim=64)
    print("fit LightFM")
    lfm = fit_lightfm(fit, user_enc, item_enc, n_users, n_items)
    stats = stat_features(fit)

    X_valid = make_features(valid, user_enc, item_enc, svd_u, svd_i, als_u, als_i, lfm, stats)
    stacker = fit_stacker(X_valid, valid["time"].to_numpy(dtype=np.float32))
    valid = valid.copy()
    valid["score"] = stacker.predict(X_valid)
    val_ndcg = mean_ndcg(valid)
    print("hold-out NDCG={:.6f}".format(val_ndcg))
    print("stacker coefs={}".format(np.round(stacker.coef_, 4)))

    print("refit on full train")
    matrix = interaction_matrix(train, user_enc, item_enc, n_users, n_items)
    svd_u, svd_i = fit_svd(matrix, dim=64)
    als_u, als_i = fit_als(matrix, dim=64)
    lfm = fit_lightfm(train, user_enc, item_enc, n_users, n_items)
    stats = stat_features(train)
    # Стекер учится на hold-out, чтобы не переобучиться на те же пары, что видел ALS/SVD.
    X_valid = make_features(valid, user_enc, item_enc, svd_u, svd_i, als_u, als_i, lfm, stats)
    stacker = fit_stacker(X_valid, valid["time"].to_numpy(dtype=np.float32))

    submit = test[["user", "track"]].copy()
    X_test = make_features(submit, user_enc, item_enc, svd_u, svd_i, als_u, als_i, lfm, stats)
    submit["score"] = stacker.predict(X_test)
    submit_path.parent.mkdir(parents=True, exist_ok=True)
    submit.to_csv(submit_path, index=False)
    metrics_path.write_text(
        json.dumps(
            {
                "holdout_ndcg": val_ndcg,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "used_als": als_u is not None,
                "used_lightfm": lfm is not None,
            },
            indent=2,
        )
    )
    print("wrote {} rows={}".format(submit_path, len(submit)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DATA / "train.csv")
    parser.add_argument("--test", type=Path, default=DATA / "test.csv")
    parser.add_argument("--submit", type=Path, default=DATA / "submit.csv")
    parser.add_argument("--metrics", type=Path, default=DATA / "val_metrics.json")
    args = parser.parse_args()
    if not args.train.exists():
        raise SystemExit("Нет {}".format(args.train))
    if not args.test.exists():
        raise SystemExit("Нет {}".format(args.test))
    run(args.train, args.test, args.submit, args.metrics)


if __name__ == "__main__":
    main()
