"""Experimental VN1 grouped LightGBM with leakage-safe demand clusters.

Client×warehouse interactions and frozen demand clusters complement the
existing global LightGBM. Cluster centroids are fit ONLY on the first 65
observed weeks, so their meanings remain stable across rolling origins.
No competition phase targets are used for fitting, feature selection or
cluster construction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from vn1_from_scratch import HORIZON, KEYS, forecast as statistical_forecast, read_sales, vn1_score
from vn1_lgbm_ensemble import (
    CATEGORICAL, LGBM_PARAMS, START_TRAIN_WEEK, _features, _training_origins,
    read_history,
)

GROUP_CATEGORICAL = CATEGORICAL + ["client_warehouse", "demand_cluster"]


def demand_profile(history: pd.DataFrame, cutoff: int) -> np.ndarray:
    """Stable six-dimensional per-series profile from values available at cutoff."""
    if cutoff < START_TRAIN_WEEK or cutoff > history.shape[1] - len(KEYS):
        raise ValueError("Cutoff outside historical demand")
    y = history.iloc[:, len(KEYS):len(KEYS) + cutoff].to_numpy(dtype=np.float64)
    recent52, recent13, recent4 = y[:, -52:], y[:, -13:], y[:, -4:]
    prior13 = y[:, -26:-13]
    mean52 = recent52.mean(axis=1)
    mean13 = recent13.mean(axis=1)
    trend = (mean13 + 1.0) / (prior13.mean(axis=1) + 1.0)
    profile = np.column_stack([
        np.log1p(mean52),
        np.log1p(mean13),
        (recent52 == 0).mean(axis=1),
        np.log1p(recent13.std(axis=1)),
        np.clip(np.log(trend), -2, 2),
        np.clip((recent4.mean(axis=1) + 1) / (mean13 + 1), 0, 5),
    ])
    assert np.isfinite(profile).all()
    return profile


def fit_frozen_clusters(history: pd.DataFrame, n_clusters: int = 8,
                        seed: int = 2026) -> tuple[StandardScaler, KMeans]:
    if n_clusters < 2 or n_clusters > len(history):
        raise ValueError("Cluster count must be between 2 and number of series")
    # Every model-fitting origin starts at >=65 weeks. Fit clusters once on
    # that earliest observed period, never on future data or evaluation labels.
    profile = demand_profile(history, START_TRAIN_WEEK)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(profile)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    km.fit(scaled)
    return scaler, km


def group_feature_frame(history: pd.DataFrame, origin: int,
                        scaler: StandardScaler, km: KMeans) -> pd.DataFrame:
    base = _features(history, origin)
    values = history.iloc[:, len(KEYS):len(KEYS) + origin].to_numpy(dtype=np.float64)
    if values.shape[1] != origin:
        raise ValueError("Forecast origin exceeds available history")
    n = len(history)

    ids = history.loc[:, ["Client", "Warehouse"]]
    _, cw = np.unique(ids.to_numpy(), axis=0, return_inverse=True)
    # Stable identifier-only codes; no sales targets enter this transform.
    cw = cw.astype(np.int32)
    cluster = km.predict(scaler.transform(demand_profile(history, origin)))
    levels13 = values[:, -13:].mean(axis=1)
    levels4 = values[:, -4:].mean(axis=1)
    counts = np.bincount(cw)
    grouped13 = np.bincount(cw, weights=levels13)
    grouped4 = np.bincount(cw, weights=levels4)
    peer13 = (grouped13[cw] - levels13) / np.maximum(counts[cw] - 1, 1)
    peer4 = (grouped4[cw] - levels4) / np.maximum(counts[cw] - 1, 1)
    peer13 = np.maximum(peer13, 0.0)
    peer4 = np.maximum(peer4, 0.0)
    warehouse = history["Warehouse"].to_numpy()
    _, widx = np.unique(warehouse, return_inverse=True)
    warehouse13 = np.bincount(widx, weights=levels13)[widx]
    warehouse4 = np.bincount(widx, weights=levels4)[widx]

    for name, array in {
        "client_warehouse": cw,
        "demand_cluster": cluster,
        "cw_group_size": counts[cw],
        "cw_peer_mean13": peer13,
        "cw_peer_mean4": peer4,
        "cw_peer_growth": (peer4 + 1) / (peer13 + 1),
        "cw_peer_relative_level": (levels13 + 1) / (peer13 + 1),
        "warehouse_total_mean13": warehouse13,
        "warehouse_total_mean4": warehouse4,
        "warehouse_total_growth": (warehouse4 + 5) / (warehouse13 + 5),
    }.items():
        base[name] = np.repeat(np.asarray(array, dtype=np.float32), HORIZON)
    if not np.isfinite(base.to_numpy()).all():
        raise ValueError("Group feature frame contains invalid values")
    return base


def grouped_forecast(history: pd.DataFrame, *, trees: int = 160,
                     clusters: int = 8) -> tuple[pd.DataFrame, dict]:
    if trees < 1:
        raise ValueError("trees must be positive")
    scaler, km = fit_frozen_clusters(history, n_clusters=clusters)
    periods = history.shape[1] - len(KEYS)
    origins = _training_origins(periods)
    if not origins or any(origin + HORIZON > periods for origin in origins):
        raise ValueError("Not enough completed 13-week training windows")
    sales = history.iloc[:, len(KEYS):].to_numpy(dtype=np.float32)
    features, targets = [], []
    for origin in origins:
        X = group_feature_frame(history, origin, scaler, km)
        y = sales[:, origin:origin + HORIZON].reshape(-1)
        assert len(X) == len(y), "Training pairs must align"
        features.append(X)
        targets.append(y)
    X = pd.concat(features, ignore_index=True)
    y = np.concatenate(targets)
    params = dict(LGBM_PARAMS, n_estimators=trees)
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y, categorical_feature=GROUP_CATEGORICAL)
    X_future = group_feature_frame(history, periods, scaler, km)
    prediction = np.maximum(0, model.predict(X_future)).reshape(len(history), HORIZON)
    out = statistical_forecast(history)
    out.iloc[:, len(KEYS):] = prediction
    metadata = {
        "completed_training_origins": origins,
        "frozen_cluster_training_cutoff": START_TRAIN_WEEK,
        "num_clusters": clusters,
        "categorical_features": GROUP_CATEGORICAL,
        "rows": len(history),
        "trees": trees,
    }
    return out, metadata


def run(phase: int, data_dir: Path, output: Path, *,
        trees: int = 160, clusters: int = 8) -> dict:
    history = read_history(phase, data_dir)
    predictions, metadata = grouped_forecast(history, trees=trees, clusters=clusters)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)
    result = {"phase": phase, "output": str(output), **metadata}
    actual_path = data_dir / f"Phase {phase} - Sales.csv"
    if actual_path.exists():
        actual = read_sales(actual_path)
        result["grouped_lgbm"] = vn1_score(predictions, actual)
        result["statistical_control"] = vn1_score(statistical_forecast(history), actual)
        result["notes"] = "Retrospective comparison; phase labels only read after predictions."
    else:
        result["notes"] = "No phase labels available; no evaluation score claimed."
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="VN1 client×warehouse and frozen demand-cluster LightGBM")
    parser.add_argument("--phase", choices=(1, 2), type=int, default=1)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--trees", type=int, default=160)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("VN1_grouped.csv"))
    args = parser.parse_args()
    print(json.dumps(run(args.phase, args.data_dir, args.output,
                         trees=args.trees, clusters=args.clusters), indent=2))


if __name__ == "__main__":
    main()
