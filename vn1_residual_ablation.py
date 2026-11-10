"""Point-in-time client/warehouse residual correction and categorical ablations.

This research module *never changes the production VN1 forecast path*. It
fits out-of-fold errors using only fully observed historical 13-week horizons.
Each validation fold is scored with correction parameters fitted on strictly
earlier folds. Published competition targets are consulted solely for a
retrospective final score, never for correction training or selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from vn1_from_scratch import KEYS, HORIZON, vn1_score, read_sales, forecast as statistical_forecast
from vn1_lgbm_ensemble import (
    CATEGORICAL, LGBM_PARAMS, _features, _training_origins, read_history,
)

ABLATIONS = ("baseline", "no_ids", "no_group_stats", "no_ids_or_stats", "interactions")
STRENGTHS = (0.0, 0.5, 1.0)


def _interaction_columns(history: pd.DataFrame, origin: int) -> dict[str, np.ndarray]:
    """Client×warehouse codes and leave-one-out peer statistics at origin."""
    if origin < 65 or origin > history.shape[1] - len(KEYS):
        raise ValueError("Interaction origin is outside observed history")
    ids = history.loc[:, ["Client", "Warehouse"]].to_numpy()
    _, group = np.unique(ids, axis=0, return_inverse=True)
    counts = np.bincount(group)
    values = history.iloc[:, len(KEYS):len(KEYS) + origin].to_numpy(dtype=float)
    mean4, mean13 = values[:, -4:].mean(axis=1), values[:, -13:].mean(axis=1)
    total4 = np.bincount(group, weights=mean4)
    total13 = np.bincount(group, weights=mean13)
    peers = np.maximum(counts[group] - 1, 1)
    peer4 = np.maximum(0, (total4[group] - mean4) / peers)
    peer13 = np.maximum(0, (total13[group] - mean13) / peers)
    # Singleton groups have no peers; do not use the series itself as a peer.
    singleton = counts[group] <= 1
    peer4[singleton], peer13[singleton] = 0.0, 0.0
    return {
        "client_warehouse_id": group.astype(float),
        "client_warehouse_count": counts[group].astype(float),
        "client_warehouse_peer_mean4": peer4,
        "client_warehouse_peer_mean13": peer13,
        "client_warehouse_peer_growth": (peer4 + 1) / (peer13 + 1),
    }


def ablation_features(history: pd.DataFrame, origin: int,
                      variant: str = "baseline") -> tuple[pd.DataFrame, list[str]]:
    """Return one strictly historical feature set and its categorical columns."""
    if variant not in ABLATIONS:
        raise ValueError(f"Unknown ablation: {variant}")
    X = _features(history, origin)
    categorical = list(CATEGORICAL)
    if variant == "interactions":
        for name, values in _interaction_columns(history, origin).items():
            X[name] = np.repeat(values.astype(np.float32), HORIZON)
        categorical.append("client_warehouse_id")
    if variant in ("no_ids", "no_ids_or_stats"):
        X = X.drop(columns=["Client", "Warehouse"])
        categorical = []
    if variant in ("no_group_stats", "no_ids_or_stats"):
        group_columns = [
            c for c in X if c.startswith(("client_", "warehouse_", "product_"))
        ]
        # The statistical baseline itself contains client-pooled seasonal
        # and January shock features, so it is also group information.
        X = X.drop(columns=group_columns + ["statistical_baseline"])
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Feature set contains nonfinite values")
    return X, categorical


def fit_predict(history: pd.DataFrame, *, trees: int = 160,
                variant: str = "baseline") -> tuple[np.ndarray, list[int]]:
    """Fit a direct 13-horizon LightGBM using only completed training labels."""
    if trees < 1:
        raise ValueError("trees must be positive")
    periods = history.shape[1] - len(KEYS)
    origins = _training_origins(periods)
    if not origins or any(t + HORIZON > periods for t in origins):
        raise ValueError("No completed LightGBM training windows")
    values = history.iloc[:, len(KEYS):].to_numpy(dtype=np.float32)
    blocks, labels = [], []
    for t in origins:
        X, categories = ablation_features(history, t, variant)
        blocks.append(X)
        labels.append(values[:, t:t + HORIZON].reshape(-1))
    train = pd.concat(blocks, ignore_index=True)
    y = np.concatenate(labels)
    if len(train) != len(y):
        raise AssertionError("Misaligned direct-horizon training labels")
    params = dict(LGBM_PARAMS, n_estimators=trees)
    model = lgb.LGBMRegressor(**params)
    model.fit(train, y, categorical_feature=categories)
    future, _ = ablation_features(history, periods, variant)
    forecast = np.maximum(0, model.predict(future)).reshape(len(history), HORIZON)
    if not np.isfinite(forecast).all():
        raise ValueError("Nonfinite LightGBM predictions")
    return forecast, origins


def historical_origins(history: pd.DataFrame, max_folds: int = 6) -> list[int]:
    """Up to six fully observed, nonoverlapping 13-week validation windows."""
    if max_folds < 3:
        raise ValueError("Need at least three rolling windows")
    periods = history.shape[1] - len(KEYS)
    origins = list(range(105, periods - HORIZON + 1, HORIZON))
    if len(origins) < 3:
        raise ValueError("Need at least three completed historical windows")
    return origins[-max_folds:]


def group_correction(ids: pd.DataFrame, pred: np.ndarray,
                     actual: np.ndarray, *, shrink_client: float = 5000.0,
                     shrink_warehouse: float = 15000.0,
                     max_change: float = 0.10) -> np.ndarray:
    """Estimate bounded shrinkage-corrected relative error by historical group.

    Inputs are completed *out-of-fold* prediction/actual pairs with shape
    (folds, series, horizon). Returned factors are per series (n,). Group
    corrections use historical sum(actual-pred), damped by predicted demand
    plus an explicit prior. Unknown groups are neutral when history absent.
    """
    p, y = np.asarray(pred, dtype=float), np.asarray(actual, dtype=float)
    if p.ndim != 3 or y.shape != p.shape or p.shape[1] != len(ids):
        raise ValueError("Residual panel shape must be folds×series×horizon")
    if p.shape[0] == 0 or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Residual panel must have finite completed folds")
    if (p < 0).any() or (y < 0).any() or shrink_client <= 0 or shrink_warehouse <= 0:
        raise ValueError("Nonnegative demand and positive priors required")
    if not 0 < max_change <= 0.25:
        raise ValueError("max_change must be in (0, 0.25]")
    residual = (y - p).sum(axis=(0, 2))
    predicted = p.sum(axis=(0, 2))
    effects = []
    for key, prior in (("Client", shrink_client), ("Warehouse", shrink_warehouse)):
        _, codes = np.unique(ids[key].to_numpy(), return_inverse=True)
        errors = np.bincount(codes, weights=residual)
        volume = np.bincount(codes, weights=predicted)
        effect = errors[codes] / (volume[codes] + prior)
        effects.append(np.clip(effect, -max_change, max_change))
    factor = 1.0 + 0.5 * effects[0] + 0.5 * effects[1]
    return np.clip(factor, 1.0 - max_change, 1.0 + max_change)


def apply_correction(forecast: np.ndarray, factor: np.ndarray,
                     strength: float) -> np.ndarray:
    """Apply a fractional group bias correction without changing row order."""
    p = np.asarray(forecast, dtype=float)
    f = np.asarray(factor, dtype=float)
    if p.ndim != 2 or f.shape != (len(p),):
        raise ValueError("Expected (series, horizon) predictions and per-series factors")
    if not (0 <= strength <= 1) or not np.isfinite(strength):
        raise ValueError("strength must lie in [0, 1]")
    return np.maximum(0.0, p * (1.0 + strength * (f[:, None] - 1.0)))


def _official_loss(pred: np.ndarray, actual: np.ndarray) -> float:
    err = np.asarray(pred, dtype=float) - np.asarray(actual, dtype=float)
    denom = float(np.asarray(actual).sum())
    if denom <= 0:
        raise ValueError("Evaluation labels must have positive total sales")
    return float((np.abs(err).sum() + abs(err.sum())) / denom)


def select_strength(ids: pd.DataFrame, forecasts: np.ndarray,
                    targets: np.ndarray, *, min_gain: float = 0.005,
                    origin_dates: list[str] | None = None,
                    final_month: int | None = None) -> dict:
    """Nested rolling selection: train correction only on preceding folds.

    At validation fold j, residual factors use prediction/actual pairs from
    folds [0:j]. Evaluate original and candidate corrections on fold j.
    Pool all validation errors with the official global-bias penalty.
    A correction must win >=2 folds, deliver >=0.005 absolute VN1 gain,
    and avoid >0.02 deterioration on *any* holdout. January final origins
    also require improvement on a strictly historical January fold.
    """
    p, y = np.asarray(forecasts, dtype=float), np.asarray(targets, dtype=float)
    if p.shape != y.shape or p.ndim != 3 or p.shape[0] < 3 or p.shape[1] != len(ids):
        raise ValueError("Need at least 3 completed historical forecast/target folds")
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite historical validation panel")
    if origin_dates is not None and len(origin_dates) != len(p):
        raise ValueError("origin_dates must align with historical folds")
    trials = {strength: [] for strength in STRENGTHS}
    for j in range(2, len(p)):
        f = group_correction(ids, p[:j], y[:j])
        for strength in STRENGTHS:
            result = apply_correction(p[j], f, strength)
            trials[strength].append({"fold": j, "pred": result, "target": y[j]})
    pooled = {}
    fold_scores = {}
    for strength, entries in trials.items():
        predicted = np.concatenate([e["pred"].reshape(-1) for e in entries])
        observed = np.concatenate([e["target"].reshape(-1) for e in entries])
        pooled[str(strength)] = _official_loss(predicted, observed)
        fold_scores[str(strength)] = [_official_loss(e["pred"], e["target"]) for e in entries]
    selected = 0.0
    holdouts = len(p) - 2
    for strength in (0.5, 1.0):
        name = str(strength)
        baseline = fold_scores["0.0"]
        current = fold_scores[name]
        wins = sum(a + 1e-10 < b for a, b in zip(current, baseline))
        worst_regression = max(a - b for a, b in zip(current, baseline))
        if (holdouts >= 2 and wins >= max(2, int(np.ceil(0.6 * holdouts)))
                and worst_regression <= 0.02
                and pooled[name] < pooled["0.0"] - min_gain):
            if final_month == 1:
                if origin_dates is None:
                    continue
                matching = [
                    j for j in range(2, len(p))
                    if pd.Timestamp(origin_dates[j]).month == 1
                ]
                if not matching or not all(
                    current[j - 2] < baseline[j - 2] - min_gain
                    for j in matching
                ):
                    continue
            if selected == 0.0 or pooled[name] < pooled[str(selected)]:
                selected = strength
    return {
        "selected_strength": selected,
        "pooled_scores": pooled,
        "fold_scores": fold_scores,
        "number_of_holdouts": holdouts,
        "selection_only_historical": True,
        "min_gain": min_gain,
        "january_gate": final_month == 1,
    }


def rolling_experiment(history: pd.DataFrame, *, trees: int = 160,
                       max_folds: int = 6, ablation_folds: int = 2) -> tuple[
                           np.ndarray, np.ndarray, dict]:
    """Compute nested residual selector and category ablation diagnostics."""
    origins = historical_origins(history, max_folds)
    values = history.iloc[:, len(KEYS):].to_numpy(dtype=float)
    historical_prediction, historical_truth = [], []
    for origin in origins:
        past = history.iloc[:, :len(KEYS) + origin]
        forecast, _ = fit_predict(past, trees=trees)
        historical_prediction.append(forecast)
        historical_truth.append(values[:, origin:origin + HORIZON])
    pred = np.stack(historical_prediction)
    target = np.stack(historical_truth)
    dates = [str(history.columns[len(KEYS) + o - 1]) for o in origins]
    final_month = pd.Timestamp(history.columns[-1]).month
    selection = select_strength(history.loc[:, ["Client", "Warehouse"]],
                                pred, target, origin_dates=dates,
                                final_month=final_month)
    final, train_origins = fit_predict(history, trees=trees)
    factor = group_correction(history.loc[:, ["Client", "Warehouse"]], pred, target)
    corrected = apply_correction(final, factor, selection["selected_strength"])

    if ablation_folds < 1:
        raise ValueError("ablation_folds must be at least 1")
    ablation_origins = origins[-ablation_folds:]
    ablations = {}
    for variant in ABLATIONS:
        fold_scores = []
        for origin in ablation_origins:
            actual = values[:, origin:origin + HORIZON]
            if variant == "baseline":
                forecast = pred[origins.index(origin)]
            else:
                past = history.iloc[:, :len(KEYS) + origin]
                forecast, _ = fit_predict(past, trees=trees, variant=variant)
            fold_scores.append(_official_loss(forecast, actual))
        ablations[variant] = {
            "origins": ablation_origins, "fold_scores": fold_scores,
            "mean_fold_score": float(np.mean(fold_scores)),
        }
    return final, corrected, {
        "cv_origins": origins, "fold_end_weeks": [x + HORIZON for x in origins],
        "final_history_weeks": history.shape[1] - len(KEYS),
        "selection": selection,
        "train_origins": train_origins,
        "ablation_results": ablations,
        "features_tuned_with_phase_targets": False,
    }


def run(phase: int, data_dir: Path, *, trees: int = 160, max_folds: int = 6,
        ablation_folds: int = 2, output: Path | None = None) -> dict:
    history = read_history(phase, data_dir)
    plain, corrected, report = rolling_experiment(
        history, trees=trees, max_folds=max_folds, ablation_folds=ablation_folds)
    out_path = output or Path(f"VN1_phase{phase}_residual_research.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred = statistical_forecast(history)
    pred.iloc[:, len(KEYS):] = corrected
    pred.to_csv(out_path, index=False)
    receipt = {
        "phase": phase, "trees": trees, "csv": str(out_path),
        "report": report,
        "notes": "Research-only. Incumbent production policy untouched; phase target not used for selection.",
    }
    actual_path = data_dir / f"Phase {phase} - Sales.csv"
    if actual_path.exists():
        labels = read_sales(actual_path)
        raw = pred.copy()
        raw.iloc[:, len(KEYS):] = plain
        receipt["plain_lgbm_score"] = vn1_score(raw, labels)["vn1_score"]
        receipt["residual_corrected_score"] = vn1_score(pred, labels)["vn1_score"]
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="VN1 client/warehouse residual research")
    parser.add_argument("--phase", type=int, choices=(1, 2), default=1)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--trees", type=int, default=160)
    parser.add_argument("--max-folds", type=int, default=6)
    parser.add_argument("--ablation-folds", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.phase, args.data_dir, trees=args.trees,
                         max_folds=args.max_folds,
                         ablation_folds=args.ablation_folds,
                         output=args.output), indent=2))


if __name__ == "__main__":
    main()
