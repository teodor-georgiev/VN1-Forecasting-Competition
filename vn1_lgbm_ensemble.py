"""LightGBM direct multi-horizon forecaster and validated VN1 statistical ensemble.

The model is trained only on (origin, horizon) pairs whose sales targets are
fully observed at the training cutoff. The weight of the LightGBM component
is selected from a prior year's season-matched validation origin, not the
competition target labels being scored.

Dependencies: numpy, pandas, lightgbm. Reproduce on the official CSVs:
    python vn1_lgbm_ensemble.py --phase 1
    python vn1_lgbm_ensemble.py --phase 2 --data-dir /path/to/data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from vn1_from_scratch import HORIZON, KEYS, forecast as statistical_forecast, read_sales, vn1_score

START_TRAIN_WEEK = 65
TRAIN_STRIDE = 13
LGBM_PARAMS = dict(
    objective="tweedie",
    tweedie_variance_power=1.3,
    n_estimators=160,
    num_leaves=31,
    learning_rate=0.06,
    min_child_samples=100,
    max_bin=127,
    reg_lambda=3,
    colsample_bytree=0.9,
    subsample=0.9,
    subsample_freq=1,
    random_state=2026,
    n_jobs=2,
    verbosity=-1,
)
CATEGORICAL = ["Client", "Warehouse"]


def _as_history(data: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """Enforce strictly historical features for a training/prediction origin."""
    if cutoff < START_TRAIN_WEEK or cutoff > len(data.columns) - len(KEYS):
        raise ValueError("Origin not within the available history")
    return data.iloc[:, :len(KEYS) + cutoff]


def _category_stats(values: np.ndarray, ids: np.ndarray, cutoff: int) -> dict[str, np.ndarray]:
    _, idx = np.unique(ids, return_inverse=True)
    cardinality = int(idx.max()) + 1
    counts = np.bincount(idx, minlength=cardinality)
    features = {}
    for weeks, column in [(1, "last1"), (4, "mean4"), (13, "mean13")]:
        level = values[:, cutoff - weeks:cutoff].mean(axis=1)
        totals = np.bincount(idx, weights=level, minlength=cardinality)
        # Exclude the target series to reduce trivial in-sample signal.
        leave_one_out = (totals[idx] - level) / np.maximum(counts[idx] - 1, 1)
        features[column] = np.maximum(0.0, leave_one_out).astype("float32")
    return features


def _features(data: pd.DataFrame, origin: int) -> pd.DataFrame:
    """Build n_series * 13 rows in series-major / horizon-minor order."""
    hist = _as_history(data, origin)
    val = hist.iloc[:, len(KEYS):].to_numpy(dtype=np.float32, copy=False)
    n = val.shape[0]
    ids = hist[KEYS]
    weeks = pd.to_datetime(hist.columns[len(KEYS):]); last_date = weeks[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=HORIZON, freq="7D")
    d = {}

    def base(name: str, arr: np.ndarray) -> None:
        d[name] = np.repeat(np.asarray(arr, dtype=np.float32), HORIZON)

    for key in CATEGORICAL:
        base(key, ids[key].to_numpy(dtype=np.float32))
    for lag in [1, 2, 3, 4, 8, 13, 26, 52]:
        base(f"lag_{lag}", val[:, -lag])
    for w in [4, 8, 13, 26, 52]:
        v = val[:, -w:]
        base(f"mean_{w}", v.mean(axis=1))
        if w in [4, 13, 26]:
            base(f"std_{w}", v.std(axis=1))
            base(f"zero_share_{w}", (v == 0).mean(axis=1))
    base("max_13", val[:, -13:].max(axis=1))
    base("nonzero_13", (val[:, -13:] > 0).sum(axis=1))
    mean13 = val[:, -13:].mean(axis=1)
    prev13 = val[:, -26:-13].mean(axis=1)
    base("recent_trend", (mean13 + 1) / (prev13 + 1))
    base("recent_activity", val[:, -4:].sum(axis=1) / (val[:, -13:].sum(axis=1) + 1))
    base("mean_to_peak", mean13 / (val[:, -13:].max(axis=1) + 1))

    for kind in ["Client", "Warehouse", "Product"]:
        group = _category_stats(val, ids[kind].to_numpy(), origin)
        for name, v in group.items():
            base(f"{kind.lower()}_{name}", v)

    last_year_target = val[:, origin - 52:origin - 52 + HORIZON]
    d["last_year_target"] = last_year_target.reshape(-1).astype(np.float32)
    prior = val[:, origin - 53:origin - 53 + HORIZON]
    after = val[:, origin - 51:origin - 51 + HORIZON]
    d["last_year_smooth3"] = ((last_year_target + prior + after) / 3).reshape(-1).astype(np.float32)
    d["horizon"] = np.tile(np.arange(1, HORIZON + 1, dtype=np.float32), n)
    week_num = future_dates.isocalendar().week.to_numpy(dtype=np.float32)
    d["week_sin"] = np.tile(np.sin(2 * np.pi * week_num / 52), n).astype(np.float32)
    d["week_cos"] = np.tile(np.cos(2 * np.pi * week_num / 52), n).astype(np.float32)
    d["future_month"] = np.tile(future_dates.month.to_numpy(dtype=np.float32), n)
    d["origin_month"] = np.full(n * HORIZON, last_date.month, dtype=np.float32)
    d["future_year"] = np.tile(future_dates.year.to_numpy(dtype=np.float32) - 2020, n)
    d["origin_year"] = np.full(n * HORIZON, last_date.year - 2020, dtype=np.float32)
    d["is_january"] = np.tile((future_dates.month == 1).astype(np.float32), n)
    # Statistical baseline is a predictor, not a leakage source: same history cutoff.
    d["statistical_baseline"] = statistical_forecast(hist).iloc[:, len(KEYS):].to_numpy(dtype=np.float32).reshape(-1)
    feat = pd.DataFrame(d)
    if not np.isfinite(feat.to_numpy()).all():
        raise ValueError("Feature matrix contains nonfinite values")
    return feat


def _training_origins(available_weeks: int) -> list[int]:
    """Any label at origin+h must have occurred by the training cutoff."""
    return list(range(START_TRAIN_WEEK, available_weeks - HORIZON + 1, TRAIN_STRIDE))


def train_lgbm(history: pd.DataFrame, *, n_estimators: int = 160):
    n_weeks = history.shape[1] - len(KEYS)
    origins = _training_origins(n_weeks)
    if not origins:
        raise ValueError("Insufficient history for a full 13-week direct-training target")
    features = []
    targets = []
    values = history.iloc[:, len(KEYS):].to_numpy(dtype=np.float32)
    for origin in origins:
        features.append(_features(history, origin))
        targets.append(values[:, origin:origin + HORIZON].reshape(-1))
    X = pd.concat(features, ignore_index=True)
    y = np.concatenate(targets)
    assert X.shape[0] == y.shape[0]
    assert max(origins) + HORIZON <= n_weeks, "No future label may enter training"
    params = {**LGBM_PARAMS, "n_estimators": n_estimators}
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y, categorical_feature=CATEGORICAL)
    return model, origins


def lgbm_forecast(history: pd.DataFrame, *, n_estimators: int = 160) -> pd.DataFrame:
    model, _ = train_lgbm(history, n_estimators=n_estimators)
    feature = _features(history, history.shape[1] - len(KEYS))
    prediction = np.maximum(0.0, model.predict(feature)).reshape(len(history), HORIZON)
    out = statistical_forecast(history)
    out.iloc[:, len(KEYS):] = prediction
    return out


def choose_weight(history: pd.DataFrame, *, n_estimators: int = 160) -> dict:
    """Season-matched pre-target CV; no Phase 1/2 target is consulted.

    End-of-September origins use 2022-10-03 (origin 117). End-of-December
    origins use 2023-01-02 (origin 130). Both origins' future targets lie
    wholly within Phase 0 and CV models are trained strictly before them.
    """
    last = pd.Timestamp(history.columns[-1])
    origin = 130 if last.month == 1 else 117
    if history.shape[1] - len(KEYS) < origin + HORIZON:
        return {"weight_lgbm": 0.5, "cv_origin": None, "cv_score": None, "reason": "no historical fold"}
    earlier = _as_history(history, origin)
    model, training_origins = train_lgbm(earlier, n_estimators=n_estimators)
    future = history.iloc[:, [0, 1, 2] + list(range(len(KEYS) + origin, len(KEYS) + origin + HORIZON))].copy()
    lgb = statistical_forecast(earlier)
    lgb.iloc[:, len(KEYS):] = np.maximum(0.0, model.predict(_features(earlier, origin))).reshape(len(earlier), HORIZON)
    stat = statistical_forecast(earlier)
    # Combine by series/horizon, preserve keys and exact date order.
    lgb_values = lgb.iloc[:, 3:].to_numpy(dtype=np.float64)
    stat_values = stat.iloc[:, 3:].to_numpy(dtype=np.float64)
    best = None
    for weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
        blended = stat.copy()
        blended.iloc[:, len(KEYS):] = np.maximum(0.0, weight * lgb_values + (1 - weight) * stat_values)
        score = vn1_score(blended, future)["vn1_score"]
        if best is None or score < best["cv_score"]:
            best = {"weight_lgbm": weight, "cv_origin": origin,
                    "cv_score": float(score), "cv_train_origins": training_origins}
    return best


def ensemble_forecast(history: pd.DataFrame, *, weight_lgbm: float,
                      n_estimators: int = 160) -> pd.DataFrame:
    if not 0.0 <= weight_lgbm <= 1.0:
        raise ValueError("Ensemble weight must be in [0, 1]")
    if weight_lgbm == 0:
        return statistical_forecast(history)
    lgb = lgbm_forecast(history, n_estimators=n_estimators)
    if weight_lgbm == 1:
        return lgb
    stat = statistical_forecast(history)
    stat.iloc[:, len(KEYS):] = np.maximum(0, weight_lgbm * lgb.iloc[:, len(KEYS):].to_numpy(dtype=np.float64)
                                             + (1 - weight_lgbm) * stat.iloc[:, len(KEYS):].to_numpy(dtype=np.float64))
    return stat


def read_history(phase: int, data_dir: Path) -> pd.DataFrame:
    phase0 = read_sales(data_dir / "Phase 0 - Sales.csv")
    if phase == 1:
        return phase0
    if phase != 2:
        raise ValueError("Phase must be 1 or 2")
    phase1 = read_sales(data_dir / "Phase 1 - Sales.csv")
    if not phase0[KEYS].equals(phase1[KEYS]):
        raise ValueError("Identifiers changed between Phase 0 and Phase 1")
    return pd.concat([phase0, phase1.iloc[:, len(KEYS):]], axis=1)


def run(phase: int, data_dir: Path, output: Path, *, n_estimators: int = 160, weight: float | None = None) -> dict:
    history = read_history(phase, data_dir)
    stat = statistical_forecast(history)
    chosen = choose_weight(history, n_estimators=n_estimators) if weight is None else {"weight_lgbm": weight, "cv_origin": None, "reason": "manual"}
    lgb = lgbm_forecast(history, n_estimators=n_estimators)
    chosen_weight = chosen["weight_lgbm"]
    result = stat.copy()
    result.iloc[:, len(KEYS):] = chosen_weight * lgb.iloc[:, len(KEYS):].to_numpy(dtype=np.float64) + (1 - chosen_weight) * stat.iloc[:, len(KEYS):].to_numpy(dtype=np.float64)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    metrics = {"phase": phase, "models": "LightGBM Tweedie + statistical", "weight_selection": chosen,
               "train_origins": _training_origins(history.shape[1] - len(KEYS)), "submission": str(output)}
    target_path = data_dir / f"Phase {phase} - Sales.csv"
    if target_path.exists():
        actual = read_sales(target_path)
        metrics["statistical"] = vn1_score(stat, actual)
        metrics["lightgbm"] = vn1_score(lgb, actual)
        metrics["ensemble"] = vn1_score(result, actual)
    else:
        metrics["note"] = "Phase target not provided, so no score is claimed"
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="VN1 LightGBM + statistical ensemble")
    parser.add_argument("--phase", choices=(1, 2), type=int, default=1)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--n-estimators", type=int, default=160)
    parser.add_argument("--weight", type=float, default=None, help="Override historical-CV-chosen LightGBM blend weight")
    args = parser.parse_args()
    out = args.output or Path(f"Submission_Phase_{args.phase}_lgbm_ensemble.csv")
    print(json.dumps(run(args.phase, args.data_dir, out, n_estimators=args.n_estimators, weight=args.weight), indent=2))


if __name__ == "__main__":
    main()
