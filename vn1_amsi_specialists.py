"""Experimental anchored seasonality and extensible StatsForecast specialists for VN1.

A competition-inspired candidate library, NOT a reproduction of the winners'
private implementation. All model features come from the supplied history.
Weights are chosen on completed Phase-0 rolling windows; the newest window
is a separate acceptance gate. Competition Phase 1/2 labels are never used
to choose model names or weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vn1_from_scratch import HORIZON, KEYS, forecast as statistical_forecast, read_sales, vn1_score
from vn1_lgbm_ensemble import choose_weight, ensemble_forecast, read_history

# Fixed configurations, intentionally small to reduce selection overfitting.
AMSI_CONFIGS = {
    "amsi_last": ("last", 0.45, 0.55),
    "amsi_short": ("short", 0.60, 0.55),
    "amsi_mean13": ("mean13", 0.65, 0.70),
}
# Opt-in local statistical models; no arbitrary imports or user-supplied code.
STATS_MODELS = (
    "autoarima52", "autoets52", "dynamictheta",
    "autoces", "seasonalnaive52",
)
TRAIN_WEIGHTS = (0.10, 0.25, 0.50)


def _history_values(history: pd.DataFrame) -> np.ndarray:
    if not set(KEYS).issubset(history) or list(history.columns[:3]) != KEYS:
        raise ValueError("VN1 identifier columns must be the first three columns")
    if history.duplicated(KEYS).any():
        raise ValueError("Duplicated series identifiers")
    y = history.iloc[:, len(KEYS):].to_numpy(dtype=np.float64)
    if y.ndim != 2 or y.shape[1] < 65 or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Need at least 65 finite nonnegative historical weeks")
    return y


def amsi_forecast(history: pd.DataFrame, *, anchor: str = "short",
                  seasonality: float = 0.60, group_shrink: float = 0.55
                  ) -> np.ndarray:
    """Historical 52-week multiplicative profile anchored to recent demand.

    The forecasted 13 weeks reuse the weeks 52 periods in the past, normalized
    against the *preceding* 13 observed weeks. Group seasonality is aggregated
    by client, with a smoothed per-product contribution. No future data enters.
    """
    y = _history_values(history)
    if anchor not in ("last", "short", "mean13"):
        raise ValueError("Unknown anchor")
    if not (0 <= seasonality <= 1 and 0 <= group_shrink <= 1):
        raise ValueError("Seasonality and group_shrink must be in [0,1]")
    last = y[:, -1]
    mean4 = y[:, -4:].mean(axis=1)
    mean13 = y[:, -13:].mean(axis=1)
    if anchor == "last":
        level = last
    elif anchor == "short":
        level = 0.25 * last + 0.50 * mean4 + 0.25 * mean13
    else:
        level = mean13

    prior = y[:, -65:-52].mean(axis=1)
    matched = y[:, -52:-39]  # 52-week lag of the *future* 13 weeks
    if matched.shape[1] != HORIZON:
        raise AssertionError("Incomplete annual matched window")
    series_ratio = (matched + 3.0) / (prior[:, None] + 3.0)
    series_ratio = np.clip(series_ratio, 0.3, 2.5)
    reliability = prior / (prior + 8.0)

    _, client_idx = np.unique(history["Client"].to_numpy(), return_inverse=True)
    n_clients = int(client_idx.max()) + 1
    group_before = np.bincount(client_idx, weights=prior, minlength=n_clients)
    group_matched = np.zeros((n_clients, HORIZON), dtype=np.float64)
    np.add.at(group_matched, client_idx, matched)
    group_ratio = (group_matched + 10.0) / (group_before[:, None] + 10.0)
    group_ratio = np.clip(group_ratio, 0.4, 2.2)[client_idx]

    # Shrink sparse product factors toward the client and both toward 1.
    product_factor = 1 + reliability[:, None] * (series_ratio - 1)
    combined = ((1 - group_shrink) * product_factor
                + group_shrink * group_ratio)
    seasonal = 1 + seasonality * (combined - 1)
    return np.maximum(0, level[:, None] * np.clip(seasonal, 0.4, 2.2))


def make_stats_model(name: str):
    """Closed registry of supported Nixtla models, constructed on demand."""
    if name not in STATS_MODELS:
        raise ValueError(f"Unsupported StatsForecast model: {name}")
    from statsforecast import models as sm
    if name == "autoarima52":
        return sm.AutoARIMA(season_length=52, max_p=1, max_q=1,
                            max_P=1, max_Q=1, start_p=1, start_q=1,
                            nmodels=12, stepwise=True, approximation=True)
    if name == "autoets52":
        return sm.AutoETS(season_length=52, model="ZZZ")
    if name == "dynamictheta":
        return sm.DynamicOptimizedTheta(season_length=1)
    if name == "autoces":
        return sm.AutoCES(season_length=1)
    return sm.SeasonalNaive(season_length=52)


def statsforecast_specialist(history: pd.DataFrame, name: str, *,
                              cap: int = 32, fallback: np.ndarray | None = None
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Fit a bounded number of active, high-volume series only.

    All other series retain the supplied fallback, including failed fits.
    The deterministic eligibility rule uses only observations before origin.
    """
    y = _history_values(history)
    if name not in STATS_MODELS:
        raise ValueError("Unknown statistical model")
    if not 0 <= cap <= 512:
        raise ValueError("cap must be between 0 and 512")
    output = (statistical_forecast(history).iloc[:, 3:].to_numpy(dtype=np.float64)
              if fallback is None else np.asarray(fallback, dtype=np.float64).copy())
    if output.shape != (len(y), HORIZON) or not np.isfinite(output).all():
        raise ValueError("Fallback forecast has wrong shape or nonfinite values")
    mask = np.zeros(len(y), dtype=bool)
    if cap == 0:
        return output, mask
    recent = y[:, -52:]
    volume = recent.mean(axis=1)
    activity = (recent > 0).mean(axis=1)
    eligible = np.flatnonzero((volume >= 5) & (activity >= 0.6))
    eligible = eligible[np.argsort(-volume[eligible], kind="stable")[:cap]]
    for index in eligible:
        model = make_stats_model(name)
        try:
            result = model.forecast(y=np.ascontiguousarray(y[index]), h=HORIZON)
            point = np.asarray(result["mean"], dtype=np.float64)
            if point.shape == (HORIZON,) and np.isfinite(point).all():
                output[index] = np.maximum(0, point)
                mask[index] = True
        except (ValueError, ArithmeticError, FloatingPointError, TypeError,
                ZeroDivisionError, np.linalg.LinAlgError):
            continue
    return output, mask


def candidate_forecasts(history: pd.DataFrame, *, stats_models: tuple[str, ...] = (),
                        stats_cap: int = 32) -> tuple[dict[str, np.ndarray], dict]:
    candidates = {}
    for name, (anchor, seasonality, shrink) in AMSI_CONFIGS.items():
        candidates[name] = amsi_forecast(history, anchor=anchor,
                                         seasonality=seasonality, group_shrink=shrink)
    metadata = {"series": len(history), "stats_fitted": {}}
    fallback = statistical_forecast(history).iloc[:, 3:].to_numpy(dtype=float)
    for name in stats_models:
        if name not in STATS_MODELS:
            raise ValueError(f"Unknown model {name}")
        pred, mask = statsforecast_specialist(history, name, cap=stats_cap,
                                               fallback=fallback)
        candidates[name] = pred
        metadata["stats_fitted"][name] = int(mask.sum())
    return candidates, metadata


def _score(pred: np.ndarray, actual: np.ndarray) -> float:
    if pred.shape != actual.shape or pred.ndim != 2:
        raise ValueError("Forecast / actual shape mismatch")
    if actual.sum() <= 0:
        raise ValueError("Actual sales total must be positive")
    e = pred - actual
    return float((np.abs(e).sum() + abs(e.sum())) / actual.sum())


def historical_origins(history: pd.DataFrame) -> list[int]:
    periods = history.shape[1] - len(KEYS)
    origins = list(range(105, periods - HORIZON + 1, HORIZON))
    if len(origins) < 3:
        raise ValueError("Need at least three completed 13-week validation origins")
    return origins


def _champion(history: pd.DataFrame, estimators: int) -> np.ndarray:
    w = choose_weight(history, n_estimators=estimators)["weight_lgbm"]
    return ensemble_forecast(history, weight_lgbm=w,
                             n_estimators=estimators).iloc[:, 3:].to_numpy(dtype=float)


def choose_challenger(history: pd.DataFrame, *,
                      stats_models: tuple[str, ...] = (), stats_cap: int = 32,
                      estimators: int = 160, min_gain: float = 0.002,
                      champion_fn=None) -> dict:
    """Pre-holdout grid search, then separate last-origin acceptance gate.

    The last historical fold only decides whether to keep a challenger
    selected using *earlier* completed folds; it never optimizes its weight.
    This is a conservative research gate, not independent blind scoring.
    """
    if not 0 <= min_gain <= 1:
        raise ValueError("min_gain must be in [0,1]")
    actual = _history_values(history)
    origins = historical_origins(history)
    baseline_fn = champion_fn or (lambda h: _champion(h, estimators))
    fold_data = []
    for origin in origins:
        past = history.iloc[:, :len(KEYS) + origin]
        control = np.asarray(baseline_fn(past), dtype=np.float64)
        candidates, _ = candidate_forecasts(past, stats_models=stats_models,
                                             stats_cap=stats_cap)
        target = actual[:, origin:origin + HORIZON]
        if control.shape != target.shape or not np.isfinite(control).all():
            raise ValueError("Champion shape or values invalid")
        fold_data.append((control, candidates, target))
    names = list(fold_data[0][1])
    prior = fold_data[:-1]
    def pooled_error(name: str | None, weight: float) -> float:
        errors, total = [], 0.0
        for control, candidates, y in prior:
            pred = control if name is None else (
                (1 - weight) * control + weight * candidates[name])
            errors.append(pred - y)
            total += y.sum()
        e = np.concatenate([e.reshape(-1) for e in errors])
        return float((np.abs(e).sum() + abs(e.sum())) / total)
    baseline_train = pooled_error(None, 0)
    best = {"model": "champion", "weight": 0.0,
            "train": baseline_train}
    for name in names:
        for weight in TRAIN_WEIGHTS:
            score = pooled_error(name, weight)
            if score < best["train"]:
                best = {"model": name, "weight": weight, "train": score}
    control, candidates, target = fold_data[-1]
    baseline_holdout = _score(control, target)
    attempted = best.copy()
    if best["model"] == "champion":
        challenger_holdout = baseline_holdout
    else:
        challenger = ((1 - best["weight"]) * control
                      + best["weight"] * candidates[best["model"]])
        challenger_holdout = _score(challenger, target)
    accept = (best["model"] != "champion"
              and best["train"] < baseline_train - min_gain
              and challenger_holdout < baseline_holdout - min_gain)
    return {
        "candidate": attempted, "selected_model": best["model"] if accept else "champion",
        "selected_weight": best["weight"] if accept else 0.0,
        "train_champion": baseline_train,
        "holdout_champion": baseline_holdout,
        "holdout_candidate": challenger_holdout,
        "holdout_origin": origins[-1],
        "training_origins": origins[:-1],
        "accepted": bool(accept),
        "stats_models": list(stats_models),
    }


def run(phase: int, data_dir: Path, output: Path, *, stats_models: tuple[str, ...] = (),
        stats_cap: int = 32, estimators: int = 160) -> dict:
    history = read_history(phase, data_dir)
    selected = choose_challenger(history, stats_models=stats_models,
                                 stats_cap=stats_cap, estimators=estimators)
    control = _champion(history, estimators)
    output_values = control.copy()
    if selected["accepted"]:
        name = selected["selected_model"]
        weight = selected["selected_weight"]
        challenger, _ = candidate_forecasts(history, stats_models=stats_models,
                                             stats_cap=stats_cap)
        output_values = (1 - weight) * control + weight * challenger[name]
    result = statistical_forecast(history)
    result.iloc[:, 3:] = np.maximum(output_values, 0)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    metrics = {"phase": phase, "selected": selected,
               "forecast": str(output), "series": len(result),
               "horizon": HORIZON, "champion_changed": selected["accepted"]}
    actual_path = data_dir / f"Phase {phase} - Sales.csv"
    if actual_path.exists():
        objective = read_sales(actual_path)
        metrics["retrospective"] = vn1_score(result, objective)
        # Public targets are used for reporting ONLY, never for selection.
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="VN1 AMSI / local StatsForecast research")
    parser.add_argument("--phase", type=int, choices=(1, 2), default=1)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("VN1_amsi_experiment.csv"))
    parser.add_argument("--stats-models", nargs="*", choices=STATS_MODELS, default=[])
    parser.add_argument("--stats-cap", type=int, default=32)
    parser.add_argument("--n-estimators", type=int, default=160)
    args = parser.parse_args()
    print(json.dumps(run(args.phase, args.data_dir, args.output,
                         stats_models=tuple(args.stats_models),
                         stats_cap=args.stats_cap, estimators=args.n_estimators),
                     indent=2))


if __name__ == "__main__":
    main()
