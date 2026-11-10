"""Adaptive VN1 13-week forecasts: LightGBM variants, Dynamic Theta, intermittent
demand, hierarchical baseline, optional CatBoost, and per-product model weights.

Every training label and every model-selection target is from an already
completed historical window. Phase 2 labels are read ONLY after predictions
have been finalized. This is a retrospective benchmark, not a blind entry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from vn1_from_scratch import KEYS, HORIZON, forecast as statistical_forecast, read_sales, vn1_score
from vn1_lgbm_ensemble import _features, _training_origins, LGBM_PARAMS, CATEGORICAL, read_history

MODELS = ("statistical", "tweedie", "poisson", "sba", "tsb", "theta", "hierarchical")


def intermittent_forecasts(history: pd.DataFrame, alpha: float = 0.1,
                           beta: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Croston-SBA and TSB across independent demand series.

    Croston-SBA estimates nonzero event sizes and inter-event gaps, applying
    the (1-alpha/2) bias correction. TSB updates demand-event probability
    every period (including zero periods), allowing demand obsolescence.
    """
    y = history.iloc[:, len(KEYS):].to_numpy(dtype=np.float64)
    n, periods = y.shape
    if periods < 2 or (y < 0).any() or not np.isfinite(y).all():
        raise ValueError("Invalid historical sales")
    size = np.zeros(n)
    interval = np.ones(n)
    seen = np.zeros(n, dtype=bool)
    elapsed = np.zeros(n, dtype=np.float64)
    p = np.zeros(n)
    z = np.zeros(n)
    for t in range(periods):
        current = y[:, t]
        event = current > 0
        elapsed += 1
        first = event & ~seen
        subsequent = event & seen
        size[first] = current[first]
        interval[first] = elapsed[first]
        size[subsequent] = alpha * current[subsequent] + (1 - alpha) * size[subsequent]
        interval[subsequent] = alpha * elapsed[subsequent] + (1 - alpha) * interval[subsequent]
        elapsed[event] = 0
        seen[event] = True
        z[first] = current[first]
        z[subsequent] = beta * current[subsequent] + (1 - beta) * z[subsequent]
        if t == 0:
            p = event.astype(float)
        else:
            p = alpha * event.astype(float) + (1 - alpha) * p
    sba = (1 - alpha / 2) * size / np.maximum(interval, 1.0)
    tsb = z * p
    return (np.broadcast_to(np.maximum(sba, 0)[:, None], (n, HORIZON)).copy(),
            np.broadcast_to(np.maximum(tsb, 0)[:, None], (n, HORIZON)).copy())


def theta_forecast(history: pd.DataFrame, cap: int = 250, *,
                   fallback: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Nixtla's genuine DynamicOptimizedTheta on high-volume, active series.

    A deterministic top-volume cap avoids fitting 15,053 slow local optimizers
    on predominantly zero-demand series. Other series retain the safe
    statistical candidate, so all forecasts remain defined. Return mask
    indicating where Theta was actually computed.
    """
    from statsforecast.models import DynamicOptimizedTheta

    y = history.iloc[:, len(KEYS):].to_numpy(dtype=np.float64)
    baseline = (np.asarray(fallback, dtype=np.float64).copy()
                if fallback is not None else statistical_forecast(history).iloc[:, len(KEYS):].to_numpy(dtype=float))
    if cap < 0:
        raise ValueError("theta cap must be nonnegative")
    if cap == 0:
        return baseline, np.zeros(len(y), dtype=bool)
    recent = y[:, -52:]
    volume = recent.mean(axis=1)
    active = (recent > 0).mean(axis=1)
    eligible = np.flatnonzero((volume >= 5.0) & (active >= 0.5))
    selected = eligible[np.argsort(-volume[eligible], kind="stable")[:cap]]
    mask = np.zeros(len(y), dtype=bool)
    for i in selected:
        values = np.ascontiguousarray(y[i], dtype=np.float64)
        try:
            result = DynamicOptimizedTheta(season_length=1).forecast(y=values, h=HORIZON)
            arr = np.asarray(result["mean"], dtype=float)
            if arr.shape == (HORIZON,) and np.isfinite(arr).all():
                baseline[i] = np.clip(arr, 0, np.inf)
                mask[i] = True
        except (ValueError, ArithmeticError, FloatingPointError, TypeError):
            # Local statistical fitting may fail on degenerate series.
            continue
    return baseline, mask


def hierarchical_forecast(history: pd.DataFrame, base: np.ndarray) -> np.ndarray:
    """Client-level recent growth shrinks product forecast in the direction
    of aggregate demand; never estimates future aggregates from target labels.
    """
    values = history.iloc[:, len(KEYS):].to_numpy(dtype=float)
    clients = history.Client.to_numpy()
    _, idx = np.unique(clients, return_inverse=True)
    group4 = np.bincount(idx, weights=values[:, -4:].mean(axis=1))
    group13 = np.bincount(idx, weights=values[:, -13:].mean(axis=1))
    ratio = np.clip((group4 + 5.0) / (group13 + 5.0), 0.65, 1.4)
    return np.maximum(0.0, base * (0.8 + 0.2 * ratio[idx, None]))


def _training_matrix(history: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    periods = history.shape[1] - len(KEYS)
    origins = _training_origins(periods)
    if not origins:
        raise ValueError("No completed 13-week training windows")
    sales = history.iloc[:, len(KEYS):].to_numpy(dtype=np.float32)
    X, y = [], []
    for origin in origins:
        if origin + HORIZON > periods:
            raise AssertionError("Future training-label leakage")
        X.append(_features(history, origin))
        y.append(sales[:, origin: origin + HORIZON].reshape(-1))
    return pd.concat(X, ignore_index=True), np.concatenate(y), origins


def global_model_predictions(history: pd.DataFrame, *,
                             trees: int = 100, catboost: bool = False) -> dict[str, np.ndarray]:
    """Train global models on common fully observed origin/horizon examples."""
    X, y, _ = _training_matrix(history)
    future = _features(history, history.shape[1] - len(KEYS))
    preds = {}
    for name, objective in (("tweedie", "tweedie"), ("poisson", "poisson")):
        params = dict(LGBM_PARAMS)
        params.update(n_estimators=trees, objective=objective,
                      n_jobs=2, learning_rate=0.06,
                      min_child_samples=100)
        model = lgb.LGBMRegressor(**params)
        model.fit(X, y, categorical_feature=CATEGORICAL)
        preds[name] = np.maximum(0.0, model.predict(future)).reshape(len(history), HORIZON)
    if catboost:
        from catboost import CatBoostRegressor
        cb = CatBoostRegressor(iterations=trees, depth=6, learning_rate=0.08,
                               loss_function="Poisson", thread_count=2,
                               random_seed=2026, verbose=False,
                               l2_leaf_reg=6.0)
        cb.fit(X, y)
        preds["catboost"] = np.maximum(0.0, cb.predict(future)).reshape(len(history), HORIZON)
    return preds


def forecast_candidates(history: pd.DataFrame, *, trees: int = 100,
                        theta_cap: int = 250, catboost: bool = False
                        ) -> tuple[dict[str, np.ndarray], dict]:
    stat = statistical_forecast(history).iloc[:, len(KEYS):].to_numpy(dtype=float)
    sba, tsb = intermittent_forecasts(history)
    theta, mask = theta_forecast(history, cap=theta_cap, fallback=stat)
    candidates = {
        "statistical": stat, "sba": sba, "tsb": tsb, "theta": theta,
        "hierarchical": hierarchical_forecast(history, stat),
    }
    candidates.update(global_model_predictions(history, trees=trees, catboost=catboost))
    assert all(v.shape == (len(history), HORIZON)
               and np.isfinite(v).all() and (v >= 0).all() for v in candidates.values())
    return candidates, {"theta_fitted": int(mask.sum()), "num_series": len(history)}


def _segments(history: pd.DataFrame) -> np.ndarray:
    val = history.iloc[:, len(KEYS):].to_numpy(dtype=float)[:, -52:]
    sparsity = (val == 0).mean(axis=1)
    rate = val.mean(axis=1)
    return np.where(sparsity >= 0.8, 0,
             np.where(sparsity >= 0.5, 1,
             np.where(rate < 10, 2, 3))).astype(np.int8)


def _scores_for_candidates(candidates: dict[str, np.ndarray], target: pd.DataFrame
                           ) -> dict[str, float]:
    return {name: vn1_score(_as_submission(target, predictions), target)["vn1_score"]
            for name, predictions in candidates.items()}


def _as_submission(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    if values.shape != (len(template), HORIZON):
        raise ValueError("Forecast shape does not match submission template")
    output = template.loc[:, KEYS].copy()
    for i, col in enumerate(template.columns[len(KEYS):]):
        output[col] = values[:, i]
    return output


def _softmax_losses(loss: np.ndarray, sharpness: float = 6.0) -> np.ndarray:
    raw = np.clip(-sharpness * loss, -90, 90)
    raw = raw - raw.max(axis=-1, keepdims=True)
    weight = np.exp(raw)
    return weight / weight.sum(axis=-1, keepdims=True)


def adaptive_weights(pred: np.ndarray, truth: np.ndarray, seg: np.ndarray,
                     mode: str = "product") -> np.ndarray:
    """Stable product/segment/global weights from historical predictions.

    pred: (folds, products, models, horizon); truth: (folds, products, horizon).
    Compute normalized error, then shrink per-product weights to their segment.
    No future evaluation-period labels appear in this function.
    """
    if pred.ndim != 4 or truth.shape != pred.shape[:2] + pred.shape[3:]:
        raise ValueError("Historical prediction / target shape mismatch")
    if mode not in ("global", "segment", "product"):
        raise ValueError("Unknown blending mode")
    folds, n, models, horizon = pred.shape
    errors = np.abs(pred - truth[:, :, None, :]).sum(axis=(0, 3))
    volume = truth.sum(axis=(0, 2))
    loss_item = errors / (volume[:, None] + 13 * folds)
    aggregate_loss = errors.sum(axis=0) / max(volume.sum(), 1.0)
    global_w = _softmax_losses(aggregate_loss[None, :])[0]
    segment_w = np.broadcast_to(global_w, (n, models)).copy()
    for group in np.unique(seg):
        ix = seg == group
        local_loss = errors[ix].sum(axis=0) / max(volume[ix].sum(), 1.0)
        seg_w = _softmax_losses(local_loss[None, :])[0]
        # Shrink each segment toward global, limiting regime-overfitting.
        evidence = float(volume[ix].sum())
        shrink = evidence / (evidence + 5000)
        segment_w[ix] = shrink * seg_w + (1 - shrink) * global_w
    if mode == "global":
        return np.broadcast_to(global_w, (n, models)).copy()
    if mode == "segment":
        return segment_w
    item_w = _softmax_losses(loss_item)
    # Only series with sufficient observed volume may deviate materially.
    reliability = np.minimum(0.75, volume / (volume + 40))
    return (reliability[:, None] * item_w +
            (1 - reliability[:, None]) * segment_w)


def apply_weights(candidates: dict[str, np.ndarray], weights: np.ndarray,
                  names: list[str]) -> np.ndarray:
    stacked = np.stack([candidates[name] for name in names], axis=1)
    return np.maximum(0, (stacked * weights[:, :, None]).sum(axis=1))


def historical_folds(history: pd.DataFrame) -> list[int]:
    total = history.shape[1] - len(KEYS)
    origins = list(range(105, total - HORIZON + 1, 13))
    if len(origins) < 3:
        raise ValueError("Need at least three completed historical 13-week folds")
    return origins


def choose_adaptive(history: pd.DataFrame, *, trees: int = 100,
                    theta_cap: int = 250, catboost: bool = False) -> tuple[
                        np.ndarray, list[str], dict]:
    """Use the last historical fold as a true holdout for blend complexity.

    Fit weights on earlier folds only, compare modes on the last fold, then
    refit the selected mode on all completed historical folds. Predictions for
    the actual competition phase are NEVER used in this selection.
    """
    origins = historical_folds(history)
    names = list(MODELS) + (["catboost"] if catboost else [])
    all_pred, all_target, fold_details = [], [], []
    hist_values = history.iloc[:, len(KEYS):].to_numpy(dtype=np.float32)
    for origin in origins:
        earlier = history.iloc[:, :len(KEYS) + origin]
        candidate, metadata = forecast_candidates(earlier, trees=trees,
                                                   theta_cap=theta_cap, catboost=catboost)
        all_pred.append(np.stack([candidate[name] for name in names], axis=1).astype(np.float32))
        all_target.append(hist_values[:, origin: origin + HORIZON])
        fold_details.append({"origin_week": origin, **metadata})
    preds = np.stack(all_pred, axis=0)
    targets = np.stack(all_target, axis=0)
    last_fold_origin = origins[-1]
    prior_segments = _segments(history.iloc[:, :len(KEYS) + last_fold_origin])
    selection_scores = {}
    for mode in ("global", "segment", "product"):
        weights = adaptive_weights(preds[:-1], targets[:-1], prior_segments, mode)
        holdout = (preds[-1] * weights[:, :, None]).sum(axis=1)
        errors = holdout - targets[-1]
        selection_scores[mode] = float((np.abs(errors).sum() + abs(errors.sum())) /
                                       max(targets[-1].sum(), 1.0))
    # Honest holdout gate: favor simpler weights unless a more complex model
    # improves VN1 by >0.002 absolute.
    selected_mode = "global"
    for mode in ("segment", "product"):
        if selection_scores[mode] < selection_scores[selected_mode] - 0.002:
            selected_mode = mode
    final_weight = adaptive_weights(preds, targets, _segments(history), selected_mode)
    return final_weight, names, {
        "cv_origins": origins, "heldout_origin": last_fold_origin,
        "heldout_scores": selection_scores, "selected_mode": selected_mode,
        "historical_folds": fold_details, "number_of_models": len(names),
        "mean_weights": dict(zip(names, map(float, final_weight.mean(axis=0)))),
    }


def run(phase: int, data_dir: Path, output: Path, *, trees: int = 100,
        theta_cap: int = 250, catboost: bool = False) -> dict:
    history = read_history(phase, data_dir)
    weight, names, selection = choose_adaptive(history, trees=trees,
                                                theta_cap=theta_cap, catboost=catboost)
    candidate, metadata = forecast_candidates(history, trees=trees,
                                               theta_cap=theta_cap, catboost=catboost)
    values = apply_weights(candidate, weight, names)
    origin = pd.Timestamp(history.columns[-1])
    future_columns = [(origin + pd.Timedelta(weeks=j + 1)).strftime("%Y-%m-%d")
                      for j in range(HORIZON)]
    template = pd.concat([history[KEYS], pd.DataFrame(np.zeros((len(history), HORIZON)),
                                                  columns=future_columns)], axis=1)
    submission = _as_submission(template, values)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    result = {"phase": phase, "models": names, "selection": selection, "metadata": metadata,
              "submission": str(output)}
    actual_path = data_dir / f"Phase {phase} - Sales.csv"
    if actual_path.exists():
        actual = read_sales(actual_path)
        result["individual_scores"] = _scores_for_candidates(candidate, actual)
        result["ensemble_score"] = vn1_score(submission, actual)
        result["notes"] = "Retrospective result; Phase 2 targets NOT used in model selection."
    else:
        result["notes"] = "Evaluation target absent; no final score claimed."
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="VN1 specialist models with adaptive blending")
    parser.add_argument("--phase", choices=(1, 2), type=int, default=1)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--theta-cap", type=int, default=250)
    parser.add_argument("--catboost", action="store_true",
                        help="Include optional CatBoost model (requires catboost package)")
    args = parser.parse_args()
    output = args.output or Path(f"Submission_Phase_{args.phase}_adaptive.csv")
    print(json.dumps(run(args.phase, args.data_dir, output, trees=args.trees,
                         theta_cap=args.theta_cap, catboost=args.catboost), indent=2))


if __name__ == "__main__":
    main()
