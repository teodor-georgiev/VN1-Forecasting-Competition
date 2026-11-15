"""Optuna LightGBM hyperparameter screening on VN1 historical rolling origins.

Training and tuning use only Phase 0 sales, never competition Phase 1/2
targets. The final historical fold is held out from all Optuna trials.
Feature creation uses all series at each origin for correct group statistics;
screening then takes a deterministic, historically selected subset of rows.
Top configurations must be revalidated on all series before promotion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from vn1_from_scratch import HORIZON, KEYS, read_sales
from vn1_lgbm_ensemble import CATEGORICAL, LGBM_PARAMS, START_TRAIN_WEEK, TRAIN_STRIDE, _features

ROLLING_ORIGINS = [105, 118, 131, 144, 157]
DEFAULT_TRIALS = 60


def choose_series(history: pd.DataFrame, max_series: int, seed: int = 2026) -> np.ndarray:
    """Select before validation using only the first 65 observed weeks."""
    values = history.iloc[:, len(KEYS):len(KEYS) + START_TRAIN_WEEK].to_numpy(dtype=float)
    if len(values) == 0 or len(values[0]) < START_TRAIN_WEEK:
        raise ValueError("Insufficient history for historical stratification")
    n = len(values)
    if max_series <= 0 or max_series >= n:
        return np.arange(n, dtype=int)
    # Balance sparse, medium and dense series, without looking at target windows.
    groups = np.digitize((values == 0).mean(axis=1), [0.25, 0.8])
    rng = np.random.default_rng(seed)
    pools = [rng.permutation(np.flatnonzero(groups == i)).tolist() for i in range(3)]
    chosen = []
    while len(chosen) < max_series:
        progress = False
        for pool in pools:
            if pool and len(chosen) < max_series:
                chosen.append(pool.pop())
                progress = True
        if not progress:
            break
    return np.array(sorted(chosen), dtype=int)


def valid_origins(history: pd.DataFrame) -> list[int]:
    periods = history.shape[1] - len(KEYS)
    origins = [x for x in ROLLING_ORIGINS if x + HORIZON <= periods]
    if len(origins) < 4 or 131 not in origins:
        raise ValueError("Require at least 4 complete historical folds including January")
    return origins


def setup_cache(history: pd.DataFrame, series: np.ndarray, origins: list[int]) -> dict:
    """One expensive full-catalog feature pass per training or validation origin.

    After each pass the table is sliced to the exact same sampled series, so
    client/warehouse peer aggregates are never accidentally recomputed on the
    smaller catalogue.
    """
    periods = history.shape[1] - len(KEYS)
    if max(origins) + HORIZON > periods:
        raise ValueError("A validation target extends beyond observed history")
    train_origins = sorted({
        t for o in origins for t in
        range(START_TRAIN_WEEK, o - HORIZON + 1, TRAIN_STRIDE)
    })
    needed = sorted(set(train_origins) | set(origins))
    indices = (series[:, None] * HORIZON + np.arange(HORIZON)).reshape(-1)
    X = {}
    for origin in needed:
        full = _features(history, origin)
        X[origin] = full.iloc[indices].reset_index(drop=True)
    y = history.iloc[:, len(KEYS):].to_numpy(dtype=np.float32)[series]
    return {
        "features": X,
        "sales": y,
        "train_origins": train_origins,
        "valid_origins": origins,
        "n_series": len(series),
    }


def score_vn1(pred: np.ndarray, actual: np.ndarray) -> float:
    """Official bias-adjusted WAPE, independent of pandas output schema."""
    p, a = np.asarray(pred, dtype=float), np.asarray(actual, dtype=float)
    if p.shape != a.shape or not np.isfinite(p).all() or not np.isfinite(a).all():
        raise ValueError("Prediction/label arrays must be finite and aligned")
    if np.sum(a) <= 0:
        raise ValueError("No positive sales in validation data")
    error = p - a
    return float((np.abs(error).sum() + abs(error.sum())) / a.sum())


def score_folds(cache: dict, parameters: dict, origins: list[int]) -> dict:
    """Refit at each origin with strictly completed historical labels."""
    X, sales = cache["features"], cache["sales"]
    errors = []
    fold_results = []
    for origin in origins:
        past = [t for t in cache["train_origins"] if t + HORIZON <= origin]
        if not past:
            raise ValueError("Validation origin has no earlier completed labels")
        train_x = pd.concat([X[t] for t in past], ignore_index=True)
        train_y = np.concatenate([sales[:, t:t+HORIZON].reshape(-1) for t in past])
        model = lgb.LGBMRegressor(**parameters)
        model.fit(train_x, train_y, categorical_feature=CATEGORICAL)
        preds = np.maximum(0, model.predict(X[origin])).reshape(len(sales), HORIZON)
        truth = sales[:, origin:origin + HORIZON]
        err = preds - truth
        errors.append(err.reshape(-1))
        fold_results.append({"origin": origin, "score": score_vn1(preds, truth),
                             "train_origins": past})
    all_error = np.concatenate(errors)
    all_truth = np.concatenate([sales[:, o:o + HORIZON].reshape(-1) for o in origins])
    combined_score = float((np.abs(all_error).sum() + abs(all_error.sum())) / all_truth.sum())
    return {"pooled_vn1_score": combined_score, "folds": fold_results}


def trial_params(trial: optuna.Trial, max_trees: int = 400) -> dict:
    if max_trees < 8:
        raise ValueError("max_trees must be at least 8")
    if max_trees < 80:
        min_trees, tree_step = 8, 8
    else:
        min_trees, tree_step = 80, 40
    upper = min_trees + (max_trees - min_trees) // tree_step * tree_step
    params = {**LGBM_PARAMS,
        "objective": "tweedie",
        "n_estimators": trial.suggest_int("n_estimators", min_trees, upper, step=tree_step),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.16, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 250),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50.0, log=True),
        "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.1, 1.9, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "n_jobs": 2, "random_state": 2026, "verbosity": -1,
    }
    return params


def optimize(history: pd.DataFrame, *, trials: int = DEFAULT_TRIALS,
             max_series: int = 512, max_trees: int = 400,
             seed: int = 2026) -> dict:
    if trials < 1:
        raise ValueError("At least one trial required")
    origins = valid_origins(history)
    chosen = choose_series(history, max_series=max_series, seed=seed)
    cache = setup_cache(history, chosen, origins)
    tune, holdout = origins[:-1], origins[-1:]
    assert 131 in tune, "The tuning folds must include January"
    baseline = score_folds(cache, dict(LGBM_PARAMS), tune)
    baseline_heldout = score_folds(cache, dict(LGBM_PARAMS), holdout)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    def objective(trial):
        params = trial_params(trial, max_trees=max_trees)
        receipt = score_folds(cache, params, tune)
        trial.set_user_attr("folds", receipt["folds"])
        return receipt["pooled_vn1_score"]
    study.optimize(objective, n_trials=trials, gc_after_trial=True)
    best = dict(LGBM_PARAMS, **study.best_trial.params)
    best["n_jobs"] = 2
    best["random_state"] = 2026
    best["verbosity"] = -1
    tuned = score_folds(cache, best, tune)
    # Entire chronological holdout remains untouched until AFTER Optuna.
    tuned_holdout = score_folds(cache, best, holdout)
    result = {
        "number_of_trials": len(study.trials),
        "sampled_series": len(chosen),
        "total_series": len(history),
        "selection_uses_only_first_65_weeks": True,
        "tuning_origins": tune,
        "holdout_origins": holdout,
        "best_parameters": study.best_trial.params,
        "baseline_tuning": baseline,
        "tuned_tuning": tuned,
        "baseline_holdout": baseline_heldout,
        "tuned_holdout": tuned_holdout,
        "trials": [{"number": t.number, "state": t.state.name,
                    "score": t.value, "params": t.params}
                   for t in study.trials],
        "promotion_policy": "No automatic champion change; require full-catalogue follow-up and holdout improvement",
        "no_phase1_or_phase2_labels_used": True,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="VN1 Optuna rolling-origin LightGBM search")
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--max-series", type=int, default=512)
    parser.add_argument("--max-trees", type=int, default=400)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("VN1_Optuna_results.json"))
    args = parser.parse_args()
    history = read_sales(args.data_dir / "Phase 0 - Sales.csv")
    result = optimize(history, trials=args.trials, max_series=args.max_series,
                      max_trees=args.max_trees, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf8")
    print(json.dumps({
        "output": str(args.output),
        "trials": result["number_of_trials"],
        "best_parameters": result["best_parameters"],
        "baseline_tuning": result["baseline_tuning"]["pooled_vn1_score"],
        "tuned_tuning": result["tuned_tuning"]["pooled_vn1_score"],
        "baseline_holdout": result["baseline_holdout"]["pooled_vn1_score"],
        "tuned_holdout": result["tuned_holdout"]["pooled_vn1_score"],
    }, indent=2))


if __name__ == "__main__":
    main()
