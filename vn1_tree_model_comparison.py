"""Apples-to-apples historical VN1 CatBoost/XGBoost/LightGBM benchmark.

The three global regressors receive the same origin-aligned lag, calendar,
categorical and group-statistical features. Each train row's label is fully
observed before its evaluation origin. A deterministic sample is selected
using only the first 65 weeks, AFTER full-catalogue group features are built.
This is a research benchmark, not a new production model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vn1_from_scratch import HORIZON, KEYS, read_sales
from vn1_lgbm_ensemble import _features, CATEGORICAL, LGBM_PARAMS, START_TRAIN_WEEK, TRAIN_STRIDE

ORIGINS = (105, 118, 131, 144, 157)
MODEL_NAMES = ("lightgbm_tweedie", "catboost_poisson", "xgboost_poisson")


def select_historical_series(history: pd.DataFrame, max_series: int, seed: int = 2026) -> np.ndarray:
    first = history.iloc[:, len(KEYS):len(KEYS) + START_TRAIN_WEEK].to_numpy(dtype=np.float32)
    if first.shape[1] < START_TRAIN_WEEK or not len(first):
        raise ValueError("Need at least 65 historical weeks")
    if max_series <= 0 or max_series >= len(first):
        return np.arange(len(first))
    zeros = (first == 0).mean(axis=1)
    strata = np.digitize(zeros, [0.25, 0.8])
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(np.flatnonzero(strata == g)).tolist() for g in range(3)]
    chosen = []
    while len(chosen) < max_series:
        any_left = False
        for group in groups:
            if group and len(chosen) < max_series:
                chosen.append(group.pop())
                any_left = True
        if not any_left:
            break
    return np.asarray(sorted(chosen), dtype=int)


def feature_panels(history: pd.DataFrame, chosen: np.ndarray, origins: list[int]) -> dict:
    weeks = history.shape[1] - len(KEYS)
    if not origins or min(origins) < 105 or max(origins) + HORIZON > weeks:
        raise ValueError("Validation folds not complete or not historical")
    train_origins = sorted({
        t for origin in origins for t in
        range(START_TRAIN_WEEK, origin - HORIZON + 1, TRAIN_STRIDE)
    })
    rows = (chosen[:, None] * HORIZON + np.arange(HORIZON)).reshape(-1)
    panels = {}
    for origin in sorted(set(train_origins) | set(origins)):
        # Preserve peer features from whole catalog; take same rows for all models.
        panels[origin] = _features(history, origin).iloc[rows].reset_index(drop=True)
    sales = history.iloc[:, len(KEYS):].to_numpy(dtype=np.float32)[chosen]
    return {"panels": panels, "sales": sales, "train_origins": train_origins}


def model_inputs(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    """Consistent IDs; native categorical encoding for CatBoost and XGBoost."""
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model}")
    X = frame.copy()
    if model == "catboost_poisson":
        for c in CATEGORICAL:
            X[c] = X[c].round().astype("int64").astype(str)
    elif model == "xgboost_poisson":
        for c in CATEGORICAL:
            X[c] = X[c].round().astype("int64").astype("category")
    return X


def fit_predict(name: str, train_x: pd.DataFrame, train_y: np.ndarray,
                valid_x: pd.DataFrame, *, trees: int = 160) -> np.ndarray:
    if trees < 1:
        raise ValueError("trees must be positive")
    if name == "lightgbm_tweedie":
        import lightgbm as lgb
        model = lgb.LGBMRegressor(**{**LGBM_PARAMS, "n_estimators": trees})
        model.fit(train_x, train_y, categorical_feature=CATEGORICAL)
    elif name == "catboost_poisson":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(
            loss_function="Poisson", iterations=trees, depth=6,
            learning_rate=0.08, l2_leaf_reg=6.0,
            thread_count=2, random_seed=2026, verbose=False,
            allow_writing_files=False,
        )
        model.fit(model_inputs(train_x, name), train_y, cat_features=CATEGORICAL)
        return np.maximum(0.0, model.predict(model_inputs(valid_x, name)))
    elif name == "xgboost_poisson":
        from xgboost import XGBRegressor
        model = XGBRegressor(
            objective="count:poisson", n_estimators=trees, max_depth=6,
            learning_rate=0.08, reg_lambda=3, tree_method="hist",
            colsample_bytree=0.9, subsample=0.9, n_jobs=2,
            random_state=2026, enable_categorical=True,
        )
        model.fit(model_inputs(train_x, name), train_y)
        return np.maximum(0.0, model.predict(model_inputs(valid_x, name)))
    else:
        raise ValueError(f"Unknown model: {name}")
    return np.maximum(0.0, model.predict(valid_x))


def official_loss(pred: np.ndarray, actual: np.ndarray) -> float:
    pred, actual = np.asarray(pred, dtype=float), np.asarray(actual, dtype=float)
    if pred.shape != actual.shape or not np.isfinite(pred).all() or not np.isfinite(actual).all():
        raise ValueError("Invalid aligned forecasts/labels")
    if actual.sum() <= 0:
        raise ValueError("Total actual sales must be positive")
    error = pred - actual
    return float((np.abs(error).sum() + abs(error.sum())) / actual.sum())


def evaluate_model(name: str, cache: dict, origins: list[int], *, trees: int = 160) -> dict:
    x, sales = cache["panels"], cache["sales"]
    results = []
    errors = []
    targets = []
    for origin in origins:
        train_origins = [t for t in cache["train_origins"] if t + HORIZON <= origin]
        if not train_origins:
            raise ValueError("No completed label windows at forecast origin")
        training = pd.concat([x[t] for t in train_origins], ignore_index=True)
        labels = np.concatenate([sales[:, t:t+HORIZON].reshape(-1) for t in train_origins])
        pred = fit_predict(name, training, labels, x[origin], trees=trees).reshape(len(sales), HORIZON)
        actual = sales[:, origin:origin + HORIZON]
        results.append({"origin": origin, "score": official_loss(pred, actual),
                        "train_origins": train_origins})
        errors.append((pred-actual).reshape(-1))
        targets.append(actual.reshape(-1))
    all_err, all_y = np.concatenate(errors), np.concatenate(targets)
    pooled = float((np.abs(all_err).sum() + abs(all_err.sum())) / all_y.sum())
    return {"model": name, "pooled_vn1": pooled, "fold_scores": results,
            "sampled_series": len(sales)}


def run(history: pd.DataFrame, *, max_series: int = 1024,
        trees: int = 160, seed: int = 2026) -> dict:
    origins = [o for o in ORIGINS if o + HORIZON <= history.shape[1] - len(KEYS)]
    if len(origins) != len(ORIGINS):
        raise ValueError("Need all five historical folds including January")
    selected = select_historical_series(history, max_series, seed=seed)
    cache = feature_panels(history, selected, origins)
    results = {}
    for name in MODEL_NAMES:
        results[name] = evaluate_model(name, cache, origins, trees=trees)
    return {
        "all_models_same_selected_series": True,
        "all_models_same_folds": True,
        "sampled_series": len(selected), "total_series": len(history),
        "selection_uses_initial_65_weeks": True, "origins": origins,
        "january_origin": 131,
        "trees": trees,
        "results": results,
        "experimental_not_production": True,
        "no_competition_targets_used": True,
    }


def main():
    parser = argparse.ArgumentParser(description="VN1 matched CatBoost/XGBoost/LightGBM pilot")
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--max-series", type=int, default=1024)
    parser.add_argument("--trees", type=int, default=160)
    parser.add_argument("--output", type=Path, default=Path("VN1_Tree_Model_Comparison.json"))
    args = parser.parse_args()
    sales = read_sales(args.data_dir / "Phase 0 - Sales.csv")
    results = run(sales, max_series=args.max_series, trees=args.trees)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf8")
    print(json.dumps({"output": str(args.output),
                      "model_scores": {name: x["pooled_vn1"]
                                       for name, x in results["results"].items()},
                      "origins": results["origins"]}, indent=2))


if __name__ == "__main__":
    main()
