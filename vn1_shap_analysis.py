"""Aggregate LightGBM feature attributions for VN1 without target-label leakage.

LightGBM's built-in TreeSHAP reports additive contributions on the *raw*
prediction scale. For Tweedie/Poisson objectives that is log-expected-sales,
not additional units sold. Report only aggregate feature importance: never
interpret a feature's SHAP value as a causal effect.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vn1_lgbm_ensemble import _features, read_history, train_lgbm
from vn1_from_scratch import KEYS


def shap_summary(feature_names: list[str], contributions: np.ndarray,
                 raw_scores: np.ndarray | None = None, *,
                 top_k: int = 40) -> dict:
    """Summarize feature contributions and verify additive raw-score identity.

    The last SHAP column is the model's expected raw score (bias term).
    A finite, correctly shaped matrix is required. The summary contains
    no series, client, or warehouse identifiers.
    """
    values = np.asarray(contributions, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("SHAP matrix must be a nonempty two-dimensional array")
    if values.shape[1] != len(feature_names) + 1:
        raise ValueError("Expected one contribution per feature plus bias")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("Feature names must be unique")
    if not np.isfinite(values).all():
        raise ValueError("SHAP contributions must be finite")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    expected = values[:, -1]
    reconstructed = values.sum(axis=1)
    max_discrepancy = None
    if raw_scores is not None:
        raw = np.asarray(raw_scores, dtype=float).reshape(-1)
        if raw.shape != reconstructed.shape or not np.isfinite(raw).all():
            raise ValueError("Raw score shape and finiteness must match SHAP")
        max_discrepancy = float(np.max(np.abs(raw - reconstructed)))
        if not np.allclose(raw, reconstructed, rtol=1e-4, atol=1e-4):
            raise ValueError("Feature contributions do not reconstruct raw scores")
    absolute = np.abs(values[:, :-1]).mean(axis=0)
    signed = values[:, :-1].mean(axis=0)
    denominator = max(float(absolute.sum()), 1e-12)
    ranked = sorted(range(len(feature_names)), key=lambda i: (-absolute[i], feature_names[i]))
    families = {}
    for i, name in enumerate(feature_names):
        if name.startswith("client_") or name == "Client":
            group = "client"
        elif name.startswith("warehouse_") or name == "Warehouse":
            group = "warehouse"
        elif name.startswith("product_") or name == "Product":
            group = "product"
        elif name.startswith(("week_", "future_", "origin_", "is_january", "horizon")):
            group = "calendar"
        else:
            group = "demand_history"
        families[group] = families.get(group, 0.0) + float(absolute[i])
    return {
        "observations": int(values.shape[0]),
        "scale": "raw_log_expected_sales_for_tweedie_poisson",
        "expected_raw_score_mean": float(expected.mean()),
        "max_raw_score_reconstruction_error": max_discrepancy,
        "family_abs_share": {key: value / denominator for key, value in sorted(families.items())},
        "features": [
            {"name": feature_names[i], "mean_abs_shap_raw": float(absolute[i]),
             "mean_signed_shap_raw": float(signed[i]),
             "importance_share": float(absolute[i] / denominator)}
            for i in ranked[:top_k]
        ],
    }


def run(phase: int, data_dir: Path, *, trees: int = 160, sample_rows: int = 2000,
        seed: int = 2026, top_k: int = 40) -> dict:
    if sample_rows < 1 or trees < 1:
        raise ValueError("trees and sample_rows must be positive")
    history = read_history(phase, data_dir)
    model, training_origins = train_lgbm(history, n_estimators=trees)
    X = _features(history, history.shape[1] - len(KEYS))
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(X), size=min(sample_rows, len(X)), replace=False))
    sampled = X.iloc[chosen]
    booster = model.booster_
    contributions = booster.predict(sampled, pred_contrib=True)
    if isinstance(contributions, list):
        raise ValueError("Only one-output regressors are supported")
    raw = booster.predict(sampled, raw_score=True)
    summary = shap_summary(list(sampled.columns), contributions, raw, top_k=top_k)
    summary.update({
        "phase": phase,
        "trees": trees,
        "sample_rows_requested": sample_rows,
        "sample_seed": seed,
        "training_origins": training_origins,
        "data_boundary": "Only historical sales through the phase prediction origin were used",
        "interpretation": "SHAP explains model predictions, not causal demand drivers",
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="VN1 LightGBM aggregate SHAP report")
    parser.add_argument("--phase", type=int, choices=(1, 2), default=1)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--trees", type=int, default=160)
    parser.add_argument("--sample-rows", type=int, default=2000)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--output", type=Path, default=Path("VN1_SHAP_summary.json"))
    args = parser.parse_args()
    summary = run(args.phase, args.data_dir, trees=args.trees,
                  sample_rows=args.sample_rows, top_k=args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "top_features": summary["features"][:10]}, indent=2))


if __name__ == "__main__":
    main()
