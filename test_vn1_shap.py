"""Regression tests for per-feature LightGBM attribution receipts."""
import numpy as np
import pytest

from vn1_shap_analysis import shap_summary


def test_shap_summary_additivity_and_rankings():
    names = ["Client", "Warehouse", "mean_13", "future_month"]
    features = np.array([[0.2, 0.1, 1.3, 0.0],
                         [0.0, -0.2, 0.9, 0.1],
                         [0.1, 0.0, 1.1, 0.1]])
    bias = np.full((3, 1), 1.5)
    contrib = np.column_stack([features, bias])
    raw = contrib.sum(axis=1)
    out = shap_summary(names, contrib, raw_scores=raw)
    assert out["features"][0]["name"] == "mean_13"
    assert out["max_raw_score_reconstruction_error"] == 0
    assert out["observations"] == 3
    assert set(out["family_abs_share"]) == {"client", "warehouse", "calendar", "demand_history"}
    assert sum(out["family_abs_share"].values()) == pytest.approx(1.0)
    assert sum(x["importance_share"] for x in out["features"]) == pytest.approx(1.0)
    assert out["scale"] == "raw_log_expected_sales_for_tweedie_poisson"


def test_shap_checks_raw_scale_and_rejects_bad_inputs():
    contribution = np.ones((4, 3), dtype=float)
    with pytest.raises(ValueError, match="reconstruct"):
        shap_summary(["a", "b"], contribution, raw_scores=np.ones(4))
    with pytest.raises(ValueError, match="one contribution"):
        shap_summary(["a", "b", "c"], contribution)
    with pytest.raises(ValueError, match="unique"):
        shap_summary(["a", "a"], contribution)
    with pytest.raises(ValueError, match="finite"):
        shap_summary(["a", "b"], np.full((4, 3), np.nan))
    with pytest.raises(ValueError, match="positive"):
        shap_summary(["a", "b"], contribution, top_k=0)


def test_shap_actual_lightgbm_tweedie_explains_raw_not_sales():
    import lightgbm as lgb

    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, size=(120, 3)).astype("float32")
    y = rng.poisson(2 + X[:, 0] * 3).astype(float)
    model = lgb.LGBMRegressor(objective="tweedie", n_estimators=7,
                              min_child_samples=4, verbosity=-1, n_jobs=1)
    model.fit(X, y)
    sample = X[:20]
    booster = model.booster_
    contributions = booster.predict(sample, pred_contrib=True)
    raw = booster.predict(sample, raw_score=True)
    summary = shap_summary(["a", "b", "c"], contributions, raw)
    assert summary["max_raw_score_reconstruction_error"] < 1e-4
    # Tweedie predictions are exp(raw), not sums of raw contributions.
    assert not np.allclose(model.predict(sample), np.asarray(contributions).sum(axis=1))
