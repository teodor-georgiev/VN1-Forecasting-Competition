"""Synthetic regression tests for specialist forecasters and dynamic weighting."""
import numpy as np
import pandas as pd
import pytest

from vn1_adaptive import (
    _segments, adaptive_weights, apply_weights, forecast_candidates,
    historical_folds, hierarchical_forecast, intermittent_forecasts,
    theta_forecast, global_model_predictions, choose_adaptive,
)
from vn1_from_scratch import KEYS, HORIZON, forecast as statistical_forecast, vn1_score


def example_history(n=12, weeks=170, seed=2026):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-07-06", periods=weeks, freq="W-MON").strftime("%Y-%m-%d")
    ids = pd.DataFrame({"Client": np.arange(n) // 3,
                        "Warehouse": np.arange(n) % 3,
                        "Product": 1000 + np.arange(n)})
    levels = np.maximum(0.1, np.arange(n) + 1)[:, None]
    data = rng.poisson(levels, size=(n, weeks)).astype(float)
    data[0] = 0
    data[1, 1::4] = 0
    return pd.concat([ids, pd.DataFrame(data, columns=dates)], axis=1)


def test_intermit_all_zero_and_decay_after_obsolescence():
    hist = example_history()
    vals = list(hist.columns[3:])
    hist.loc[1, vals] = 0.0
    hist.loc[1, vals[2]] = 10.0
    sba, tsb = intermittent_forecasts(hist, alpha=0.1, beta=0.1)
    assert sba.shape == tsb.shape == (12, HORIZON)
    assert np.all(sba[0] == 0) and np.all(tsb[0] == 0)
    assert sba[1, 0] > tsb[1, 0], "TSB should decay after prolonged zero demand"
    assert np.isfinite(sba).all() and np.isfinite(tsb).all()


def test_no_future_leakage_in_intermittent_or_hierarchical():
    hist = example_history()
    stat = statistical_forecast(hist).iloc[:, 3:].to_numpy()
    sba, tsb = intermittent_forecasts(hist)
    hierarchy = hierarchical_forecast(hist, stat)
    assert hierarchy.shape == stat.shape
    assert (hierarchy >= 0).all()
    mutated = hist.copy()
    mutated.iloc[:, -1] += 1000
    assert not np.allclose(intermittent_forecasts(mutated)[0], sba)


def test_theta_cap_and_fallback_defined_for_all_products():
    hist = example_history()
    safe = statistical_forecast(hist).iloc[:, 3:].to_numpy()
    values, mask = theta_forecast(hist, cap=2, fallback=safe)
    assert values.shape == safe.shape
    assert mask.sum() <= 2
    assert np.isfinite(values).all() and (values >= 0).all()
    assert np.allclose(values[~mask], safe[~mask])


def test_weights_sum_to_one_and_shrink_for_sparse_data():
    rng = np.random.default_rng(33)
    pred = rng.random((3, 10, 4, HORIZON))
    truth = rng.poisson(3, (3, 10, HORIZON)).astype(float)
    seg = np.arange(10) % 4
    for mode in ("global", "segment", "product"):
        w = adaptive_weights(pred, truth, seg, mode=mode)
        assert w.shape == (10, 4)
        np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-8)
        assert np.isfinite(w).all() and (w >= 0).all()
    with pytest.raises(ValueError):
        adaptive_weights(pred, truth, seg, mode="future")


def test_product_model_selection_prefers_consistently_better_model():
    pred = np.ones((4, 6, 3, HORIZON))
    truth = np.ones((4, 6, HORIZON))
    pred[:, :, 1, :] = 8.0
    pred[:, :, 2, :] = 5.0
    w = adaptive_weights(pred, truth, np.array([0, 1, 1, 2, 2, 3]), mode="product")
    assert np.all(w[:, 0] > w[:, 1])
    assert np.all(w[:, 0] > w[:, 2])


def test_origin_folds_require_observed_targets():
    assert historical_folds(example_history(weeks=170)) == [105, 118, 131, 144, 157]
    assert historical_folds(example_history(weeks=183))[-1] == 170
    with pytest.raises(ValueError):
        historical_folds(example_history(weeks=120))


def test_global_lightgbm_models_fit_and_forecast():
    hist = example_history(n=12, weeks=100)
    models = global_model_predictions(hist, trees=5)
    assert set(models) == {"tweedie", "poisson"}
    assert all(arr.shape == (12, 13) for arr in models.values())


def test_ensemble_pipeline_uses_historical_selection_only():
    hist = example_history(n=12, weeks=157)
    # 3 folds available, future targets end at week 157, all historical.
    assert historical_folds(hist) == [105, 118, 131, 144]
    weights, names, result = choose_adaptive(hist, trees=4, theta_cap=0)
    assert result["heldout_origin"] == 144
    assert all(origin + HORIZON <= 157 for origin in result["cv_origins"])
    assert result["selected_mode"] in ("champion", "global", "segment", "product")
    assert np.isfinite(weights).all()
    assert set(names) >= {"tweedie", "poisson", "sba", "tsb", "theta", "hierarchical"}


def test_optional_catboost_smoke():
    pytest.importorskip("catboost")
    hist = example_history(n=12, weeks=100)
    out = global_model_predictions(hist, trees=4, catboost=True)
    assert np.isfinite(out["catboost"]).all()
    assert out["catboost"].shape == (12, 13)


def test_model_eligibility_keeps_dense_series_on_champion():
    hist = example_history(n=12, weeks=100)
    candidates, meta = forecast_candidates(hist, trees=5, theta_cap=0)
    zero_fraction = (hist.iloc[:, 3:].to_numpy()[:, -52:] == 0).mean(axis=1)
    dense = zero_fraction < 0.5
    for model in ("sba", "tsb", "theta"):
        np.testing.assert_allclose(candidates[model][dense], candidates["champion"][dense])
    assert (candidates["poisson"] >= 0).all()
