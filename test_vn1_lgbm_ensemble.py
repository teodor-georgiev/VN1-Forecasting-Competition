"""Synthetic leakage, shape, score and ensemble tests for the new LGBM model."""
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from vn1_from_scratch import KEYS, HORIZON, forecast as baseline, vn1_score
from vn1_lgbm_ensemble import (_features, _training_origins, choose_weight,
    ensemble_forecast, lgbm_forecast, train_lgbm)


def fixture(weeks=145, series=20):
    rng = np.random.default_rng(2026)
    dates = pd.date_range("2020-07-06", periods=weeks, freq="W-MON").strftime("%Y-%m-%d")
    ids = pd.DataFrame({"Client": np.arange(series) // 5,
                       "Warehouse": np.arange(series) % 5,
                       "Product": np.arange(series) + 100})
    sales = rng.poisson(np.arange(series)[:, None] % 7 + 3, size=(series, weeks)).astype(float)
    return pd.concat([ids, pd.DataFrame(sales, columns=dates)], axis=1)


def test_training_origins_respect_observation_boundary():
    assert _training_origins(170) == list(range(65, 158, 13))
    assert all(t + 13 <= 170 for t in _training_origins(170))
    assert _training_origins(65) == []


def test_features_are_horizon_aligned_and_history_only():
    hist = fixture(weeks=90, series=12)
    features = _features(hist, 78)
    assert features.shape[0] == 12 * HORIZON
    assert features.iloc[:13].horizon.tolist() == list(range(1, 14))
    assert features.notna().all().all()
    # Modifying future labels in the input must not change features for an earlier origin.
    copy = hist.copy()
    copy.iloc[:, 3 + 78:] += 100_000
    pd.testing.assert_frame_equal(features, _features(copy, 78))


def test_lgbm_and_ensemble_run_on_synthetic_data():
    hist = fixture(weeks=105, series=20)
    lgb = lgbm_forecast(hist, n_estimators=6)
    stat = baseline(hist)
    ens = ensemble_forecast(hist, weight_lgbm=0.3, n_estimators=6)
    assert lgb.shape == stat.shape == ens.shape == (20, 16)
    assert ens[KEYS].equals(hist[KEYS])
    assert np.isfinite(ens.iloc[:, 3:].to_numpy()).all()
    assert (ens.iloc[:, 3:].to_numpy() >= 0).all()
    np.testing.assert_allclose(
        ens.iloc[:, 3:].to_numpy(),
        0.3 * lgb.iloc[:, 3:].to_numpy() + 0.7 * stat.iloc[:, 3:].to_numpy(),
        rtol=1e-6, atol=1e-6,
    )


def test_cv_uses_only_pre_origin_data():
    hist = fixture(weeks=143, series=20)
    chosen = choose_weight(hist, n_estimators=6)
    assert chosen["cv_origin"] == 117
    assert 0 <= chosen["weight_lgbm"] <= 1
    assert max(chosen["cv_train_origins"]) + HORIZON <= 117


def test_invalid_ensemble_weight():
    with pytest.raises(ValueError, match="weight"):
        ensemble_forecast(fixture(weeks=105), weight_lgbm=1.2, n_estimators=6)
