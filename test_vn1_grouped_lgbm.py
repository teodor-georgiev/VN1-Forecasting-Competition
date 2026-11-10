"""Synthetic and leakage tests for grouped VN1 LightGBM experiment."""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from vn1_grouped_lgbm import (
    START_TRAIN_WEEK, demand_profile, fit_frozen_clusters,
    group_feature_frame, grouped_forecast,
)
from vn1_from_scratch import HORIZON, KEYS


def example_history(series=16, weeks=170):
    rng = np.random.default_rng(20261110)
    ids = pd.DataFrame({
        "Client": np.arange(series) // 4,
        "Warehouse": np.arange(series) % 3,
        "Product": np.arange(series) + 1000,
    })
    levels = np.arange(series)[:, None] + 0.7
    sales = rng.poisson(levels, size=(series, weeks)).astype(float)
    sales[0, :] = 0.0
    sales[1, 1::3] = 0.0
    dates = pd.date_range("2020-07-06", periods=weeks, freq="W-MON").strftime("%Y-%m-%d")
    return pd.concat([ids, pd.DataFrame(sales, columns=dates)], axis=1)


def test_static_cluster_centroids_and_point_in_time_group_features():
    hist = example_history()
    scaler, km = fit_frozen_clusters(hist, n_clusters=4)
    earlier = group_feature_frame(hist, 105, scaler, km)
    assert earlier.shape[0] == len(hist) * HORIZON
    assert len(earlier) == len(hist) * HORIZON
    assert set(("client_warehouse", "demand_cluster", "cw_peer_mean13",
                "warehouse_total_mean13")).issubset(earlier.columns)
    assert (earlier["cw_group_size"] >= 1).all()

    mutated = hist.copy()
    mutated.iloc[:, len(KEYS) + 105:] += 10_000
    same_scaler, same_km = fit_frozen_clusters(mutated, n_clusters=4)
    assert_frame_equal(group_feature_frame(mutated, 105, same_scaler, same_km), earlier)
    np.testing.assert_array_equal(km.cluster_centers_, same_km.cluster_centers_)


def test_no_invalid_demand_profiles():
    hist = example_history(weeks=100)
    with pytest.raises(ValueError):
        demand_profile(hist, START_TRAIN_WEEK - 1)
    with pytest.raises(ValueError):
        demand_profile(hist, 101)
    with pytest.raises(ValueError):
        fit_frozen_clusters(hist, n_clusters=1000)


def test_grouped_model_trains_from_completed_historical_windows_only():
    hist = example_history()
    forecast, receipt = grouped_forecast(hist, trees=5, clusters=4)
    assert forecast.shape == (len(hist), len(KEYS) + HORIZON)
    assert np.isfinite(forecast.iloc[:, len(KEYS):].to_numpy()).all()
    assert (forecast.iloc[:, len(KEYS):].to_numpy() >= 0).all()
    assert all(o + HORIZON <= 170 for o in receipt["completed_training_origins"])
    assert receipt["frozen_cluster_training_cutoff"] == START_TRAIN_WEEK
    assert receipt["num_clusters"] == 4
    assert "client_warehouse" in receipt["categorical_features"]
