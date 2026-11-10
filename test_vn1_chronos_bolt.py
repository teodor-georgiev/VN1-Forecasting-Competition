"""Offline synthetic checks for the optional Chronos-Bolt adapter."""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from vn1_chronos_bolt import (
    HORIZON, experiment, median_quantile, predict_batched, stratified_indices,
)
from vn1_from_scratch import KEYS, forecast as statistical_forecast


def example_history(n=24, weeks=170):
    rng = np.random.default_rng(601)
    ids = pd.DataFrame({"Client": np.arange(n) // 4,
                        "Warehouse": np.arange(n) % 3,
                        "Product": 1000 + np.arange(n)})
    data = rng.poisson(2 + np.arange(n)[:, None] / 3, size=(n, weeks)).astype(float)
    data[0, :] = 0
    data[1, ::2] = 0
    data[2, ::4] = 0
    dates = pd.date_range("2020-07-06", periods=weeks, freq="W-MON").strftime("%Y-%m-%d")
    return pd.concat([ids, pd.DataFrame(data, columns=dates)], axis=1)


def dummy_quantiles(batch):
    n = len(batch)
    median = np.broadcast_to(batch[:, -13:].mean(axis=1)[:, None],
                             (n, HORIZON))
    return np.stack([median + offset for offset in range(-4, 5)], axis=1)


def test_median_extraction_shapes_and_clips():
    raw = np.stack([np.full((2, HORIZON), q - 4) for q in range(9)], axis=1)
    np.testing.assert_array_equal(median_quantile(raw, n_series=2),
                                  np.zeros((2, HORIZON)))
    with pytest.raises(ValueError):
        median_quantile(np.zeros((2, 2, HORIZON)), n_series=2)
    with pytest.raises(ValueError):
        median_quantile(np.zeros((2, 9, HORIZON - 1)), n_series=2)
    with pytest.raises(ValueError):
        median_quantile(np.full((2, 9, HORIZON), np.nan), n_series=2)


def test_stratification_and_batching_are_deterministic():
    history = example_history()
    first = stratified_indices(history, 11, seed=2026)
    np.testing.assert_array_equal(first, stratified_indices(history, 11, seed=2026))
    assert len(first) == 11 and len(set(first)) == 11
    assert len(stratified_indices(history, 0)) == len(history)
    values = history.iloc[first, len(KEYS):].to_numpy(dtype=float)
    output = predict_batched(values, dummy_quantiles, batch_size=4)
    assert output.shape == (11, HORIZON)
    assert np.isfinite(output).all()
    with pytest.raises(ValueError):
        predict_batched(values, dummy_quantiles, batch_size=0)


def test_experiment_keeps_incumbent_outside_historical_selection():
    history = example_history()
    incumbent = statistical_forecast(history)
    original = incumbent.copy(deep=True)
    result, info, selected = experiment(
        history, incumbent, dummy_quantiles, max_series=7,
        blend=0.5, batch_size=3)
    assert len(selected) == info["tested_series"] == 7
    assert_frame_equal(incumbent, original)
    not_selected = np.setdiff1d(np.arange(len(history)), selected)
    np.testing.assert_allclose(
        result.iloc[not_selected, len(KEYS):],
        original.iloc[not_selected, len(KEYS):])
    assert (result.iloc[selected, len(KEYS):].to_numpy() >= 0).all()
    assert not np.allclose(result.iloc[selected, len(KEYS):],
                           original.iloc[selected, len(KEYS):])
    with pytest.raises(ValueError):
        experiment(history, incumbent, dummy_quantiles, blend=1.5)


def test_chronos_does_not_consume_unobserved_sales():
    history = example_history()
    observed = history.iloc[:, :len(KEYS) + 150]
    # The 13-week horizon after week 150 is withheld entirely.
    ids = stratified_indices(observed, 8)
    measured = observed.iloc[ids, len(KEYS):].to_numpy()
    seen = []
    def fake_predict(batch):
        seen.append(batch.copy())
        return dummy_quantiles(batch)
    predictions = predict_batched(measured, fake_predict, batch_size=3)
    assert predictions.shape == (8, HORIZON)
    assert all(batch.shape[1] == 150 for batch in seen)
