"""Synthetic-data regression and no-leakage checks for the standalone forecaster."""
import numpy as np
import pandas as pd
import pytest

from vn1_from_scratch import KEYS, forecast, vn1_score


def example_history(n=6, weeks=80):
    dates = pd.date_range("2021-01-04", periods=weeks, freq="W-MON").strftime("%Y-%m-%d")
    rows = []
    for i in range(n):
        row = {"Client": i // 3, "Warehouse": i % 3, "Product": 1000 + i}
        row.update({date: float(i + 1) for date in dates})
        rows.append(row)
    return pd.DataFrame(rows)


def test_constant_series_produces_same_constant_for_13_weeks():
    history = example_history()
    pred = forecast(history)
    assert pred.shape == (6, 16)
    assert pred.columns[:3].tolist() == KEYS
    assert np.allclose(pred.iloc[:, 3:].to_numpy(), np.arange(1, 7)[:, None])
    assert np.isfinite(pred.iloc[:, 3:].to_numpy()).all()


def test_score_formula_and_bias():
    history = example_history(n=1)
    pred = forecast(history)
    truth = pred.copy()
    truth.iloc[:, 3:] = 2.0
    pred.iloc[:, 3:] = 0.0
    m = vn1_score(pred, truth)
    assert np.isclose(m["vn1_score"], 2.0)
    assert np.isclose(m["bias_fraction"], -1.0)


def test_future_truth_is_not_a_forecasting_input():
    history = example_history(n=3)
    original = history.copy(deep=True)
    p = forecast(history)
    assert history.equals(original)
    assert (p.iloc[:, 3:] >= 0).all().all()
    with pytest.raises(ValueError, match="IDs"):
        wrong = p.copy()
        wrong.loc[0, "Product"] = 999
        vn1_score(p, wrong)


def test_requires_65_weeks():
    with pytest.raises(ValueError, match="65"):
        forecast(example_history(weeks=40))


def test_january_boundary_uses_only_available_weekly_history():
    history = example_history(n=6, weeks=183)
    dates = pd.date_range("2020-07-06", periods=183, freq="W-MON").strftime("%Y-%m-%d")
    history.columns = KEYS + list(dates)
    history.loc[:, dates[-4:]] = 10.0
    history.loc[:, dates[-1]] = 2.0
    pred = forecast(history)
    assert pred.columns[-1] == "2024-04-01"
    factor = (3 * 2 + 5) / (3 * 8 + 5)
    expected = 0.75 * 2.0 + 0.25 * 8.0 * factor
    assert np.allclose(pred.iloc[:, 3:].to_numpy(), expected)
