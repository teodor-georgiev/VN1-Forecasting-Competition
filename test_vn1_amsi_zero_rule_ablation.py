"""Policy ablation tests: group isolation, no future leakage, numerical safety."""
import numpy as np
import pandas as pd
import pytest

from vn1_amsi_zero_rule_ablation import (
    HORIZON, POLICIES, ORIGINS, _metric, candidates, rules, select_phase0,
)


def synthetic():
    date = pd.date_range("2020-07-06", periods=170, freq="7D")
    columns = list(date.strftime("%Y-%m-%d"))
    ids = pd.DataFrame({
        "Client": [1, 1, 1, 2, 2, 3],
        "Warehouse": [10, 10, 10, 10, 10, 11],
        "Product": [101, 102, 103, 201, 202, 301],
    })
    sales = np.zeros((6, 170), dtype=float)
    sales[:3, 30:100] = 2  # Group 1 warehouse 10 closed, used to sell.
    sales[3, -40:] = 3      # Same warehouse ID but another client's active group.
    sales[5, -52:-39] = 3  # Prior-year same-calendar activity protects this SKU.
    return pd.concat([ids, pd.DataFrame(sales, columns=columns)], axis=1)


def test_key_is_client_warehouse_pair_not_warehouse_alone():
    h = synthetic()
    masks = rules(h)
    assert masks["warehouse_closed26"].tolist() == [
        True, True, True, False, False, False
    ]
    assert masks["all_history_zero"][4]
    assert masks["inactive52"][0]
    assert not masks["inactive26_annual_zero"][5]


def test_empty_and_sparse_rules_never_use_future_labels():
    h = synthetic()
    original = rules(h.iloc[:, :3 + 131].copy())
    altered = h.copy()
    altered.iloc[:, 3 + 131:] += 1000  # Future actuals not allowed at week 131.
    revised = rules(altered.iloc[:, :3 + 131].copy())
    for name in original:
        np.testing.assert_array_equal(original[name], revised[name])


def test_hard_rule_changes_only_eligible_rows_and_preserves_input():
    h = synthetic()
    base = np.ones((len(h), HORIZON), dtype=float)
    out, counts = candidates(h, base)
    assert tuple(out) == POLICIES
    assert np.all(base == 1)
    assert np.all(out["warehouse_closed26"][:3] == 0)
    assert np.all(out["warehouse_closed26"][3:] == 1)
    assert np.all(out["all_history_zero"][4] == 0)
    assert counts["all_history_zero"] >= 1
    assert all(p.shape == (6, HORIZON) and np.isfinite(p).all() for p in out.values())


def test_selection_folds_complete_only_historical_phase0(monkeypatch):
    h = synthetic()
    calls = []
    def fake_incumbent(past, trees):
        calls.append(past.shape[1] - 3)
        return np.full((len(past), HORIZON), 1.0)
    monkeypatch.setattr("vn1_amsi_zero_rule_ablation.incumbent", fake_incumbent)
    receipt = select_phase0(h, trees=3)
    assert calls == list(ORIGINS)
    assert receipt["train_origins"] == [131, 144]
    assert receipt["holdout_origin"] == 157
    assert receipt["selected"] in POLICIES
    assert receipt["min_gain"] > 0
    assert all(np.isfinite(x) for x in receipt["holdout_scores"].values())


def test_vn1_metric_matches_absolute_error_plus_bias():
    true = np.array([[1.0, 2.0]])
    pred = np.array([[1.0, 5.0]])
    assert _metric(pred,true) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        _metric(pred, np.zeros_like(true))
