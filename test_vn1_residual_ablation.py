"""Synthetic shape, causality, and selection checks for VN1 residual experiments."""
import numpy as np
import pandas as pd
import pytest

from vn1_from_scratch import HORIZON, KEYS
from vn1_residual_ablation import (
    ABLATIONS, _interaction_columns, _official_loss,
    ablation_features, apply_correction, fit_predict, group_correction,
    historical_origins, rolling_experiment, select_strength,
)


def synthetic_history(n=16, weeks=170):
    rng = np.random.default_rng(20261110)
    ids = pd.DataFrame({
        "Client": np.arange(n) // 4,
        "Warehouse": np.arange(n) // 2,
        "Product": 100 + np.arange(n),
    })
    dates = pd.date_range("2020-07-06", periods=weeks, freq="W-MON").strftime("%Y-%m-%d")
    levels = (2 + np.arange(n) % 8)[:, None]
    sales = rng.poisson(levels, size=(n, weeks)).astype(float)
    return pd.concat([ids, pd.DataFrame(sales, columns=dates)], axis=1)


def test_feature_ablations_are_precise_and_historical():
    history = synthetic_history(weeks=125)
    expected = {
        "baseline": ["Client", "Warehouse", "client_mean4", "warehouse_mean4"],
        "no_ids": ["client_mean4", "warehouse_mean4"],
        "no_group_stats": ["Client", "Warehouse"],
        "no_ids_or_stats": ["lag_1", "mean_13"],
        "interactions": ["Client", "Warehouse", "client_warehouse_id",
                         "client_warehouse_peer_mean13"],
    }
    future_mutation = history.copy()
    future_mutation.iloc[:, len(KEYS) + 104:] += 9999
    for variant in ABLATIONS:
        X, categories = ablation_features(history, 104, variant)
        altered, categories2 = ablation_features(future_mutation, 104, variant)
        pd.testing.assert_frame_equal(X, altered)
        assert X.shape[0] == len(history) * HORIZON
        assert categories == categories2
        assert set(expected[variant]).issubset(X.columns)
        assert all(x in X.columns for x in categories)
        if variant in ("no_group_stats", "no_ids_or_stats"):
            assert "statistical_baseline" not in X
            assert not any(c.startswith(("client_", "warehouse_", "product_")) for c in X)
        if variant == "no_ids_or_stats":
            assert not {"Client", "Warehouse", "client_mean4"}.intersection(X.columns)
    with pytest.raises(ValueError):
        ablation_features(history, 104, "unknown")


def test_peer_group_excludes_self_and_singletons():
    history = synthetic_history(n=8, weeks=105)
    # Force the first group to share a client+warehouse and the last row to
    # occupy a single group, checking the exact leave-one-out calculation.
    history.loc[0, ["Client", "Warehouse"]] = [77, 77]
    history.loc[1, ["Client", "Warehouse"]] = [77, 77]
    history.loc[7, ["Client", "Warehouse"]] = [77, 78]
    columns = _interaction_columns(history, 104)
    values = history.iloc[:, len(KEYS):len(KEYS)+104].to_numpy()
    assert columns["client_warehouse_count"][0] == 2
    assert columns["client_warehouse_count"][1] == 2
    assert columns["client_warehouse_peer_mean13"][0] == pytest.approx(values[1, -13:].mean())
    assert columns["client_warehouse_peer_mean13"][1] == pytest.approx(values[0, -13:].mean())
    assert columns["client_warehouse_peer_mean13"][7] == 0


def test_residual_prior_bounds_and_selected_strength():
    n, folds = 50, 6
    ids = pd.DataFrame({
        "Client": [0] * 25 + [1] * 25,
        "Warehouse": [0] * 25 + [1] * 25,
    })
    prediction = np.full((folds, n, HORIZON), 10.0)
    observed = prediction.copy()
    observed[:, :25, :] = 13.0
    observed[:, 25:, :] = 7.0
    factors = group_correction(ids, prediction[:2], observed[:2])
    assert factors.shape == (n,)
    assert (factors <= 1.1 + 1e-12).all()
    assert (factors >= 0.9 - 1e-12).all()
    assert factors[0] > 1.0 and factors[-1] < 1.0
    assert np.allclose(apply_correction(prediction[2], factors, 0.0), prediction[2])
    improved = apply_correction(prediction[2], factors, 1.0)
    assert _official_loss(improved, observed[2]) < _official_loss(prediction[2], observed[2])
    choice = select_strength(ids, prediction, observed, origin_dates=[
        "2022-07-03", "2022-10-02", "2023-01-01",
        "2023-04-02", "2023-07-02", "2023-10-01",
    ])
    assert choice["number_of_holdouts"] == 4
    assert choice["selected_strength"] in (0.5, 1.0)
    assert choice["pooled_scores"][str(choice["selected_strength"])] < choice["pooled_scores"]["0.0"]


def test_future_labels_do_not_affect_earlier_backtest_folds():
    n = 40
    ids = pd.DataFrame({"Client": np.arange(n)//20, "Warehouse": np.arange(n)//20})
    prediction = np.full((6, n, HORIZON), 10.0)
    actual = prediction.copy()
    actual[:, :20] = 12.0
    actual[:, 20:] = 8.0
    receipt = select_strength(ids, prediction, actual)
    mutated = actual.copy()
    mutated[-1] = 10000
    after = select_strength(ids, prediction, mutated)
    for name in ("0.0", "0.5", "1.0"):
        np.testing.assert_allclose(
            receipt["fold_scores"][name][:-1], after["fold_scores"][name][:-1]
        )


def test_january_gate_keeps_champion_without_earlier_january_fold():
    ids = pd.DataFrame({"Client": [0] * 25 + [1] * 25,
                        "Warehouse": [0] * 25 + [1] * 25})
    prediction = np.full((5, 50, HORIZON), 10.0)
    actual = prediction.copy()
    actual[:, :25] = 13.0
    actual[:, 25:] = 7.0
    dates = ["2022-07-03", "2022-10-02", "2023-04-02",
             "2023-07-02", "2023-10-01"]
    assert select_strength(ids, prediction, actual, origin_dates=dates,
                           final_month=1)["selected_strength"] == 0.0


def test_october_requires_a_completed_october_validation_fold():
    # The only October example is fold 1, which has <2 earlier training
    # folds; that is not enough evidence to switch at an October final origin.
    n = 50
    ids = pd.DataFrame({"Client": [0] * 25 + [1] * 25,
                        "Warehouse": [0] * 25 + [1] * 25})
    prediction = np.full((5, n, HORIZON), 10.0)
    actual = prediction.copy()
    actual[:, :25] = 13.0
    actual[:, 25:] = 7.0
    dates = ["2022-07-03", "2022-10-02", "2023-01-02",
             "2023-04-02", "2023-07-02"]
    unrestricted = select_strength(ids, prediction, actual)
    assert unrestricted["selected_strength"] > 0.0
    guarded = select_strength(ids, prediction, actual,
                              origin_dates=dates, final_month=10)
    assert guarded["selected_strength"] == 0.0
    assert guarded["seasonal_gate_month"] == 10
    with pytest.raises(ValueError, match="valid calendar"):
        select_strength(ids, prediction, actual,
                        origin_dates=dates, final_month=13)


def test_pipeline_runs_without_seeing_phase_target():
    history = synthetic_history(weeks=170)
    assert historical_origins(history) == [105, 118, 131, 144, 157]
    plain, corrected, report = rolling_experiment(history, trees=5,
                                                   max_folds=3, ablation_folds=1)
    assert plain.shape == corrected.shape == (len(history), HORIZON)
    assert np.isfinite(corrected).all() and (corrected >= 0).all()
    assert set(report["ablation_results"]) == set(ABLATIONS)
    assert all(t + HORIZON <= 170 for t in report["cv_origins"])
    assert report["features_tuned_with_phase_targets"] is False
    with pytest.raises(ValueError):
        historical_origins(history, max_folds=2)
    with pytest.raises(ValueError):
        apply_correction(plain, np.ones(len(plain)), 1.2)
    with pytest.raises(ValueError):
        fit_predict(history, trees=0)
