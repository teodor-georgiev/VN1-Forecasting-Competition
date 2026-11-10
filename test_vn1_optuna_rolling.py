"""Regression checks for VN1 time-separated Optuna trials."""
import numpy as np
import pandas as pd
import pytest

from vn1_from_scratch import KEYS, HORIZON
from vn1_optuna_rolling import (
    choose_series, valid_origins, score_vn1, setup_cache, score_folds,
    trial_params, optimize,
)
from vn1_lgbm_ensemble import LGBM_PARAMS


def fixture(series=16, weeks=170):
    rng = np.random.default_rng(438)
    dates = pd.date_range("2020-07-06", periods=weeks, freq="W-MON").strftime("%Y-%m-%d")
    ids = pd.DataFrame({"Client": np.arange(series) // 4,
                        "Warehouse": np.arange(series) % 4,
                        "Product": np.arange(series) + 100})
    levels = 1.0 + np.arange(series)[:, None] % 5
    values = rng.poisson(levels, size=(series, weeks)).astype(float)
    values[0, :] = 0
    return pd.concat([ids, pd.DataFrame(values, columns=dates)], axis=1)


def test_fixed_historical_selection_and_holdout():
    history = fixture()
    a = choose_series(history, 8)
    np.testing.assert_array_equal(a, choose_series(history, 8))
    assert len(a) == 8
    assert valid_origins(history) == [105, 118, 131, 144, 157]
    mutated = history.copy()
    mutated.iloc[:, len(KEYS) + 65:] += 100000
    np.testing.assert_array_equal(choose_series(mutated, 8), a)
    with pytest.raises(ValueError):
        valid_origins(fixture(weeks=125))


def test_historical_feature_and_label_boundary():
    history = fixture()
    selection = choose_series(history, 8)
    cache = setup_cache(history, selection, [105, 118, 131, 144, 157])
    params = {**LGBM_PARAMS, "n_estimators": 5, "n_jobs": 1}
    result = score_folds(cache, params, [105, 131])
    assert np.isfinite(result["pooled_vn1_score"])
    assert all(max(f["train_origins"]) + HORIZON <= f["origin"] for f in result["folds"])
    changed = history.copy()
    changed.iloc[:, len(KEYS) + 131:] += 100000
    same = setup_cache(changed, selection, [105, 118, 131])
    for origin in [65, 78, 91, 105, 118, 130, 131]:
        pd.testing.assert_frame_equal(cache["features"][origin], same["features"][origin])


def test_two_trial_optuna_smoke_with_disjoint_holdout():
    result = optimize(fixture(), trials=2, max_series=8, max_trees=8)
    assert result["number_of_trials"] == 2
    assert 131 in result["tuning_origins"]
    assert result["holdout_origins"] == [157]
    assert result["no_phase1_or_phase2_labels_used"] is True
    assert len(result["trials"]) == 2
    assert all(t["state"] == "COMPLETE" for t in result["trials"])
    for name in ["baseline_tuning", "tuned_tuning", "baseline_holdout", "tuned_holdout"]:
        assert np.isfinite(result[name]["pooled_vn1_score"])


def test_vn1_scoring_matches_bias_adjusted_wape():
    actual = np.array([1, 3, 6.])
    predicted = np.array([2, 2, 5.])
    assert score_vn1(predicted, actual) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        score_vn1(np.array([1.]), actual)
