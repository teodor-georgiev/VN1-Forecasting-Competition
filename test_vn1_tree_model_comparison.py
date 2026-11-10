"""Checks for comparable time-safe tree-model validation."""
import numpy as np
import pandas as pd
import pytest

from vn1_from_scratch import KEYS, HORIZON
from vn1_tree_model_comparison import (
    ORIGINS, MODEL_NAMES, select_historical_series, feature_panels,
    model_inputs, evaluate_model, official_loss, run,
)


def history_fixture(n=18, weeks=170):
    rng = np.random.default_rng(3432)
    ids = pd.DataFrame({"Client": np.arange(n)//3, "Warehouse": np.arange(n)%3,
                        "Product": np.arange(n)+100})
    vals = rng.poisson(3 + np.arange(n)[:,None]%7, size=(n,weeks)).astype(float)
    vals[0] = 0.
    dates = pd.date_range("2020-07-06", periods=weeks, freq="W-MON").strftime("%Y-%m-%d")
    return pd.concat([ids, pd.DataFrame(vals, columns=dates)], axis=1)


def test_selection_is_independent_of_future_labels():
    hist = history_fixture()
    chosen = select_historical_series(hist, 12)
    mutated = hist.copy()
    mutated.iloc[:, len(KEYS)+65:] += 10_000
    np.testing.assert_array_equal(chosen, select_historical_series(mutated, 12))
    assert len(chosen) == 12


def test_same_features_and_completed_origin_boundaries():
    hist = history_fixture()
    chosen = select_historical_series(hist, 8)
    cache = feature_panels(hist, chosen, [105, 118, 131])
    for origin in [105, 118, 131]:
        assert cache["panels"][origin].shape[0] == 8 * HORIZON
    assert max(t for t in cache["train_origins"] if t+HORIZON <= 131) <= 118
    assert all(np.isfinite(x.to_numpy(dtype=float)).all() for x in cache["panels"].values())
    for name in MODEL_NAMES:
        X = model_inputs(cache["panels"][105], name)
        if name == "catboost_poisson":
            assert X["Client"].dtype == object
        if name == "xgboost_poisson":
            assert isinstance(X["Client"].dtype, pd.CategoricalDtype)
    with pytest.raises(ValueError):
        feature_panels(hist, chosen, [170])


def test_all_models_predict_and_score_same_windows():
    hist = history_fixture()
    results = run(hist, max_series=10, trees=4)
    assert set(results["results"]) == set(MODEL_NAMES)
    assert results["january_origin"] == 131
    assert results["origins"] == list(ORIGINS)
    for detail in results["results"].values():
        assert detail["sampled_series"] == 10
        assert len(detail["fold_scores"]) == 5
        assert np.isfinite(detail["pooled_vn1"])
        for fold in detail["fold_scores"]:
            assert max(fold["train_origins"])+HORIZON <= fold["origin"]
    assert results["no_competition_targets_used"]


def test_official_score_includes_bias_penalty():
    predicted = np.array([[2.,2.,2.]])
    target = np.array([[1.,2.,3.]])
    assert official_loss(predicted,target) == pytest.approx(2/6)
    with pytest.raises(ValueError):
        official_loss(predicted, np.ones((2,3)))
