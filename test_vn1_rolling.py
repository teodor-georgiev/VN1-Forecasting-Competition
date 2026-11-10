"""Leakage-safe synthetic tests for experimental rolling VN1 model selection."""
import numpy as np
import pytest

from vn1_adaptive import rolling_origin_validation, series_switch_weights
from vn1_from_scratch import HORIZON


def example_panels(folds=6, products=12):
    """All forecasts are shaped (fold, series, candidate, 13-week horizon)."""
    rng = np.random.default_rng(20261110)
    truth = rng.poisson(10, size=(folds, products, HORIZON)).astype(float)
    preds = np.stack([truth + 5.0, truth.copy()], axis=2)
    segments = np.zeros((folds, products), dtype=np.int8)
    return preds, truth, segments


def test_product_switch_requires_consistent_evidence_and_shrinks_to_champion():
    pred, truth, _ = example_panels()
    names = ["champion", "better"]
    weights = series_switch_weights(pred[:4], truth[:4], names)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    np.testing.assert_allclose(weights[:, 0], 0.25)
    np.testing.assert_allclose(weights[:, 1], 0.75)

    # Sparse and inconsistent series must not switch despite lucky folds.
    truth[:, 0, :] = 0
    pred[:, 0, :, :] = 0
    pred[2:, 1, 1, :] = truth[2:, 1, :] + 20
    guarded = series_switch_weights(pred, truth, names)
    np.testing.assert_array_equal(guarded[0], [1.0, 0.0])
    np.testing.assert_array_equal(guarded[1], [1.0, 0.0])


def test_expanding_window_never_trains_on_heldout_or_later_labels():
    pred, truth, segments = example_panels()
    names = ["champion", "better"]
    first = rolling_origin_validation(pred, truth, segments, names)
    assert first["selected_mode"] in ("global", "segment", "product", "series_switch")
    assert first["validation_folds"] == 4
    assert first["scores"]["champion"] > first["scores"]["series_switch"]
    assert all(fold["fold_index"] >= 2 for fold in first["folds"])

    # A changed future target cannot affect the earlier holdout fold's scores.
    changed_truth = truth.copy()
    changed_truth[-1] = changed_truth[-1] + 1000
    changed = rolling_origin_validation(pred, changed_truth, segments, names)
    assert first["folds"][0]["scores"] == changed["folds"][0]["scores"]
    assert first["folds"][1]["scores"] == changed["folds"][1]["scores"]
    assert first["folds"][2]["scores"] == changed["folds"][2]["scores"]


def test_rolling_gate_rejects_inconsistent_and_one_fold_validation():
    pred, truth, segments = example_panels()
    names = ["champion", "specialist"]
    # Specialist is excellent early, but catastrophic in the final two folds.
    pred[-2:, :, 1, :] = truth[-2:] + 50
    result = rolling_origin_validation(pred, truth, segments, names)
    assert result["selected_mode"] == "champion"
    short = rolling_origin_validation(pred[:3], truth[:3], segments[:3], names)
    assert short["validation_folds"] == 1
    assert short["selected_mode"] == "champion"


def test_invalid_rolling_inputs_are_rejected():
    pred, truth, segments = example_panels()
    with pytest.raises(ValueError):
        rolling_origin_validation(pred, truth, segments[:-1], ["champion", "better"])
    with pytest.raises(ValueError):
        rolling_origin_validation(pred[:2], truth[:2], segments[:2], ["champion", "better"])
    with pytest.raises(ValueError):
        series_switch_weights(pred[:1], truth[:1], ["champion", "better"])
    with pytest.raises(ValueError):
        series_switch_weights(pred, truth, ["a", "b"])
