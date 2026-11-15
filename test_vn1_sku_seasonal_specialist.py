"""Causality, conservation and selection tests for VN1 SKU-seasonal research."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import vn1_sku_seasonal_specialist as v


def history(n: int = 40, weeks: int = 170) -> pd.DataFrame:
    keys = pd.DataFrame({
        "Client": np.arange(n) // 5,
        "Warehouse": np.arange(n) % 3,
        "Product": np.arange(n) + 1000,
    })
    vals = np.full((n, weeks), 1., dtype=float)
    vals[0] = 40.
    vals[1] = 20.
    vals[2] = 10.
    vals[-1] = 0.
    # At origin 157, previous year's 13 future weeks differ from its
    # 13 prior weeks for the largest SKU.
    vals[0, 92:105] = 20.
    vals[0, 105:118] = 60.
    dates = pd.date_range("2020-07-06", periods=weeks, freq="7D")
    return pd.concat([
        keys, pd.DataFrame(vals, columns=dates.strftime("%Y-%m-%d"))
    ], axis=1)


def completed_records(n: int) -> list[dict]:
    records = []
    for origin in (105, 118):
        control = np.full((n, 13), 10., dtype=float)
        truth = control.copy()
        truth[0] += 3.
        records.append({"origin": origin, "champion": control, "truth": truth})
    return records


def test_only_past_weeks_can_change_annual_features():
    full = history()
    past = full.iloc[:, :3 + 157].copy()
    altered = full.copy()
    altered.iloc[:, 3 + 157:] += 1_000_000
    a = v.annual_signals(past)
    b = v.annual_signals(altered.iloc[:, :3 + 157])
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert a[0][0] > 0
    assert np.isfinite(a[1]).all()


def test_candidate_is_bounded_conserves_volume_and_leaves_inactive_alone():
    past = history().iloc[:, :3 + 157]
    current = np.full((len(past), 13), 10.)
    forecasts, diagnostic = v.make_candidates(
        past, current, completed_records(len(past)))
    assert tuple(forecasts) == v.CANDIDATES
    np.testing.assert_array_equal(forecasts["incumbent"], current)
    assert diagnostic["top_volume"] == 2
    assert diagnostic["completed_oof_origins"] == [105, 118]
    for name, pred in forecasts.items():
        assert pred.shape == current.shape
        assert np.isfinite(pred).all() and (pred >= 0).all()
        np.testing.assert_allclose(pred.sum(axis=0), current.sum(axis=0),
                                   atol=1e-8, rtol=0)
        np.testing.assert_array_equal(pred[-1], current[-1])
        np.testing.assert_array_equal(pred[3], current[3])
    assert not np.array_equal(forecasts["sku_annual10"], current)


def test_incomplete_oof_and_invalid_predictions_are_rejected():
    past = history().iloc[:, :3 + 131]
    control = np.full((len(past), 13), 10.)
    invalid = completed_records(len(past))
    invalid[1]["origin"] = 131  # its target horizon isn't observed
    with pytest.raises(ValueError, match="future"):
        v.make_candidates(past, control, invalid)
    with pytest.raises(ValueError, match="Invalid current"):
        v.make_candidates(past, control[:, :-1], completed_records(len(past)))


def test_historical_gate_rejects_failure_on_later_fold(monkeypatch):
    h = history()

    def mock_records(phase0, cutoff, cache, **kwargs):
        return completed_records(len(phase0))[:1]

    def mock_champion(past, **kwargs):
        return np.full((len(past), 13), 2.), {}

    def mock_incumbent(past, champion, records):
        return {"group_unscaled25": champion.copy()}, {}

    def mock_candidates(past, existing, records):
        pred = {"incumbent": existing.copy()}
        at_holdout = past.shape[1] - 3 == v.base.HOLDOUT_ORIGIN
        for name in v.CANDIDATES[1:]:
            if at_holdout:
                candidate = np.full_like(existing, 5.)
            else:
                # Synthetic training predictions match the known fixture
                # while the later holdout intentionally deteriorates.
                candidate = np.full_like(existing, 1.)
                candidate[0, :] = 40.
                candidate[1, :] = 20.
                candidate[2, :] = 10.
                candidate[-1, :] = 0.
            pred[name] = candidate
        return pred, {}

    monkeypatch.setattr(v.base, "_completed_records", mock_records)
    monkeypatch.setattr(v.base, "champion_forecast", mock_champion)
    monkeypatch.setattr(v.incumbent, "candidate_forecasts", mock_incumbent)
    monkeypatch.setattr(v, "make_candidates", mock_candidates)

    r = v.historical_validation(h)
    assert r["winner_training_only"] != "incumbent"
    assert r["selected"] == "incumbent" and r["accepted"] is False
    assert r["training_origins"] == [118, 131, 144]
    assert r["holdout_origin"] == 157
    assert r["used_phase1_labels"] is False
    assert r["used_phase2_labels"] is False


def test_short_history_is_rejected():
    with pytest.raises(ValueError, match="65 observed weeks"):
        v.annual_signals(history(weeks=64))
