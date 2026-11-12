"""Chronology, eligibility, and fallback tests for persistent champion bias."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import vn1_persistent_bias_specialist as v


def history(n: int = 40, weeks: int = 170) -> pd.DataFrame:
    ids = pd.DataFrame({
        "Client": np.arange(n) // 5,
        "Warehouse": np.arange(n) % 3,
        "Product": np.arange(n) + 100,
    })
    dates = pd.date_range("2020-07-06", periods=weeks, freq="7D").strftime("%Y-%m-%d")
    levels = np.ones((n, weeks), dtype=float)
    levels[0] = 40.0
    levels[1] = 20.0
    levels[2] = 10.0
    levels[-1] = 0.0
    return pd.concat([ids, pd.DataFrame(levels, columns=dates)], axis=1)


def records(n: int, origins=(105, 118), *, reverse_second: bool = True) -> list[dict]:
    out = []
    for index, origin in enumerate(origins):
        champion = np.full((n, 13), 10.0)
        actual = champion.copy()
        actual[0, :] += 5.0  # persistent underforecast
        actual[1, :] += 4.0 if index == 0 or not reverse_second else -4.0
        out.append({"origin": origin, "champion": champion, "truth": actual})
    return out


def test_stable_bias_is_positive_but_reversed_bias_is_not_trusted():
    h = history().iloc[:, :3 + 131]
    effects = v.estimate_effects(h, records(len(h)))
    assert effects["series"][0] > 0
    assert effects["series"][1] == 0
    assert effects["top_volume"][0]
    assert not effects["top_volume"][-1]
    assert effects["active"][-1] == 0


def test_candidate_forecasts_preserve_identity_mass_and_nonnegative_output():
    h = history().iloc[:, :3 + 131]
    champion = np.full((len(h), 13), 10.0)
    champion[-1] = 0.0
    choices, details = v.candidate_forecasts(h, champion, records(len(h)))
    assert tuple(choices) == v.POLICIES
    np.testing.assert_array_equal(choices["champion"], champion)
    assert details["oof_origins"] == [105, 118]
    for key, forecast in choices.items():
        assert np.isfinite(forecast).all() and (forecast >= 0).all(), key
        assert forecast.shape == champion.shape
        if key != "group_unscaled25":
            np.testing.assert_allclose(
                forecast.sum(axis=0), champion.sum(axis=0), atol=1e-8)
        np.testing.assert_array_equal(forecast[-1], champion[-1])
    assert not np.array_equal(choices["sku_stable35"], champion)


def test_records_must_be_completed_ordered_and_full_grid():
    h = history().iloc[:, :3 + 131]
    with pytest.raises(ValueError, match="future"):
        v.estimate_effects(h, records(len(h), origins=(118, 131)))
    with pytest.raises(ValueError, match="order"):
        v.estimate_effects(h, records(len(h), origins=(118, 105)))
    with pytest.raises(ValueError, match="shape"):
        v.estimate_effects(h, records(len(h) - 1))
    with pytest.raises(ValueError, match="nonnegative"):
        bad = records(len(h))
        bad[0]["truth"][0, 0] = -2
        v.estimate_effects(h, bad)


def test_mutations_after_origin_do_not_change_features_or_candidates():
    full = history()
    prior = full.iloc[:, :3 + 131].copy()
    changed = full.copy()
    changed.iloc[:, 3 + 131:] += 100000.0
    before = v.estimate_effects(prior, records(len(prior)))
    after = v.estimate_effects(changed.iloc[:, :3 + 131], records(len(prior)))
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])


def test_historical_gate_rejects_candidate_when_last_holdout_regresses(monkeypatch):
    h = history()
    origins_seen = []

    def fake_records(data, cutoff, cache, **kwargs):
        return records(len(data), origins=(105,))  # completed by first validation

    def fake_champion(past, **kwargs):
        origins_seen.append(past.shape[1] - 3)
        return np.ones((len(past), 13)), {}

    def fake_candidates(past, champion, completed):
        pred = {key: champion.copy() for key in v.POLICIES}
        # Earlier windows: high predictions generally reduce error (observed >= 1);
        # but holdout intentionally uses worse predictions.
        multiplier = 2.0 if past.shape[1] - 3 == 157 else 1.1
        for name in v.POLICIES[1:]:
            pred[name] = champion * multiplier
        return pred, {}

    monkeypatch.setattr(v.base, "_completed_records", fake_records)
    monkeypatch.setattr(v.base, "champion_forecast", fake_champion)
    monkeypatch.setattr(v, "candidate_forecasts", fake_candidates)
    receipt = v.historical_validation(h)
    assert origins_seen == [118, 131, 144, 157]
    assert receipt["holdout_origin"] == 157
    assert receipt["accepted"] is False
    assert receipt["selected"] == "champion"
    assert receipt["target_phase2_seen"] is False
