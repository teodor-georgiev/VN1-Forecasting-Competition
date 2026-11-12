"""Causal tests for VN1 hierarchical total-demand and product allocation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import vn1_hierarchical_demand as v


def history(n=40, weeks=183):
    ids = pd.DataFrame({
        "Client": np.arange(n) // 5,
        "Warehouse": np.arange(n) % 3,
        "Product": np.arange(n) + 100,
    })
    dates = pd.date_range("2020-07-06", periods=weeks, freq="7D").strftime("%Y-%m-%d")
    level = np.full((n, weeks), 8.0)
    level[0] = 40.0
    level[1] = 20.0
    level[-1] = 0.0
    return pd.concat([ids, pd.DataFrame(level, columns=dates)], axis=1)


def completed_records(n, origins=(105, 118)):
    records = []
    for origin in origins:
        pred = np.full((n, 13), 8.0)
        actual = pred.copy()
        actual[:5] += 3.0  # stable positive client demand bias
        actual[0] += 4.0  # stable top-volume SKU bias
        records.append({"origin": origin, "champion": pred, "truth": actual})
    return records


def test_decomposition_matches_official_absolute_plus_bias_formula():
    target = np.array([[1.0, 7.0]])
    forecast = np.array([[2.0, 4.0]])
    metric = v.components(forecast, target)
    assert metric["absolute"] == pytest.approx(4.0 / 8.0)
    assert metric["signed"] == pytest.approx(-2.0 / 8.0)
    assert metric["score"] == pytest.approx(6.0 / 8.0)


def test_no_future_labels_or_duplicate_oof_origins():
    data = history().iloc[:, :3 + 131]
    with pytest.raises(ValueError, match="future"):
        v.historical_bias(data, completed_records(len(data), origins=(118, 131)))
    with pytest.raises(ValueError, match="chronological"):
        v.historical_bias(data, completed_records(len(data), origins=(118, 105)))
    with pytest.raises(ValueError, match="shape"):
        v.historical_bias(data, completed_records(len(data) - 1))


def test_bilevel_preserves_inactive_fallback_and_known_client_totals():
    data = history().iloc[:, :3 + 131]
    n = len(data)
    champion = np.full((n, 13), 8.0)
    champion[-1, :] = 3.0  # inactive product must not be modified
    recs = completed_records(n)
    choices, info = v.candidates(data, champion, recs)
    assert tuple(choices) == v.POLICIES
    np.testing.assert_array_equal(choices["champion"], champion)
    assert info["oof_origins"] == [105, 118]
    estimates = v.historical_bias(data, recs)
    active = estimates["active"]
    for p in choices.values():
        assert p.shape == champion.shape
        assert np.isfinite(p).all() and (p >= 0).all()
        np.testing.assert_array_equal(p[~active], champion[~active])
    # Stage 1 sets the client active-pool totals. The two hierarchy
    # alternatives redistribute within the same totals.
    for client in data["Client"].unique():
        mask = (data["Client"].to_numpy() == client) & active
        if not mask.any():
            continue
        expected = champion[mask].sum(axis=0) * (
            1.0 + 0.25 * (0.75 * estimates["client"][mask][0]
                          + 0.25 * estimates["global"]))
        for name in ("hier_clientwarehouse25", "hier_sku25"):
            np.testing.assert_allclose(choices[name][mask].sum(axis=0), expected,
                                       rtol=0, atol=1e-8)
    assert not np.array_equal(choices["hier_sku25"], choices["hier_clientwarehouse25"])


def test_adding_unobserved_future_does_not_change_bias_features():
    full = history()
    before = full.iloc[:, :3 + 131]
    after = full.copy()
    after.iloc[:, 3 + 131:] *= 1000
    a = v.historical_bias(before, completed_records(len(before)))
    b = v.historical_bias(after.iloc[:, :3 + 131], completed_records(len(before)))
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_phase1_is_separate_untuned_acceptance_gate(monkeypatch):
    data = history()
    n = len(data)
    dates = pd.date_range(pd.Timestamp(data.columns[-1]) + pd.Timedelta(weeks=1),
                          periods=13, freq="7D").strftime("%Y-%m-%d")
    following = pd.concat([
        data.iloc[:, :3],
        pd.DataFrame(np.ones((n, 13)), columns=dates),
    ], axis=1)
    origins = []

    monkeypatch.setattr(v.base, "_completed_records",
                        lambda *args, **kwargs: completed_records(n, origins=(105,)))
    def champion(past, **kwargs):
        origins.append(past.shape[1] - 3)
        return np.full((n, 13), 0.8), {}
    monkeypatch.setattr(v.base, "champion_forecast", champion)

    def choices(past, champ, recs):
        at = past.shape[1] - 3
        forecasts = {"champion": champ.copy()}
        forecasts["persistent_unscaled25"] = champ * (
            1.0 if at == len(data.columns)-3 else 1.1)
        for name in v.POLICIES[2:]:
            forecasts[name] = champ * (3.0 if at == len(data.columns)-3 else 1.2)
        return forecasts, {}
    monkeypatch.setattr(v, "candidates", choices)
    report = v.validate(data, following)
    assert origins == [118, 131, 144, 157, 183]
    assert report["winner_training_only"] not in ("champion", "persistent_unscaled25")
    assert report["holdout_scores"][report["winner_training_only"]]["score"] < report["holdout_scores"]["persistent_unscaled25"]["score"]
    assert report["phase1_scores"][report["winner_training_only"]]["score"] > report["phase1_scores"]["persistent_unscaled25"]["score"]
    assert report["accepted"] is False
    assert report["selected"] == "persistent_unscaled25"
    assert report["phase2_targets_read"] is False
    assert report["reference_policy"] == "persistent_unscaled25"


def test_invalid_phase1_row_or_date_alignment_fails_before_scoring(monkeypatch):
    data = history()
    n = len(data)
    dates = pd.date_range(pd.Timestamp(data.columns[-1]) + pd.Timedelta(weeks=1),
                          periods=13, freq="7D").strftime("%Y-%m-%d")
    phase1 = pd.concat([data.iloc[:, :3],
                        pd.DataFrame(np.ones((n, 13)), columns=dates)], axis=1)
    phase1.loc[0, "Product"] += 10000
    with pytest.raises(ValueError, match="Phase 1"):
        v.validate(data, phase1)


def test_hierarchical_fallback_when_active_client_has_zero_forecast():
    data = history().iloc[:, :3 + 131]
    n = len(data)
    champion = np.full((n, 13), 10.0)
    champion[:5] = 0.0
    outputs, _ = v.candidates(data, champion, completed_records(n))
    for choice in outputs.values():
        assert np.isfinite(choice).all()
        np.testing.assert_array_equal(choice[:5], np.zeros((5, 13)))
