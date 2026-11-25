import numpy as np
import pytest

import vn1_horizon_shape_oof as v
from test_vn1_persistent_bias_specialist import history


def fixture():
    h = history(40, 131)
    # Historical demand is consistently stronger during the first lead week.
    h.iloc[0, 3+105] = 4000.0
    h.iloc[0, 3+118] = 4000.0
    y = v.base._history_matrix(h)
    records = [{"origin": o, "champion": np.full((len(y), 13), 10.0),
                "truth": y[:, o:o+13].copy()} for o in (105, 118)]
    return h, np.full((len(y), 13), 10.0), records


def test_shape_correction_conserves_each_sku_and_protects_inactive():
    h, p, records = fixture()
    p[1] = 0.0
    out = v.candidate_forecasts(h, p, records)
    assert tuple(out) == v.POLICIES
    np.testing.assert_array_equal(out["incumbent"], p)
    for x in out.values():
        np.testing.assert_allclose(x.sum(axis=1), p.sum(axis=1))
        np.testing.assert_array_equal(x[3:], p[3:])
        np.testing.assert_array_equal(x[1], p[1])
        assert np.isfinite(x).all() and (x >= 0).all()
    assert out["group30"][0, 0] > p[0, 0]
    assert out["group30"][0, -1] < p[0, -1]


def test_future_misaligned_and_duplicate_records_fail_closed():
    h, p, records = fixture()
    with pytest.raises(ValueError, match="future"):
        v.candidate_forecasts(h, p, [dict(records[-1], origin=119)])
    with pytest.raises(ValueError, match="chronological"):
        v.candidate_forecasts(h, p, records + records[-1:])
    bad = dict(records[-1], truth=records[-1]["truth"] + 1.)
    with pytest.raises(ValueError, match="differs"):
        v.candidate_forecasts(h, p, [records[0], bad])


def test_level_only_bias_does_not_change_shape():
    p = [np.full((2, 13), 10.0)]
    a = [p[0] * 2.]
    np.testing.assert_allclose(v._shape_rate(p, a, np.zeros(2, dtype=int), 2500.), 0.)


def test_historical_gate_rejects_later_regression(monkeypatch):
    h = history()
    monkeypatch.setattr(v.base, "_completed_records", lambda *a, **kw: [])
    monkeypatch.setattr(v.base, "champion_forecast", lambda past, **kw: (np.ones((len(past), 13)), {}))
    monkeypatch.setattr(v.persistent, "candidate_forecasts", lambda past, p, r: ({"group_unscaled25": p}, {}))
    def fake(past, p, records):
        y = v.base._history_matrix(past)
        alt = np.repeat(y[:, -1:], 13, axis=1)
        if y.shape[1] == v.base.HOLDOUT_ORIGIN:
            alt = p * 100.
        return {"incumbent": p, **{name: alt for name in v.POLICIES[1:]}}
    monkeypatch.setattr(v, "candidate_forecasts", fake)
    result = v.historical_validation(h)
    assert result["training_winner"] != "incumbent"
    assert not result["accepted"]
    assert result["selected"] == "incumbent"
    assert not result["used_phase2_labels"]


def test_calendar_uses_matching_observed_weeks_and_shrinks_groups():
    h, p, records = fixture()
    # A seasonal demand peak one and two years before the forecast origin.
    h.iloc[0, 3+131-52+5] = 100000.0
    h.iloc[0, 3+131-104+5] = 100000.0
    y = v.base._history_matrix(h)
    shares = v._calendar_shares(h, y)
    np.testing.assert_allclose(shares.sum(axis=1), 1.0)
    assert np.argmax(shares[0]) == 5
    result = v.candidate_forecasts(h, p, records)
    assert result["calendar30"][0, 5] > result["calendar15"][0, 5] > p[0, 5]
    np.testing.assert_allclose(result["calendar30"].sum(axis=1), p.sum(axis=1))
    np.testing.assert_array_equal(result["calendar30"][3:], p[3:])
    # No observed matched demand means no artificial seasonal profile.
    assert v._calendar_shares(h, np.zeros_like(y)) is None


def test_calendar_can_redistribute_into_zero_week_without_increasing_total():
    h, p, records = fixture()
    p[0, 0] = 0.
    result = v.candidate_forecasts(h, p, records)
    assert result["calendar30"][0, 0] > 0.
    np.testing.assert_allclose(result["calendar30"].sum(axis=1), p.sum(axis=1))
