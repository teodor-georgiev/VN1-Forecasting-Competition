"""Causality and safety contracts for OOF-gated simple forecast experts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import vn1_counterfactual_expert_gate as v


def history(weeks=170, n=40):
    keys = pd.DataFrame({
        "Client": np.arange(n) // 5,
        "Warehouse": np.arange(n) % 3,
        "Product": np.arange(n) + 100,
    })
    values = np.full((n, weeks), 2., dtype=float)
    values[0] = 30.
    values[1] = 20.
    values[2] = 15.
    values[-1] = 0.
    # Distinct previous-year and recent amounts; requires no future labels.
    values[0, 55:68] = 100.
    dates = pd.date_range("2020-07-06", periods=weeks, freq="W-MON")
    return pd.concat([keys, pd.DataFrame(values, columns=dates.strftime("%Y-%m-%d"))],
                     axis=1)


def completed(n, origins=(105,118), baseline=10.0):
    result = []
    for origin in origins:
        champ = np.full((n,13), baseline, dtype=float)
        truth = champ.copy()
        truth[0] = 30.0
        result.append({"origin":origin, "champion": champ, "truth":truth})
    return result


def test_simpler_expert_matches_only_observed_annual_weeks():
    a = np.arange(170, dtype=float).reshape(1,170)
    experts=v.simpler_experts(a)
    np.testing.assert_array_equal(experts["annual"].shape, (1,13))
    np.testing.assert_array_equal(
        experts["recent"], np.full((1,13),a[0,-13:].mean()))
    expected=0.5*a[:, -52:-39]+0.5*a[:,-13:].mean(axis=1,keepdims=True)
    np.testing.assert_allclose(experts["annual"], expected)
    with pytest.raises(ValueError,match="65"):
        v.simpler_experts(a[:,:60])


def test_OOF_only_completed_chronological_inputs():
    h=history().iloc[:,:3+131]
    current=completed(len(h))
    trusted, meta=v.oof_trust_masks(h,current)
    assert meta["origins_checked"]==[105,118]
    assert meta["validated_on_completed_oof_only"] is True
    assert trusted["recent"][0]
    bad=completed(len(h),origins=(118,105))
    with pytest.raises(ValueError, match="chronological"):
        v.oof_trust_masks(h,bad)
    bad=completed(len(h),origins=(118,131))
    with pytest.raises(ValueError, match="future"):
        v.oof_trust_masks(h,bad)


def test_future_mutation_cannot_change_expert_gate_or_forecasts():
    full=history()
    past=full.iloc[:,:3+131].copy()
    mutated=full.copy()
    mutated.iloc[:,3+131:] += 1e6
    records=completed(len(past))
    incumbent=np.full((len(past),13),11.)
    before,meta=v.candidate_forecasts(past,incumbent,records)
    after,_=v.candidate_forecasts(mutated.iloc[:,:3+131],incumbent,records)
    assert meta["top_decile_series"]>0
    for name in v.POLICIES:
        np.testing.assert_array_equal(before[name],after[name])


def test_exact_total_and_inactive_policy_invariance():
    h=history().iloc[:,:3+131]
    inc=np.full((len(h),13),11.)
    inc[-1]=0.
    candidates,meta=v.candidate_forecasts(h,inc,completed(len(h)))
    np.testing.assert_array_equal(candidates["incumbent"],inc)
    assert tuple(candidates)==v.POLICIES
    for name,pred in candidates.items():
        assert np.isfinite(pred).all() and (pred>=0).all()
        np.testing.assert_allclose(pred.sum(axis=0),inc.sum(axis=0),
                                   atol=1e-7,rtol=1e-10)
        np.testing.assert_array_equal(pred[-1], inc[-1])
    assert meta["oof_recent15_affected_series"] >= 1
    assert not np.array_equal(candidates["oof_recent15"],inc)


def test_no_eligibility_from_unseen_future_and_score_fallback(monkeypatch):
    h=history()
    n=len(h)
    monkeypatch.setattr(v.base, "_completed_records",
                        lambda *a,**k: completed(n,origins=(105,)))
    monkeypatch.setattr(v.base, "champion_forecast",
                        lambda *a,**k:(np.full((n,13),10.),{}))
    monkeypatch.setattr(v.persistent, "candidate_forecasts",
                        lambda *a,**k:({"group_unscaled25":np.full((n,13),10.)},{}))

    def fake_experts(past,inc,records):
        origin=past.shape[1]-3
        predictions={"incumbent":inc.copy()}
        for name in v.POLICIES[1:]:
            # The four historical folds are intentionally dissimilar:
            # early pooled train prefers a candidate, late holdout rejects it.
            x=np.full_like(inc,20. if origin==157 else 2.)
            if origin != 157:
                x[0]=30.
                x[1]=20.
                x[2]=15.
                x[-1]=0.
            predictions[name]=x
        return predictions, {}

    monkeypatch.setattr(v,"candidate_forecasts",fake_experts)
    result=v.historical_validation(h)
    assert result["training_origins"] == [118,131,144]
    assert result["holdout_origin"] == 157
    assert result["training_winner"] != "incumbent"
    assert result["selected"] == "incumbent"
    assert result["accepted"] is False
    assert result["used_phase1_labels"] is False
    assert result["used_phase2_labels"] is False
    assert result["holdout_reused_exploratory"] is True
