"""Tests for strictly chronological, volume-conserving dormant-demand experiments."""
import numpy as np
import pandas as pd
import pytest

import vn1_soft_dormancy as s


def sample_history(n=12, periods=170, seed=43):
    dates = pd.date_range("2020-07-06", periods=periods, freq="7D").strftime("%Y-%m-%d")
    ids = pd.DataFrame({
        "Client": np.repeat(np.arange(n // 3),3),
        "Warehouse": np.tile([1,2,3],n//3),
        "Product": np.arange(100,100+n),
    })
    rng = np.random.default_rng(seed)
    y = rng.poisson(np.arange(n)[:, None] % 4 + 1, size=(n,periods)).astype(float)
    y[0, -52:] = 0
    y[1, -26:] = 0
    y[1,-52:-39]=0
    y[2,-13:]=0
    y[2,-52:-39]=0
    return pd.concat([ids, pd.DataFrame(y,columns=dates)],axis=1)


def test_weekly_mass_conservation_and_positive_outputs():
    base=np.arange(1,1+4*s.HORIZON,dtype=float).reshape(4,s.HORIZON)
    mask=np.array([True,False,True,False])
    out=s.redistribute(base,mask,method="weekly")
    np.testing.assert_allclose(out[mask],0)
    np.testing.assert_allclose(out.sum(axis=0),base.sum(axis=0))
    assert np.isfinite(out).all()


def test_soft_strength_and_both_boundary_cases():
    base=np.ones((3,s.HORIZON))
    mask=np.array([True,False,False])
    o=s.redistribute(base,mask,strength=0.5)
    assert np.all(o[0]==0.5)
    assert np.isclose(o.sum(),base.sum())
    np.testing.assert_array_equal(s.redistribute(base,mask,0),base)
    np.testing.assert_array_equal(s.redistribute(base,np.ones(3,dtype=bool)),base)
    with pytest.raises(ValueError):
        s.redistribute(base,mask,strength=1.1)


def test_client_and_recent_rebalance_never_cross_user_group():
    h=sample_history()
    b=np.ones((len(h),s.HORIZON))
    mask=np.zeros(len(h),dtype=bool)
    mask[0]=True
    out=s.redistribute(b,mask,method="client",history=h)
    np.testing.assert_allclose(out.sum(),b.sum())
    np.testing.assert_array_equal(out[3:],b[3:])
    recent=s.redistribute(b,mask,method="recent",history=h)
    np.testing.assert_allclose(recent.sum(),b.sum())
    assert (recent >= 0).all()
    with pytest.raises(ValueError):
        s.redistribute(b,mask,method="client",history=h.iloc[:-1])


def test_tsb_and_zero_rules_are_finite_and_volume_preserving():
    h=sample_history()
    b=np.full((len(h),s.HORIZON),7.,dtype=float)
    preds,masks=s.candidate_forecasts(h,b)
    assert tuple(preds)==s.CANDIDATES
    assert masks["quiet26_annual"]>=1
    assert all(p.shape==b.shape and np.isfinite(p).all() for p in preds.values())
    for name,pred in preds.items():
        np.testing.assert_allclose(pred.sum(),b.sum(),rtol=1e-10,
                                   err_msg=name)


def test_future_sales_mutation_outside_origin_does_not_change_candidates():
    h=sample_history()
    origin=118
    early=h.iloc[:,:3+origin].copy()
    p1,_=s.candidate_forecasts(early,np.full((len(h),s.HORIZON),1.0))
    later=h.copy()
    later.iloc[:,3+origin:]+=9999
    p2,_=s.candidate_forecasts(later.iloc[:,:3+origin],np.full((len(h),s.HORIZON),1.0))
    for name in s.CANDIDATES:
        np.testing.assert_array_equal(p1[name],p2[name])


def test_selection_uses_only_phase0_complete_origins(monkeypatch):
    h=sample_history()
    calls=[]
    def fake_baseline(past,trees):
        calls.append(len(past.columns)-3)
        return np.full((len(past),s.HORIZON),2.)
    monkeypatch.setattr(s,"incumbent",fake_baseline)
    result=s.select_on_phase0(h,5)
    assert calls==list(s.ORIGINS)
    assert result["origins_train"]==list(s.ORIGINS[:-1])
    assert result["origin_holdout"]==157
    assert result["selected"] in s.CANDIDATES
    assert result["minimum_absolute_gain"]==0.001
    assert len(result["per_origin_scores"])==len(s.ORIGINS)


def test_freeze_without_phase_truth_and_score_checks_hash(monkeypatch,tmp_path):
    h=sample_history()
    h.to_csv(tmp_path/"Phase 0 - Sales.csv",index=False)
    monkeypatch.setattr(s,"select_on_phase0",lambda phase0,trees:{
        "selected":"incumbent","accepted":False,"origins_train":[105,118,131,144],
        "origin_holdout":157})
    monkeypatch.setattr(s,"incumbent",lambda history,trees:
                        np.ones((len(history),s.HORIZON)))
    output=tmp_path/"freeze"
    receipt=s.freeze(tmp_path,output,phase=1,trees=4)
    assert receipt["phase_truth_was_not_loaded_in_freeze"]
    assert receipt["forecast_candidates"][-1]=="phase0_selected"
    assert (output/"forecast-frozen.npz").exists()
    with pytest.raises((OSError,FileNotFoundError)):
        s.score(tmp_path,output)
    frozen=output/"forecast-frozen.npz"
    with frozen.open("ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError, match="changed"):
        s.score(tmp_path,output)
