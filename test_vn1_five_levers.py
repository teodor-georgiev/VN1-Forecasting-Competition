"""Synthetic regression tests for five fixed VN1 research levers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import vn1_five_levers as v


def history(n=18,weeks=170,seed=13):
    rng=np.random.default_rng(seed)
    ids=pd.DataFrame({
        "Client":np.arange(n)//3,
        "Warehouse":np.arange(n)%3,
        "Product":np.arange(n)+100,
    })
    dates=pd.date_range("2020-07-06",periods=weeks,freq="7D").strftime("%Y-%m-%d")
    rates=np.array([0,.1,.5,1.5,4,15]*(n//6))
    y=rng.poisson(rates[:,None],size=(n,weeks)).astype(float)
    y[0,:]=0
    y[2,:40]=0
    y[2,40:]=rng.poisson(2.,size=weeks-40)
    return pd.concat([ids,pd.DataFrame(y,columns=dates)],axis=1)


def prices_for(h):
    p=h.copy()
    rng=np.random.default_rng(4)
    values=10+rng.uniform(0,7,size=(len(h),len(h.columns)-3))
    values[1,:]=np.nan
    values[2,:20]=0
    p.iloc[:,3:]=values
    return p


def test_family_registry_covers_all_five_without_price_fake():
    assert set(v.FAMILIES)=={"loss","targets","price","hierarchy","trim"}
    assert set(sum(v.FAMILIES.values(),[]))==set(v.POLICIES)-{"learned_activity"}
    assert len(v.MODEL_VARIANTS)==8


def test_price_origin_protects_future_and_missing():
    h=history()
    prices=prices_for(h)
    first=v._price_features(prices,105)
    updated=prices.copy()
    updated.iloc[:,3+105:]+=100000
    second=v._price_features(updated,105)
    for k in first:
        np.testing.assert_array_equal(first[k],second[k])
    assert first["price_missing"][1]==1
    assert np.isfinite(np.stack(list(first.values()))).all()
    with pytest.raises(ValueError):
        v._price_features(prices,171)


def test_price_read_resorts_identifiers_without_leaking(tmp_path):
    h=history()
    p=prices_for(h)
    shuffled=p.sample(frac=1,random_state=21)
    path=tmp_path/"Phase 0 - Price.csv"
    shuffled.to_csv(path,index=False)
    aligned=v.read_price(path,h)
    assert aligned[v.KEYS].equals(h[v.KEYS])
    np.testing.assert_allclose(aligned.iloc[2,3:20].to_numpy(dtype=float),
                               p.iloc[2,3:20].to_numpy(dtype=float),equal_nan=True)
    truncated=aligned.iloc[:,:3+105]
    assert len(v._X(h,105,truncated).columns)>len(v._X(h,105,None).columns)
    with pytest.raises(ValueError):
        v._aligned_prices(h,prices_for(h).iloc[::-1].reset_index(drop=True))


def test_objectives_and_transforms_with_nonnegative_predictions():
    h=history()
    # Use a small synthetic model, but fit every real objective/transform.
    train=v._fit_dataset(h.iloc[:,:3+105],None)
    X,y,folds=train
    assert folds==[65,78,91]
    p=v._X(h,105,None)
    for transform in ("raw","log1p","sqrt","season_norm"):
        coded=v._transform_target(y,X,transform)
        assert np.isfinite(coded).all()
        recovered=v._invert_target(coded,X,transform)
        np.testing.assert_allclose(recovered,y,atol=1e-4,rtol=1e-4)
    base=np.ones((len(h),13))
    out=v._train_predict_variants(h.iloc[:,:3+105],base,prices=None,trees=6)
    assert set(out)==set(v.FAMILIES["loss"]+v.FAMILIES["targets"])
    for name,pred in out.items():
        assert np.isfinite(pred).all() and (pred>=0).all()
        np.testing.assert_allclose(pred.sum(axis=0),base.sum(axis=0),err_msg=name)


def test_all_five_families_run_with_synthetic_prices(monkeypatch):
    h=history()
    p=prices_for(h)
    base=np.full((len(h),13),3.0)
    monkeypatch.setattr(v,"predict_activity",
                        lambda hist,estimators:(
                            np.linspace(.05,.99,len(hist)),{"p50":.75}))
    result,meta=v.candidates(h.iloc[:,:3+105],base,
                             prices=p.iloc[:,:3+105],trees=6,activity_trees=6)
    assert tuple(result)==v.POLICIES
    assert meta["prices_available"]
    assert len(result)==11
    for name,forecast in result.items():
        assert forecast.shape==base.shape
        np.testing.assert_allclose(forecast.sum(axis=0),base.sum(axis=0),
                                   err_msg=name)


def test_hierarchy_never_modifies_weekly_total_or_series_order():
    h=history()
    base=np.arange(1,len(h)*13+1,dtype=float).reshape(len(h),13)
    h2=h.copy()
    h2.iloc[:,3+144:]+=111111
    a=v._hierarchy(h.iloc[:,:3+144],base)
    b=v._hierarchy(h2.iloc[:,:3+144],base)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_allclose(a.sum(axis=0),base.sum(axis=0))
    np.testing.assert_array_equal(base,np.arange(1,len(h)*13+1).reshape(len(h),13))


def test_leading_zeros_uses_postlaunch_only_and_preserves_bias():
    h=history()
    baseline=np.full((len(h),13),4.0)
    output=v._trim_leading_zeros(h,baseline)
    np.testing.assert_allclose(output.sum(axis=0),baseline.sum(axis=0))
    assert output.shape==baseline.shape
    assert np.isfinite(output).all() and (output>=0).all()
    # All-history zero series have no learned launch date and remain eligible
    # for the mass-conserving global scale only, never a forced positive boost.


def test_historical_selection_origin_boundaries_and_gate(monkeypatch):
    h=history()
    calls=[]
    def fake_control(hist,trees):
        calls.append(len(hist.columns)-3)
        return np.full((len(hist),13),1.)
    def fake_candidates(hist,control,prices,trees,activity_trees):
        return {key:control.copy() for key in v.POLICIES},{}
    monkeypatch.setattr(v,"incumbent",fake_control)
    monkeypatch.setattr(v,"candidates",fake_candidates)
    result=v._phase0(h,None,trees=6,activity_trees=6,baseline_trees=6)
    assert calls==list(v.ORIGINS)
    assert result["origins_train"]==[105,118,131,144]
    assert result["holdout_origin"]==157
    assert result["selected"]=="learned_activity"
    assert result["accepted"] is False
    assert result["minimum_absolute_gain"]==.001


def test_phase1_archive_frozen_before_labels_and_no_overwrite(monkeypatch,tmp_path):
    h=history()
    h.to_csv(tmp_path/"Phase 0 - Sales.csv",index=False)
    monkeypatch.setattr(v,"_phase0",lambda *_args,**kwargs:{
        "selected":"learned_activity","accepted":False})
    monkeypatch.setattr(v,"incumbent",lambda *_args,**kwargs:np.ones((len(h),13)))
    monkeypatch.setattr(v,"candidates",
                        lambda hist,control,prices,trees,activity_trees:(
                            {"learned_activity":control.copy()},{}))
    dest=tmp_path/"receipt"
    with pytest.raises(FileNotFoundError):
        v.phase1(tmp_path,dest,price0=None,trees=6,
                 activity_trees=6,baseline_trees=6)
    # Even if truth is absent, frozen immutable predictions are produced first.
    assert (dest/"frozen-phase1.npz").exists()
    assert (dest/"manifest.json").exists()
    import json,hashlib
    manifest=json.loads((dest/"manifest.json").read_text())
    assert hashlib.sha256((dest/"frozen-phase1.npz").read_bytes()).hexdigest()==manifest["sha256"]
    with pytest.raises(FileExistsError):
        v.phase1(tmp_path,dest,price0=None,trees=6,
                 activity_trees=6,baseline_trees=6)
