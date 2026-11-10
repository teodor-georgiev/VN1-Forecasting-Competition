"""Chronological learned-activity model tests: no future labels and conserved VN1 mass."""
import numpy as np
import pandas as pd
import pytest

import vn1_learned_activity as model


def make_history(n=18, weeks=170, seed=23):
    assert n % 3 == 0
    ids=pd.DataFrame({
        "Client":np.repeat(np.arange(n//3),3),
        "Warehouse":np.tile([1,2,3],n//3),
        "Product":100+np.arange(n),
    })
    dates=pd.date_range("2020-07-06",periods=weeks,freq="7D").strftime("%Y-%m-%d")
    rng=np.random.default_rng(seed)
    rates=np.array([0,0.01,0.1,0.3,1,3] * (n//6),dtype=float)
    y=rng.poisson(rates[:,None],size=(n,weeks)).astype(float)
    y[0]=0
    y[2,-26:]=0
    return pd.concat([ids,pd.DataFrame(y,columns=dates)],axis=1)


def test_training_origins_never_leak_target_horizon():
    assert model.training_origins(105)==[65,78,91]
    assert model.training_origins(170)==[65,78,91,104,117,130,143,156]
    for cutoff in model.ORIGINS:
        assert all(origin+model.HORIZON<=cutoff
                   for origin in model.training_origins(cutoff))


def test_predictor_features_use_only_visible_cutoff_data():
    history=make_history()
    cutoff=105
    early=history.iloc[:,:3+cutoff].copy()
    old=model.activity_features(early)
    mutated=history.copy()
    mutated.iloc[:,3+cutoff:]+=10_000
    updated=model.activity_features(mutated.iloc[:,:3+cutoff])
    pd.testing.assert_frame_equal(old,updated)
    assert np.isfinite(old.to_numpy()).all()
    assert "annual_matched_sum" in old.columns
    assert old.loc[0,"weeks_since_event"]==52


def test_learned_training_is_deterministic_and_valid_probs():
    h=make_history(n=18)
    p1,info=model.predict_activity(h.iloc[:,:3+105],estimators=8)
    p2,info2=model.predict_activity(h.iloc[:,:3+105],estimators=8)
    np.testing.assert_allclose(p1,p2,atol=1e-12)
    assert info["train_origins"]==[65,78,91]
    assert (p1>=0.001).all() and (p1<=0.999).all()
    assert 0<=info["train_event_rate"]<=1
    assert info==info2


def test_forecast_mass_conserved_per_week_and_zero_weight_guard():
    b=np.arange(1,1+5*model.HORIZON,dtype=float).reshape(5,model.HORIZON)
    weights=np.array([0.01,0.1,0.2,0.5,1.])
    adjusted=model.allocate_weekly(b,weights)
    np.testing.assert_allclose(adjusted.sum(axis=0),b.sum(axis=0))
    np.testing.assert_array_equal(model.allocate_weekly(b,np.zeros(5)),b)
    with pytest.raises(ValueError):
        model.allocate_weekly(b,np.ones(4))
    with pytest.raises(ValueError):
        model.allocate_weekly(b,np.full(5,-1.))


def test_candidates_valid_and_unchanged_after_future_sales_mutation(monkeypatch):
    h=make_history()
    base=np.full((len(h),model.HORIZON),1.5)
    monkeypatch.setattr(model,"predict_activity",
                        lambda hist,estimators:(np.linspace(.01,.99,len(hist)),{"p50":.5}))
    a,info=model.learned_candidates(h,base,estimators=5)
    assert tuple(a)==model.CANDIDATE_NAMES
    assert info["p50"]==.5
    assert np.all(base==1.5)
    for name,pred in a.items():
        assert pred.shape==base.shape and np.isfinite(pred).all()
        np.testing.assert_allclose(pred.sum(axis=0),base.sum(axis=0),
                                   atol=1e-6,err_msg=name)


def test_hist_selection_never_uses_targets_beyond_each_origin(monkeypatch):
    h=make_history()
    orig=[]
    def fake_base(past,trees):
        orig.append(past.shape[1]-3)
        return np.full((len(past),model.HORIZON),1.2)
    def fake_models(past,base,estimators):
        return {name:base.copy() for name in model.CANDIDATE_NAMES},{"train_origins":[]}
    monkeypatch.setattr(model,"incumbent",fake_base)
    monkeypatch.setattr(model,"learned_candidates",fake_models)
    selected=model.select_on_phase0(h,trees=4,activity_trees=8)
    assert orig==list(model.ORIGINS)
    assert selected["origins_train"]==list(model.ORIGINS[:-1])
    assert selected["origin_holdout"]==157
    assert selected["selected"]=="incumbent"
    assert not selected["accepted"]
    assert selected["minimum_improvement"]==.001


def test_frozen_artifact_rejects_tampering_and_unavailable_phase_truth(monkeypatch,tmp_path):
    h=make_history()
    h.to_csv(tmp_path/"Phase 0 - Sales.csv",index=False)
    monkeypatch.setattr(model,"select_on_phase0",
                        lambda phase0,trees,activity_trees:{
                            "selected":"incumbent","accepted":False})
    monkeypatch.setattr(model,"incumbent",
                        lambda h,trees:np.ones((len(h),model.HORIZON)))
    monkeypatch.setattr(model,"learned_candidates",
                        lambda h,base,estimators:(
                            {name:base.copy() for name in model.CANDIDATE_NAMES}, {}))
    out=tmp_path/"result"
    receipt=model.freeze(tmp_path,out,phase=1,trees=4,activity_trees=5)
    assert receipt["truth_read_after_freeze_only"]
    assert receipt["candidate_names"][-1]=="phase0_selected"
    with pytest.raises(FileNotFoundError):
        model.score(tmp_path,out)
    with (out/"frozen.npz").open("ab") as fd:
        fd.write(b"tampered!")
    with pytest.raises(ValueError,match="SHA256"):
        model.score(tmp_path,out)
    with pytest.raises(FileExistsError):
        model.freeze(tmp_path,out,phase=1,trees=4,activity_trees=5)
