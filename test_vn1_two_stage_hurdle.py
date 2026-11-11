"""Two-stage hurdle: conditional-only labels, chronological folds and frozen receipts."""
import numpy as np
import pandas as pd
import pytest

import vn1_two_stage_hurdle as h


def sample(n=18,weeks=170,seed=19):
    rng=np.random.default_rng(seed)
    dates=pd.date_range("2020-07-06",periods=weeks,freq="7D").strftime("%Y-%m-%d")
    ids=pd.DataFrame({"Client":np.arange(n)//3,
                      "Warehouse":np.arange(n)%3,
                      "Product":np.arange(n)+100})
    rates=np.array([0, .01, .1, .5, 1, 3]*(n//6),dtype=float)
    y=rng.poisson(rates[:,None],size=(n,weeks)).astype(float)
    y[0]=0
    y[1,-26:]=0
    return pd.concat([ids,pd.DataFrame(y,columns=dates)],axis=1)


def test_training_targets_are_strictly_positive_and_fully_observed(monkeypatch):
    hist=sample()
    seen={}
    class StubRegressor:
        def __init__(self,**kwargs):
            pass
        def fit(self,X,targets):
            seen["rows"]=len(X)
            seen["target"]=np.array(targets)
            seen["columns"]=list(X.columns)
            return self
    monkeypatch.setattr(h.lgb,"LGBMRegressor",StubRegressor)
    model,prior,details=h.fit_conditional(hist.iloc[:,:3+105],n_estimators=9)
    assert model is not None
    assert details["origins"]==[65,78,91]
    assert seen["rows"]==sum(details["positive_counts"])
    assert np.all(seen["target"]>=np.log(2))
    assert prior>0
    assert "weeks_since_event" in seen["columns"]


def test_conditional_ignores_future_targets():
    hist=sample()
    early=hist.iloc[:,:3+105].copy()
    first,first_meta=h.predict_positive_total(early,n_estimators=8)
    future=hist.copy()
    future.iloc[:,3+105:]+=10000
    second,second_meta=h.predict_positive_total(
        future.iloc[:,:3+105],n_estimators=8)
    np.testing.assert_array_equal(first,second)
    assert first_meta==second_meta
    assert (first>=0).all()


def test_expected_total_factorization_and_weekly_shape():
    base=np.array([[2.,4.]+[0.]*11,[0.]*13])
    p=np.array([0.5,0.25])
    conditional=np.array([20.,8.])
    outcome=h.hurdle_matrix(base,p,conditional)
    np.testing.assert_allclose(outcome.sum(axis=1),p*conditional)
    assert outcome[0,1]==pytest.approx(2*outcome[0,0])
    assert outcome[1,0]==pytest.approx(2./13)
    assert np.isfinite(outcome).all()


def test_weekly_volume_conservation_after_hurdle_blend():
    b=np.array([[4.]*13,[1.]*13,[2.]*13])
    alternative=np.array([[0.]*13,[1.]*13,[3.]*13])
    for w in (0,0.25,0.5,1):
        pred=h.normalize_weekly((1-w)*b+w*alternative,b)
        np.testing.assert_allclose(pred.sum(axis=0),b.sum(axis=0))
        assert (pred>=0).all()
    with pytest.raises(ValueError):
        h.normalize_weekly(np.full_like(b,-1),b)
    np.testing.assert_array_equal(h.normalize_weekly(np.zeros_like(b),b),b)


def test_positive_only_fallback_when_no_completed_events():
    hist=sample(n=18,weeks=105)
    hist.iloc[:,3:]=0
    model,prior,details=h.fit_conditional(hist,n_estimators=5)
    assert model is None and prior==0
    assert sum(details["positive_counts"])==0
    quantity,info=h.predict_positive_total(hist,n_estimators=5)
    np.testing.assert_array_equal(quantity,np.zeros(len(hist)))


def test_candidates_are_deterministic_and_tunable(monkeypatch):
    hist=sample()
    n=len(hist)
    base=np.full((n,13),2.0)
    monkeypatch.setattr(h,"predict_activity",lambda past,estimators:(
        np.linspace(.05,.95,len(past)),{"train_origins":[65]}))
    monkeypatch.setattr(h,"predict_positive_total",lambda past,n_estimators:(
        np.linspace(2,100,len(past)),{"origins":[65]}))
    x,details=h.candidates(hist,base,occurrence_trees=3,conditional_trees=4)
    assert tuple(x)==h.POLICIES
    assert details["conditional"]["origins"]==[65]
    for p in x.values():
        np.testing.assert_allclose(p.sum(axis=0),base.sum(axis=0))
    assert not np.array_equal(x["activity_p100"],x["hurdle100"])


def test_historical_selection_never_opens_future_labels(monkeypatch):
    hist=sample()
    calls=[]
    def control(past,trees):
        calls.append(past.shape[1]-3)
        return np.full((len(past),h.HORIZON),2.)
    def candidate(past,base,occurrence_trees,conditional_trees):
        return {k:base.copy() for k in h.POLICIES},{}
    monkeypatch.setattr(h,"incumbent",control)
    monkeypatch.setattr(h,"candidates",candidate)
    selection=h.select_phase0(hist,trees=3,occurrence_trees=4,conditional_trees=5)
    assert calls==list(h.ORIGINS)
    assert selection["training_origins"]==list(h.ORIGINS[:-1])
    assert selection["holdout_origin"]==157
    assert selection["selected"]=="activity_p100"
    assert not selection["accepted"]


def test_frozen_predictions_refuse_overwrite_or_corrupt_scores(monkeypatch,tmp_path):
    hist=sample()
    hist.to_csv(tmp_path/"Phase 0 - Sales.csv",index=False)
    monkeypatch.setattr(h,"select_phase0",lambda *_args,**kwargs:{
        "selected":"activity_p100","accepted":False})
    monkeypatch.setattr(h,"incumbent",lambda history,trees:
        np.ones((len(history),h.HORIZON)))
    monkeypatch.setattr(h,"candidates",lambda history,base,**kwargs:(
        {k:base.copy() for k in h.POLICIES},{}))
    dst=tmp_path/"freeze"
    receipt=h.freeze(tmp_path,dst,phase=1)
    assert receipt["fit_only_historical_observed_targets"]
    assert "phase0_selected" in receipt["forecast_names"]
    with pytest.raises(FileNotFoundError):
        h.score(tmp_path,dst)
    with (dst/"frozen.npz").open("ab") as file:
        file.write(b"tamper")
    with pytest.raises(ValueError,match="hash"):
        h.score(tmp_path,dst)
    with pytest.raises(FileExistsError):
        h.freeze(tmp_path,dst,phase=1)
