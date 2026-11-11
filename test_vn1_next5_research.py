"""Leakage and metric regression tests for VN1 active residual/adaptive research."""
import json
import numpy as np
import pandas as pd
import pytest

import vn1_next5_research as v


def data(n=18, t=170, seed=16):
    rng=np.random.default_rng(seed)
    ids=pd.DataFrame({"Client":np.arange(n)//3, "Warehouse":np.arange(n)%3,
                      "Product":np.arange(n)+100})
    dates=pd.date_range("2020-07-06",periods=t,freq="7D").strftime("%Y-%m-%d")
    rates=np.array([.1,.5,2.,4.,8.,12.]*(n//6))
    sales=rng.poisson(rates[:,None],size=(n,t)).astype(float)
    sales[0]=0
    return pd.concat([ids,pd.DataFrame(sales,columns=dates)],axis=1)


def test_active_mask_only_uses_past():
    h=data()
    y=v.values(h)
    m=v.active_mask(y[:,:105])
    changed=y.copy()
    changed[:,105:]+=10000
    np.testing.assert_array_equal(m,v.active_mask(changed[:,:105]))
    assert m.sum()>0 and m.sum()<len(h)


def test_error_audit_aggregates_no_raw_ids_and_preserves_score():
    h=data()
    actual=v.values(h)[:,-13:]
    guess=np.full_like(actual,3.)
    report=v.audit_errors(h,guess,actual)
    assert report["series"]==18
    assert len(report["week_absolute_error_units"])==13
    assert report["private_ids_emitted"] is False
    assert 0<=report["top1pct_series_error_share"]<=1
    assert report["top5pct_series_error_share"]>=report["top1pct_series_error_share"]
    assert report["official_score"]==pytest.approx(v._metric(guess,actual))
    assert "Client" not in json.dumps(report) and "Product" not in json.dumps(report)
    with pytest.raises(ValueError):
        v.audit_errors(h,guess,actual[:,:12])


def test_adaptive_weights_are_preorigin_and_mass_conserving():
    h=data()
    base=np.ones((len(h),13))*3
    future=h.copy()
    future.iloc[:,3+131:]+=9999
    x,m=v.adaptive_amsi(h.iloc[:,:3+131],base)
    z,_=v.adaptive_amsi(future.iloc[:,:3+131],base)
    for name,output in x.items():
        np.testing.assert_array_equal(output,z[name])
        np.testing.assert_allclose(output.sum(axis=0),base.sum(axis=0))
        assert np.isfinite(output).all() and (output>=0).all()
    assert m["mean_series_weight"]>=0


def test_residual_train_only_fully_observed_origins(monkeypatch):
    h=data()
    seen=[]
    class Regressor:
        def __init__(self,**kwargs): pass
        def fit(self,X,y,categorical_feature):
            seen.append((len(X),len(y),categorical_feature))
            return self
        def predict(self,X):return np.zeros(len(X))
    monkeypatch.setattr(v.lgb,"LGBMRegressor",Regressor)
    corr,meta=v.fit_active_residual(h.iloc[:,:3+105],trees=4)
    assert meta["training_origins"]==[65,78,91]
    assert all(x+13<=105 for x in meta["training_origins"])
    assert len(seen)==1
    assert seen[0][0]==seen[0][1]
    assert seen[0][2]==["Client","Warehouse"]
    assert np.isfinite(corr).all()
    assert not np.any(corr[~v.active_mask(v.values(h.iloc[:,:3+105]))])


def test_residual_variant_correct_shape_and_conserves_mass(monkeypatch):
    h=data()
    base=np.ones((len(h),13))*5
    monkeypatch.setattr(v,"fit_active_residual",lambda hist,trees:(
        np.ones((len(hist),13)),{"trained":True}))
    pred,meta=v.active_specialists(h,base,trees=5)
    assert set(pred)=={"residual_active20","residual_active35"}
    for output in pred.values():
        np.testing.assert_allclose(output.sum(axis=0),base.sum(axis=0))
        assert np.isfinite(output).all() and (output>=0).all()
    assert meta["trained"]


def test_rolling_gate_never_uses_last_origin_to_pick_winner(monkeypatch):
    h=data()
    seen=[]
    def baseline(past,trees):
        seen.append(len(past.columns)-3)
        return np.ones((len(past),13))
    def candidates(past,control,active_trees,activity_trees):
        return {key:control.copy() for key in v.NAMES},{"specialist":{},"adaptive":{}}
    monkeypatch.setattr(v,"incumbent",baseline)
    monkeypatch.setattr(v,"candidates",candidates)
    result=v.rolling_validate(h,baseline_trees=2,active_trees=3,activity_trees=4)
    assert seen==[92,105,118,131,144,157]
    assert result["training_origins"]==[92,105,118,131,144]
    assert result["holdout_origin"]==157
    assert result["selected"]=="learned_activity"
    assert result["accepted"] is False
    assert result["required_training_folds_improved"]==3


def test_normalize_rejects_invalid_and_conserves_total():
    base=np.ones((5,13))
    proposed=np.arange(65,dtype=float).reshape(5,13)+.1
    z=v.normalize(proposed,base)
    np.testing.assert_allclose(z.sum(axis=0),base.sum(axis=0))
    with pytest.raises(ValueError):
        v.normalize(-proposed,base)
    with pytest.raises(ValueError):
        v.normalize(proposed[:,:12],base)


def test_residual_and_adaptive_training_can_run_on_short_history(monkeypatch):
    h=data(t=105)
    base=np.ones((len(h),13))*2
    monkeypatch.setattr(v,"predict_activity",lambda hist,estimators:(
        np.linspace(.05,.99,len(hist)),{"train_origins":[65]}))
    preds,info=v.candidates(h,base,active_trees=5,activity_trees=5)
    assert tuple(preds)==v.NAMES
    assert info["specialist"]["train_rows"]>0
    for prediction in preds.values():
        np.testing.assert_allclose(prediction.sum(axis=0),base.sum(axis=0))
        assert (prediction>=0).all()


def test_freeze_selection_checks_before_reading_future(monkeypatch,tmp_path):
    report=tmp_path/"historical.json"
    report.write_text(json.dumps({
        "training_origins":[92,105,118,131,144],"holdout_origin":157,
        "candidate_registry":list(v.NAMES),"selected":"learned_activity",
        "accepted":False}))
    observed=data()
    observed.to_csv(tmp_path/"Phase 0 - Sales.csv",index=False)
    dates=pd.date_range(pd.Timestamp(observed.columns[-1])+pd.Timedelta(weeks=1),
                        periods=13,freq="7D").strftime("%Y-%m-%d")
    phase1=pd.concat([observed[v.KEYS],
                      pd.DataFrame(np.ones((len(observed),13)),columns=dates)],axis=1)
    phase1.to_csv(tmp_path/"Phase 1 - Sales.csv",index=False)
    monkeypatch.setattr(v,"incumbent",lambda hist,trees:np.ones((len(hist),13)))
    monkeypatch.setattr(v,"candidates",lambda hist,control,active_trees,activity_trees:(
        {key:control.copy() for key in v.NAMES},{}))
    destination=tmp_path/"frozen"
    receipt=v.freeze(tmp_path,destination,report,phase=2)
    assert receipt["truth_not_read_on_freeze"]
    assert not (tmp_path/"Phase 2 - Sales.csv").exists()
    with pytest.raises(FileNotFoundError):
        v.score(tmp_path,destination,tmp_path/"Phase 2 - Sales.csv")
    with (destination/"frozen.npz").open("ab") as file:
        file.write(b"tamper")
    with pytest.raises(ValueError,match="SHA256"):
        v.score(tmp_path,destination,tmp_path/"Phase 1 - Sales.csv")
