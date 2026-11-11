"""Regression tests: residuals are OOF relative to exact ARIMA+AMSI champion."""
import json

import numpy as np
import pandas as pd
import pytest

import vn1_champion_residual_oof as v
from vn1_winner_arima_amsi_research import candidates as full_champion_registry


def dataset(n=18,t=170,seed=7):
    rng=np.random.default_rng(seed)
    ids=pd.DataFrame({"Client":np.arange(n)//3, "Warehouse":np.arange(n)%3,
                      "Product":np.arange(n)+10})
    dates=pd.date_range("2020-07-06",periods=t,freq="7D").strftime("%Y-%m-%d")
    rates=np.array([0,.4,2.5,4.,8.,13.]*(n//6))
    sales=rng.poisson(rates[:,None],size=(n,t)).astype(float)
    sales[0,:]=0.
    return pd.concat([ids,pd.DataFrame(sales,columns=dates)],axis=1)


def test_exact_champion_fast_path_matches_original_registry():
    h=dataset(t=105)
    # With cap=0, no statistical fit; the formula must be exactly identical
    # to PR17 historical_selected ARIMA-client+AMSI candidate.
    import vn1_champion_residual_oof as v
    from vn1_amsi_zero_rule_ablation import incumbent
    base=incumbent(h,trees=8)
    full,_=full_champion_registry(h,base,cap=0,activity_trees=8)
    fast,_=v.champion_forecast(h,trees=8,activity_trees=8,client_cap=0)
    np.testing.assert_allclose(fast,full["arima30_client_amsi_last10"],rtol=0,atol=1e-10)


def test_oof_origin_boundaries_strictly_prior_to_fit():
    assert v.completed_oof_origins(117)==[]
    assert v.completed_oof_origins(118)==[105]
    assert v.completed_oof_origins(131)==[105,118]
    assert v.completed_oof_origins(144)==[105,118,131]
    assert v.completed_oof_origins(157)==[105,118,131,144]
    with pytest.raises(ValueError):
        v.make_oof_record(dataset(),100)


def test_oof_features_ignore_sales_after_origin(monkeypatch):
    h=dataset()
    base=np.ones((len(h),13))*3.
    seen=[]
    def champ(past,trees,activity_trees,client_cap):
        seen.append(len(past.columns)-3)
        return base.copy(),{"policy":v.CHAMPION_NAME}
    monkeypatch.setattr(v,"champion_forecast",champ)
    a=v.make_oof_record(h,105)
    changed=h.copy()
    changed.iloc[:,3+118:]+=1e5
    b=v.make_oof_record(changed,105)
    assert seen==[105,105]
    pd.testing.assert_frame_equal(a["X"],b["X"])
    np.testing.assert_array_equal(a["residual"],b["residual"])
    assert a["eligible_series"]>0


def test_residual_targets_are_actual_minus_champion_forecast(monkeypatch):
    h=dataset(t=131)
    constant=np.ones((len(h),13))*5.
    monkeypatch.setattr(v,"champion_forecast",lambda past,trees,activity_trees,client_cap:(
        constant.copy(),{}))
    r=v.make_oof_record(h,105)
    truth=h.iloc[:,3+105:3+118].to_numpy(dtype=float)
    mask=r["mask"]
    expected=(truth[mask]-5.).reshape(-1)
    np.testing.assert_array_equal(r["residual"],expected)
    assert all(x.startswith("champion_") for x in (
        "champion_prediction","champion_log1p","champion_minus_stat"))
    assert np.allclose(r["full_features"]["champion_prediction"].to_numpy(),5.)


def test_regimes_and_active_predicate_only_use_last_observed_weeks():
    h=dataset()
    y=v._history_matrix(h)
    ychanged=y.copy()
    ychanged[:,131:]+=10000
    np.testing.assert_array_equal(
        v._regimes(y[:,:131]),v._regimes(ychanged[:,:131]))
    np.testing.assert_array_equal(
        v._active_mask(y[:,:131]),v._active_mask(ychanged[:,:131]))
    assert set(np.unique(v._regimes(y))) <= {0,1,2}


def test_forward_only_gate_calibrates_on_last_complete_origin(monkeypatch):
    history=dataset(t=144)
    n=len(history)
    predictions=np.full((n,13),10.)
    ids=np.ones(n,dtype=bool)
    r=[]
    for t in (105,118,131):
        f=v._regression_features(history.iloc[:,:3+t],predictions)
        r.append({"origin":t,"X":f.loc[np.repeat(ids,13)],
                  "residual":np.ones(n*13),
                  "regime":np.zeros(n,dtype=np.int8),"mask":ids,
                  "champion":predictions,"truth":predictions,
                  "full_features":f})
    order=[]
    class Dummy:
        def predict(self,x):
            return np.zeros(len(x))
    def fit(records,trees):
        order.append([a["origin"] for a in records])
        return Dummy()
    monkeypatch.setattr(v,"_fit",fit)
    gate,meta=v._gate_weights(r)
    assert order==[[105,118]]
    assert meta["calibration_origin"]==131
    assert gate.sum()==0
    assert v._gate_weights(r[:1])[0].sum()==0


def test_candidate_generator_uses_only_completed_oof_and_mass_conservation(monkeypatch):
    h=dataset(t=131); n=len(h)
    champ=np.full((n,13),6.)
    dummy={"origin":105,"X":pd.DataFrame({"Client":[1],"Warehouse":[1]}),
           "residual":np.array([1.])}
    class Stub:
        pass
    monkeypatch.setattr(v,"_fit",lambda records,trees:Stub())
    monkeypatch.setattr(v,"_corrections",lambda model,history,champion:(
        np.ones_like(champion)))
    monkeypatch.setattr(v,"_gate_weights",lambda records,trees:(
        np.array([.2,.35,0.]),{}))
    pred,meta=v.candidate_forecasts(h,oof_records=[dummy],champion=champ)
    assert tuple(pred)==v.PREDICTION_NAMES
    assert meta["completed_oof_origins"]==[105]
    for output in pred.values():
        np.testing.assert_allclose(output.sum(axis=0),champ.sum(axis=0))
    with pytest.raises(ValueError,match="future labels"):
        v.candidate_forecasts(h.iloc[:,:3+117],oof_records=[dummy],champion=champ)
    with pytest.raises(ValueError,match="Duplicate"):
        v.candidate_forecasts(h,oof_records=[dummy,dummy],champion=champ)


def test_rolling_validation_holdout_rejects_even_when_training_wins(monkeypatch):
    h=dataset()
    called=[]
    def fake_records(hist,t,cache,trees,activity_trees,client_cap):
        return [{"origin":105}]
    def fake_champ(hist,trees,activity_trees,client_cap):
        called.append(len(hist.columns)-3)
        return np.ones((len(hist),13)),{}
    def fake_candidate(hist,oof_records,champion,residual_trees):
        out={"champion":champion.copy()}
        out.update({name:champion*.9 for name in v.PREDICTION_NAMES[1:]})
        return out,{}
    monkeypatch.setattr(v,"_completed_records",fake_records)
    monkeypatch.setattr(v,"champion_forecast",fake_champ)
    monkeypatch.setattr(v,"candidate_forecasts",fake_candidate)
    receipt=v.historical_validation(h)
    assert called==[118,131,144,157]
    assert receipt["training_origins"]==[118,131,144]
    assert receipt["holdout_origin"]==157
    assert receipt["selected"] in v.PREDICTION_NAMES
    assert receipt["retrospective_not_blind"]


def test_freeze_scores_only_after_future_truth_and_sha_validation(monkeypatch,tmp_path):
    h=dataset()
    h.to_csv(tmp_path/"Phase 0 - Sales.csv",index=False)
    dates1=pd.date_range(pd.Timestamp(h.columns[-1])+pd.Timedelta(weeks=1),
                         periods=13,freq="7D").strftime("%Y-%m-%d")
    p1=pd.concat([h[v.KEYS],pd.DataFrame(np.ones((len(h),13)),columns=dates1)],axis=1)
    p1.to_csv(tmp_path/"Phase 1 - Sales.csv",index=False)
    dates2=pd.date_range(pd.Timestamp(dates1[-1])+pd.Timedelta(weeks=1),
                         periods=13,freq="7D").strftime("%Y-%m-%d")
    p2=pd.concat([h[v.KEYS],pd.DataFrame(np.ones((len(h),13)),columns=dates2)],axis=1)
    p2path=tmp_path/"Phase 2 - Sales.csv"
    p2.to_csv(p2path,index=False)
    chosen={
        "training_origins":[118,131,144],"holdout_origin":157,
        "candidate_names":list(v.PREDICTION_NAMES),
        "training_scores":{"champion":.50,"residual_uniform20":.48},
        "holdout_scores":{"champion":.50,"residual_uniform20":.48},
        "folds_better_than_champion":{"residual_uniform20":3},
        "winner_training_only":"residual_uniform20",
        "selected":"residual_uniform20","accepted":True}
    report=tmp_path/"selection.json"
    report.write_text(json.dumps(chosen))
    monkeypatch.setattr(v,"_completed_records",lambda *args,**kwargs:[{"origin":105}])
    monkeypatch.setattr(v,"champion_forecast",lambda h,trees,activity_trees,client_cap:(
        np.ones((len(h),13)),{}))
    monkeypatch.setattr(v,"candidate_forecasts",lambda h,oof_records,champion,residual_trees:(
        {name:champion.copy() for name in v.PREDICTION_NAMES},{}))
    dest=tmp_path/"frozen"
    p2path.rename(tmp_path/"hidden-truth.csv")
    manifest=v.freeze(tmp_path,dest,report,phase=2)
    assert manifest["no_target_labels_loaded_during_freeze"]
    assert manifest["selected"]=="residual_uniform20"
    with pytest.raises(FileNotFoundError):
        v.score(tmp_path,dest,p2path)
    (tmp_path/"hidden-truth.csv").rename(p2path)
    result=v.score(tmp_path,dest,p2path)
    assert result["scores"]["historical_selected"]==0.
    with (dest/"frozen.npz").open("ab") as file:
        file.write(b"corrupt")
    with pytest.raises(ValueError,match="SHA-256"):
        v.score(tmp_path,dest,p2path)
    with pytest.raises(FileExistsError):
        v.freeze(tmp_path,dest,report,phase=2)
