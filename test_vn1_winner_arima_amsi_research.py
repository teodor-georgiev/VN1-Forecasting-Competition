"""Proofs for bounded ARIMA+AMSI benchmark: no target leakage or bias drift."""
import numpy as np
import pandas as pd
import pytest

import vn1_winner_arima_amsi_research as exp


def data(n=18, weeks=170, seed=17):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-07-06", periods=weeks, freq="7D").strftime("%Y-%m-%d")
    ids = pd.DataFrame({
        "Client":np.arange(n)//3, "Warehouse":np.arange(n)%3,
        "Product":np.arange(n)+100,
    })
    y = rng.poisson(np.arange(n)[:,None]%4+8, size=(n,weeks)).astype(float)
    y[0] = 0
    return pd.concat([ids,pd.DataFrame(y,columns=dates)],axis=1)


def test_aggregated_arima_disaggregates_and_caps_groups(monkeypatch):
    history = data()
    baseline = np.full((len(history), exp.HORIZON), 3.)
    groups = []
    class ARIMAStub:
        def forecast(self, *, y, h):
            groups.append(y.shape[0])
            return {"mean": np.full(h, y[-4:].mean()*1.3)}
    monkeypatch.setattr(exp,"make_stats_model",lambda name:ARIMAStub())
    result, meta = exp.client_autoarima(history,baseline,cap=2)
    assert meta["tried_groups"]==2
    assert meta["fitted_groups"]==2
    assert meta["affected_series"]==6
    assert groups==[170,170]
    assert np.all(result>=0) and np.isfinite(result).all()
    assert int((result!=baseline).any(axis=1).sum())==6
    assert meta["eligible_groups"]>=2


def test_bad_autoarima_uses_safe_fallback(monkeypatch):
    history=data()
    base=np.ones((len(history),exp.HORIZON))
    class Broken:
        def forecast(self,*,y,h):
            raise ValueError("unfittable")
    monkeypatch.setattr(exp,"make_stats_model",lambda name:Broken())
    result,meta=exp.client_autoarima(history,base,cap=3)
    np.testing.assert_array_equal(result,base)
    assert meta["fitted_groups"]==0
    assert meta["failed_groups"]==3
    with pytest.raises(ValueError):
        exp.client_autoarima(history,base,cap=513)


def test_all_weights_keep_weekly_total_and_input_immutable():
    x = np.arange(1,1+7*exp.HORIZON,dtype=float).reshape(7,exp.HORIZON)
    original=x.copy()
    arima=x*2
    amsi=x[::-1].copy()
    for name,kw in {
        "arima":dict(arima=arima,arima_weight=.3),
        "amsi":dict(amsi=amsi,amsi_weight=.1),
        "both":dict(arima=arima,amsi=amsi,arima_weight=.3,amsi_weight=.1),
    }.items():
        result=exp._mix(x,**kw)
        np.testing.assert_allclose(result.sum(axis=0),x.sum(axis=0),err_msg=name)
        assert np.isfinite(result).all()
    np.testing.assert_array_equal(x,original)
    with pytest.raises(ValueError):
        exp._mix(x,arima=arima,arima_weight=.75,amsi_weight=.40)


def test_predeclared_candidates_eligible_and_phase1_future_untouched(monkeypatch):
    history=data()
    baseline=np.full((len(history),exp.HORIZON),3.)
    def stub_series(hist,name,cap,fallback):
        mask=np.zeros(len(hist),dtype=bool)
        mask[1]=True
        out=fallback.copy()
        out[1]*=2
        return out,mask
    def stub_client(hist, fallback,cap):
        return fallback*.95, {"fitted_groups":2,"affected_series":5}
    monkeypatch.setattr(exp,"statsforecast_specialist",stub_series)
    monkeypatch.setattr(exp,"client_autoarima",stub_client)
    monkeypatch.setattr(exp,"predict_activity",lambda hist,estimators:(
        np.linspace(.1,.9,len(hist)),{"train_origins":[65]}))
    out,info=exp.candidates(history,baseline,cap=2,activity_trees=5)
    assert tuple(out)==exp.CANDIDATES
    assert info["arima_series_fitted"]==1
    for k,matrix in out.items():
        assert matrix.shape==baseline.shape
        assert np.isfinite(matrix).all() and (matrix>=0).all()
        np.testing.assert_allclose(matrix.sum(axis=0),baseline.sum(axis=0),err_msg=k)

    first=history.iloc[:,:3+105]
    a,_=exp.candidates(first,baseline,cap=2,activity_trees=5)
    changed=history.copy()
    changed.iloc[:,3+105:]+=1e6
    b,_=exp.candidates(changed.iloc[:,:3+105],baseline,cap=2,activity_trees=5)
    for name in out:
        np.testing.assert_array_equal(a[name],b[name])


def test_selection_uses_only_complete_phase0_origins(monkeypatch):
    history=data()
    origins=[]
    def control(past,trees):
        origins.append(past.shape[1]-3)
        return np.ones((len(past),exp.HORIZON))
    monkeypatch.setattr(exp,"incumbent",control)
    monkeypatch.setattr(exp,"candidates",
                        lambda hist,control,cap,activity_trees: (
                            {k:control.copy() for k in exp.CANDIDATES},
                            {"arima_series_fitted":0,
                             "arima_client":{"fitted_groups":0,"affected_series":0}}))
    receipt=exp.select_phase0(history,cap=0,trees=3,activity_trees=4)
    assert origins==list(exp.ORIGINS)
    assert receipt["training_origins"]==list(exp.ORIGINS[:-1])
    assert receipt["holdout_origin"]==157
    assert receipt["selected"]=="learned_activity"
    assert receipt["accepted"] is False


def test_empty_arima_cap_keeps_same_baseline(monkeypatch):
    history=data()
    control=np.ones((len(history),exp.HORIZON))
    monkeypatch.setattr(exp,"statsforecast_specialist",
                        lambda hist,name,cap,fallback:(fallback.copy(),np.zeros(len(hist),dtype=bool)))
    monkeypatch.setattr(exp,"client_autoarima",lambda hist,base,cap:(
        base.copy(),{"fitted_groups":0,"affected_series":0}))
    monkeypatch.setattr(exp,"predict_activity",lambda hist,estimators:(
        np.ones(len(hist)),{}))
    c,_=exp.candidates(history,control,cap=0)
    np.testing.assert_array_equal(c["arima30_series"],c["learned_activity"])
    np.testing.assert_array_equal(c["arima30_client"],c["learned_activity"])


def test_freeze_before_truth_then_verify_hash_and_key_alignment(monkeypatch,tmp_path):
    h=data()
    h.to_csv(tmp_path/"Phase 0 - Sales.csv",index=False)
    last13=h.iloc[:,3+157:3+170].copy()
    target=pd.concat([h[exp.KEYS],last13],axis=1)
    target.to_csv(tmp_path/"Phase 1 - Sales.csv",index=False)
    monkeypatch.setattr(exp,"select_phase0",lambda phase0,cap,trees,activity_trees:{
        "selected":"learned_activity","accepted":False})
    monkeypatch.setattr(exp,"incumbent",lambda hist,trees:np.ones((len(hist),exp.HORIZON)))
    monkeypatch.setattr(exp,"candidates",lambda hist,control,cap,activity_trees:(
        {name:control.copy() for name in exp.CANDIDATES},
        {"arima_series_fitted":0,"arima_client":{"fitted_groups":0}}))
    dest=tmp_path/"bundle"
    receipt=exp.freeze(tmp_path,dest,cap=0)
    assert receipt["no_target_phase_labels_loaded_in_freeze"]
    with pytest.raises(ValueError):
        exp.score(tmp_path,dest,phase=1,truth=tmp_path/"Phase 1 - Sales.csv")
    # Above fails because synthetic Phase 1 date columns do NOT continue Phase 0.
    with (dest/"frozen-phase2.npz").open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError,match="modified"):
        exp.score(tmp_path,dest,phase=2,truth=tmp_path/"Phase 1 - Sales.csv")
    with pytest.raises(FileExistsError):
        exp.freeze(tmp_path,dest,cap=0)


def test_no_truth_file_needed_to_freeze_phase2(monkeypatch,tmp_path):
    hist=data()
    hist.to_csv(tmp_path/"Phase 0 - Sales.csv",index=False)
    future_dates=pd.date_range(pd.Timestamp(hist.columns[-1])+pd.Timedelta(weeks=1),
                               periods=exp.HORIZON,freq="7D").strftime("%Y-%m-%d")
    p1=pd.concat([hist[exp.KEYS],
                  pd.DataFrame(np.ones((len(hist),exp.HORIZON)),
                               columns=future_dates)],axis=1)
    p1.to_csv(tmp_path/"Phase 1 - Sales.csv",index=False)
    monkeypatch.setattr(exp,"select_phase0",lambda phase0,cap,trees,activity_trees:{
        "selected":"learned_activity","accepted":False})
    monkeypatch.setattr(exp,"incumbent",lambda hist,trees:
                        np.ones((len(hist),exp.HORIZON)))
    monkeypatch.setattr(exp,"candidates",lambda hist,control,cap,activity_trees:(
        {name:control.copy() for name in exp.CANDIDATES},
        {"arima_series_fitted":0,"arima_client":{"fitted_groups":0}}))
    dest=tmp_path/"frozen"
    receipt=exp.freeze(tmp_path,dest,cap=0)
    assert receipt["phase_outputs"]["2"]["series"]==len(hist)
    assert not (tmp_path/"Phase 2 - Sales.csv").exists()
    p1_score=exp.score(tmp_path,dest,phase=1,truth=tmp_path/"Phase 1 - Sales.csv")
    assert "learned_activity" in p1_score["scores"]
    assert p1_score["retrospective_not_blind"]
