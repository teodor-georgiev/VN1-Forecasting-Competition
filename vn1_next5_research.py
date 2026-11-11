"""VN1 next gains: aggregate error audit, active residuals, adaptive AMSI, rolling CV.

Research-only: never optimize on Phase 1/2 labels. The historical model ranking
uses Phase 0 completed origins 92,105,118,131,144; origin 157 is a separate
acceptance holdout. The independent 0.4693382739 ARIMA+AMSI research incumbent
is not silently replaced, even if a challenger passes this historical gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from vn1_from_scratch import HORIZON, KEYS, read_sales, forecast as statistical_forecast
from vn1_lgbm_ensemble import _features, LGBM_PARAMS, START_TRAIN_WEEK, TRAIN_STRIDE, read_history
from vn1_amsi_zero_rule_ablation import incumbent, _metric
from vn1_learned_activity import allocate_weekly, predict_activity
from vn1_amsi_specialists import amsi_forecast

TRAIN_ORIGINS = (92, 105, 118, 131, 144)
HOLDOUT_ORIGIN = 157
MIN_GAIN = 0.001
NAMES = ("learned_activity", "residual_active20", "residual_active35",
         "adaptive_amsi", "adaptive_group_amsi")
FIT_PARAMS = dict(objective="regression_l1", n_estimators=80,
                  learning_rate=.05, num_leaves=15, max_bin=127,
                  min_child_samples=80, reg_lambda=4.0,
                  colsample_bytree=.9, subsample=.9, subsample_freq=1,
                  random_state=2026, n_jobs=2, verbosity=-1)


def values(history: pd.DataFrame) -> np.ndarray:
    if list(history.columns[:3]) != list(KEYS) or history.duplicated(KEYS).any():
        raise ValueError("Series identifiers missing/duplicated")
    out=history.iloc[:,3:].to_numpy(dtype=np.float64)
    if out.ndim!=2 or out.shape[1]<65 or not np.isfinite(out).all() or (out<0).any():
        raise ValueError("Sales must be finite, nonnegative and >=65 weeks")
    return out


def active_mask(y:np.ndarray) -> np.ndarray:
    """Pre-origin active, higher-volume rows only."""
    recent=y[:,-13:]
    return (recent.mean(axis=1)>=2.0) & ((recent>0).mean(axis=1)>=.5)


def normalize(candidate:np.ndarray,control:np.ndarray) -> np.ndarray:
    b=np.asarray(control,dtype=np.float64)
    c=np.asarray(candidate,dtype=np.float64)
    if c.shape!=b.shape or c.ndim!=2 or c.shape[1]!=HORIZON:
        raise ValueError("Different forecast shapes")
    if not np.isfinite(c).all() or not np.isfinite(b).all() or (c<0).any() or (b<0).any():
        raise ValueError("Invalid forecasts")
    target=b.sum(axis=0)
    total=c.sum(axis=0)
    if np.any((total<=0)&(target>0)):return b.copy()
    result=c*np.divide(target,total,out=np.ones_like(total),where=total>0)[None,:]
    if not np.allclose(result.sum(axis=0),target,rtol=1e-10,atol=1e-5):
        raise ArithmeticError("Weekly total changed")
    return result


def fit_active_residual(history:pd.DataFrame, *, trees:int=80
                        )->tuple[np.ndarray,dict]:
    """Predict positive/negative residuals relative to historical statistical mean.

    All labels and training eligibility are formed at completed, past origins.
    At forecast time only the same pre-cutoff features and fixed eligibility
    threshold apply; predictions are bounded without looking at future truth.
    """
    y=values(history)
    n,t=y.shape
    origin_list=list(range(START_TRAIN_WEEK,t-HORIZON+1,TRAIN_STRIDE))
    features=[]; labels=[]; counts=[]
    for origin in origin_list:
        past=history.iloc[:,:len(KEYS)+origin]
        mask=active_mask(y[:,:origin])
        counts.append(int(mask.sum()))
        if not mask.any():continue
        f=_features(past,origin)
        chosen=np.repeat(mask,HORIZON)
        features.append(f.loc[chosen])
        actual=y[:,origin:origin+HORIZON].reshape(-1)[chosen]
        base=f.loc[chosen,"statistical_baseline"].to_numpy(dtype=float)
        labels.append((actual-base).astype(np.float32))
    cur=_features(history,t)
    mask=active_mask(y)
    if not features or sum(counts)<10:
        return np.zeros((n,HORIZON),dtype=np.float64),{
            "trained":False,"training_origins":origin_list,
            "training_active_series":counts,"predicted_active_series":int(mask.sum())}
    X=pd.concat(features,ignore_index=True)
    labels=np.concatenate(labels)
    model=lgb.LGBMRegressor(**dict(FIT_PARAMS,n_estimators=trees))
    model.fit(X,labels,categorical_feature=["Client","Warehouse"])
    correction=model.predict(cur).reshape(n,HORIZON)
    # Limit residual corrections relative to the observable series magnitude.
    cap=np.maximum(5.,y[:,-52:].max(axis=1)*2.0)
    correction=np.clip(correction,-cap[:,None],cap[:,None])
    correction[~mask,:]=0.
    return correction,{
        "trained":True,"training_origins":origin_list,
        "training_active_series":counts,"predicted_active_series":int(mask.sum()),
        "train_rows":int(len(labels)),
    }


def active_specialists(history:pd.DataFrame,base:np.ndarray,
                       *, trees:int=80)->tuple[dict[str,np.ndarray],dict]:
    correction,meta=fit_active_residual(history,trees=trees)
    mask=active_mask(values(history))
    # Current base is an ML/statistical learned-activity ensemble.
    # Residuals use statistical targets as a separate specialized signal.
    cur_stat=statistical_forecast(history).iloc[:,3:].to_numpy(dtype=float)
    candidate=np.maximum(0,cur_stat+correction)
    blend={}
    for name,weight in (("residual_active20",.20),("residual_active35",.35)):
        mixed=np.array(base,copy=True,dtype=np.float64)
        mixed[mask,:]=(1-weight)*base[mask,:]+weight*candidate[mask,:]
        blend[name]=normalize(mixed,base)
    return blend,meta


def adaptive_amsi(history:pd.DataFrame,base:np.ndarray
                  )->tuple[dict[str,np.ndarray],dict]:
    """History-only AMSI weighting from the last completed 13-week proxy.

    Compare past statistical and AMSI forecasts at origin t-13 against the
    now-observed last 13 weeks. No model is selected with future labels.
    This is a *proxy*, not a historical fitted learned-activity prediction.
    """
    y=values(history); n,t=y.shape
    if t<92:raise ValueError("Require 92 weeks for causal rolling proxy")
    prior=history.iloc[:,:3+t-13]
    old_stat=statistical_forecast(prior).iloc[:,3:].to_numpy(dtype=float)
    old_amsi=amsi_forecast(prior,anchor="last",seasonality=.45,group_shrink=.55)
    observed=y[:,-13:]
    staterr=np.abs(old_stat-observed).sum(axis=1)
    amsierr=np.abs(old_amsi-observed).sum(axis=1)
    improve=np.clip((staterr-amsierr)/(staterr+1),0,1)
    # Only a modest 0-25% weight can be transferred from this noisy proxy.
    weight=.25*improve
    groups,_=pd.factorize(pd.MultiIndex.from_frame(history[["Client","Warehouse"]]),sort=True)
    group_count=np.bincount(groups)
    group_stat=np.bincount(groups,weights=staterr)
    group_amsi=np.bincount(groups,weights=amsierr)
    group_improve=np.clip((group_stat-group_amsi)/(group_stat+group_count),0,1)
    shrink=group_count/(group_count+20.)
    group_weight=.25*(shrink[groups]*group_improve[groups]+
                       (1-shrink[groups])*improve)
    current_amsi=amsi_forecast(history,anchor="last",seasonality=.45,group_shrink=.55)
    out={}
    for key,w in (("adaptive_amsi",weight),("adaptive_group_amsi",group_weight)):
        pred=(1-w[:,None])*base+w[:,None]*current_amsi
        out[key]=normalize(pred,base)
    return out,{"mean_series_weight":float(weight.mean()),
                "mean_group_weight":float(group_weight.mean()),
                "weighted_rows":int((weight>0).sum())}


def candidates(history:pd.DataFrame,control:np.ndarray,*,
               active_trees:int=80,activity_trees:int=100)->tuple[dict,dict]:
    y=values(history)
    if control.shape!=(len(y),HORIZON):raise ValueError("Bad incumbent shape")
    p,activity_meta=predict_activity(history,estimators=activity_trees)
    base=allocate_weekly(control,p)
    specialists,specialist_meta=active_specialists(history,base,trees=active_trees)
    adaptive,adaptive_meta=adaptive_amsi(history,base)
    out={"learned_activity":base,**specialists,**adaptive}
    if tuple(out)!=NAMES:raise AssertionError("Unexpected candidate registry")
    for name,forecast in out.items():
        if forecast.shape!=base.shape or not np.isfinite(forecast).all() or (forecast<0).any():
            raise ValueError("Bad candidate "+name)
        if not np.allclose(forecast.sum(axis=0),base.sum(axis=0),atol=1e-4,rtol=1e-10):
            raise ArithmeticError("Changed aggregate demand")
    return out,{"activity":activity_meta,"specialist":specialist_meta,
                "adaptive":adaptive_meta}


def audit_errors(history:pd.DataFrame,forecast:np.ndarray,truth:np.ndarray)->dict:
    """Aggregate only: no customer/warehouse/product IDs or individual rows."""
    y=values(history)
    pred=np.asarray(forecast,dtype=np.float64)
    actual=np.asarray(truth,dtype=np.float64)
    if pred.shape!=actual.shape or pred.shape!=(len(y),HORIZON):
        raise ValueError("Wrong audit data shapes")
    if not np.isfinite(actual).all() or not np.isfinite(pred).all() or (pred<0).any() or (actual<0).any():
        raise ValueError("Invalid audit values")
    absrow=np.abs(pred-actual).sum(axis=1)
    signedrow=(pred-actual).sum(axis=1)
    byweek=np.abs(pred-actual).sum(axis=0)
    total=float(absrow.sum())
    count=max(1,int(np.ceil(len(y)*.01)))
    k5=max(1,int(np.ceil(len(y)*.05)))
    ordered=np.sort(absrow)[::-1]
    recent=y[:,-13:]
    prev=y[:,-26:-13]
    annual=y[:,-52:]
    mean13=recent.mean(axis=1)
    thresholds=np.quantile(mean13,[.5,.8,.95])
    categories={
        "zero_last26": (y[:,-26:].sum(axis=1)==0),
        "inactive_last13":(recent.sum(axis=1)==0),
        "newly_active":(prev.sum(axis=1)==0)&(recent.sum(axis=1)>0),
        "high_volume_top20pct":mean13>thresholds[1],
        "high_volume_top5pct":mean13>thresholds[2],
        "active_and_volume_ge2":active_mask(y),
        "seasonal_spike_history":(annual.max(axis=1)>4*np.maximum(.1,annual.mean(axis=1))),
    }
    groups={}
    for name,mask in categories.items():
        groups[name]={"series":int(mask.sum()),
                      "absolute_error":float(absrow[mask].sum()),
                      "error_share":float(absrow[mask].sum()/max(total,1)),
                      "signed_error":float(signedrow[mask].sum())}
    return {"series":len(y),"horizon":HORIZON,"absolute_error_units":total,
            "signed_error_units":float(signedrow.sum()),
            "actual_units":float(actual.sum()),
            "official_score":_metric(pred,actual),
            "top1pct_series_error_share":float(ordered[:count].sum()/max(total,1)),
            "top5pct_series_error_share":float(ordered[:k5].sum()/max(total,1)),
            "week_absolute_error_units":list(map(float,byweek)),
            "preorigin_cohorts":groups,
            "private_ids_emitted":False,
            "note":"Error shares descriptive only; do not use Phase2 for model selection."}


def rolling_validate(phase0:pd.DataFrame,*,baseline_trees:int=160,
                     active_trees:int=80,activity_trees:int=100)->dict:
    y=values(phase0)
    if y.shape[1]<HOLDOUT_ORIGIN+HORIZON:
        raise ValueError("Phase0 holdout target incomplete")
    folds=(*TRAIN_ORIGINS,HOLDOUT_ORIGIN)
    losses={name:{"abs":0.,"signed":0.} for name in NAMES}
    total=0.;scores={};diagnostics={}
    better={name:0 for name in NAMES}
    for origin in folds:
        past=phase0.iloc[:,:3+origin]
        baseline=incumbent(past,trees=baseline_trees)
        preds,meta=candidates(past,baseline,active_trees=active_trees,
                              activity_trees=activity_trees)
        target=y[:,origin:origin+HORIZON]
        scores[str(origin)]={name:_metric(p,target) for name,p in preds.items()}
        diagnostics[str(origin)]={"specialist":meta["specialist"],
                                  "adaptive":meta["adaptive"]}
        if origin!=HOLDOUT_ORIGIN:
            total+=float(target.sum())
            for name,p in preds.items():
                err=p-target
                losses[name]["abs"]+=float(np.abs(err).sum())
                losses[name]["signed"]+=float(err.sum())
                better[name]+=int(scores[str(origin)][name]<scores[str(origin)]["learned_activity"])
        print("VN1_NEXT5_FOLD="+str(origin),flush=True)
    train={k:(v["abs"]+abs(v["signed"]))/total for k,v in losses.items()}
    ordered=sorted(NAMES,key=lambda n:(train[n],NAMES.index(n)))
    winner=ordered[0]
    held=scores[str(HOLDOUT_ORIGIN)]
    accepted=(winner!="learned_activity" and
              better[winner]>=3 and
              train[winner] <= train["learned_activity"]-MIN_GAIN and
              held[winner] <= held["learned_activity"]-MIN_GAIN)
    return {"training_origins":list(TRAIN_ORIGINS),"holdout_origin":HOLDOUT_ORIGIN,
            "train_scores":train,"holdout_scores":held,"fold_scores":scores,
            "training_folds_improved":better,"diagnostics":diagnostics,
            "winner_train_only":winner,
            "selected":winner if accepted else "learned_activity",
            "accepted":bool(accepted),"minimum_gain":MIN_GAIN,
            "required_training_folds_improved":3,
            "candidate_registry":list(NAMES),
            "retrospective_not_blind":True}


def _sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(data_dir:Path,output_dir:Path,report:Path,*,phase:int,
           baseline_trees:int=160,active_trees:int=80,
           activity_trees:int=100)->dict:
    if phase not in (1,2):raise ValueError("Invalid phase")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Output directory not empty")
    selection=json.loads(report.read_text())
    if (selection["training_origins"]!=list(TRAIN_ORIGINS) or
        selection["holdout_origin"]!=HOLDOUT_ORIGIN or
        selection["candidate_registry"]!=list(NAMES)):
        raise ValueError("Historical receipt mismatch")
    hist=read_history(phase,data_dir)
    base=incumbent(hist,trees=baseline_trees)
    preds,details=candidates(hist,base,active_trees=active_trees,
                              activity_trees=activity_trees)
    preds["phase0_selected"]=preds[selection["selected"]].copy()
    output_dir.mkdir(parents=True,exist_ok=True)
    dest=output_dir/"frozen.npz"
    np.savez_compressed(dest,**preds)
    dates=list(pd.date_range(pd.Timestamp(hist.columns[-1])+pd.Timedelta(weeks=1),
                             periods=HORIZON,freq="7D").strftime("%Y-%m-%d"))
    manifest={"phase":phase,"series":len(hist),"dates":dates,
              "candidate_names":list(preds),"selection":selection["selected"],
              "accepted":selection["accepted"],"sha256":_sha(dest),
              "phase0_report_sha256":_sha(report),
              "model_diagnostics":details,"truth_not_read_on_freeze":True,
              "retrospective_not_blind":True}
    (output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("VN1_NEXT5_FROZEN="+json.dumps({k:manifest[k] for k in (
        "selection","sha256","series","phase")}),flush=True)
    return manifest


def score(data_dir:Path,output_dir:Path,truth:Path)->dict:
    receipt=json.loads((output_dir/"manifest.json").read_text())
    bundle=output_dir/"frozen.npz"
    if _sha(bundle)!=receipt["sha256"]:raise ValueError("Frozen SHA256 mismatch")
    hist=read_history(receipt["phase"],data_dir)
    observed=read_sales(truth)
    if (len(observed)!=receipt["series"] or
        not observed[KEYS].equals(hist[KEYS]) or
        list(observed.columns[3:])!=receipt["dates"]):
        raise ValueError("Incorrect target row/date alignment")
    actual=observed.iloc[:,3:].to_numpy(dtype=float)
    with np.load(bundle) as z:
        scores={name:_metric(np.asarray(z[name]),actual)
                for name in receipt["candidate_names"]}
    result={"phase":receipt["phase"],"scores":scores,
            "selected":receipt["selection"],
            "hash":receipt["sha256"],
            "retrospective_not_blind":True}
    (output_dir/"scores.json").write_text(json.dumps(result,indent=2)+"\n")
    print("VN1_NEXT5_SCORE="+json.dumps(result,sort_keys=True),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode",choices=("audit","historical","freeze","score"))
    ap.add_argument("--data-dir",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--report",type=Path)
    ap.add_argument("--archive",type=Path)
    ap.add_argument("--pred-key",default="learned_activity")
    ap.add_argument("--truth",type=Path)
    ap.add_argument("--phase",type=int,choices=(1,2),default=2)
    ap.add_argument("--baseline-trees",type=int,default=160)
    ap.add_argument("--active-trees",type=int,default=80)
    ap.add_argument("--activity-trees",type=int,default=100)
    a=ap.parse_args()
    if a.mode=="audit":
        if a.archive is None or a.truth is None:ap.error("audit requires --archive and --truth")
        hist=read_history(a.phase,a.data_dir)
        targets=read_sales(a.truth)
        if not targets[KEYS].equals(hist[KEYS]):raise ValueError("Unaligned audit keys")
        with np.load(a.archive) as bundle:
            report=audit_errors(hist,np.asarray(bundle[a.pred_key]),
                                 targets.iloc[:,3:].to_numpy(dtype=float))
        a.out.mkdir(parents=True,exist_ok=True)
        (a.out/"aggregate-audit.json").write_text(json.dumps(report,indent=2)+"\n")
        print("VN1_NEXT5_AUDIT="+json.dumps(report,sort_keys=True),flush=True)
    elif a.mode=="historical":
        phase0=read_sales(a.data_dir/"Phase 0 - Sales.csv")
        res=rolling_validate(phase0,baseline_trees=a.baseline_trees,
                              active_trees=a.active_trees,activity_trees=a.activity_trees)
        a.out.mkdir(parents=True,exist_ok=True)
        (a.out/"historical.json").write_text(json.dumps(res,indent=2)+"\n")
        print("VN1_NEXT5_HISTORICAL="+json.dumps({
            "train_scores":res["train_scores"],"holdout_scores":res["holdout_scores"],
            "selected":res["selected"],"accepted":res["accepted"]
        },sort_keys=True),flush=True)
    elif a.mode=="freeze":
        if a.report is None:ap.error("freeze requires --report")
        freeze(a.data_dir,a.out,a.report,phase=a.phase,
               baseline_trees=a.baseline_trees,active_trees=a.active_trees,
               activity_trees=a.activity_trees)
    else:
        if a.truth is None:ap.error("score requires --truth")
        score(a.data_dir,a.out,a.truth)


if __name__=="__main__":
    main()
