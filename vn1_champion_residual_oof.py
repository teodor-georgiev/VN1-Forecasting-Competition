"""Out-of-fold residual specialist on the verified VN1 ARIMA+AMSI champion.

The target is ACTUAL SALES - CHAMPION PREDICTION, NOT residuals to a
statistical surrogate. Each residual training row is generated from a
fully historical champion forecast made 13+ weeks before the fitting
cutoff, and labeled only after its entire forecast horizon is observed.

A nested forward holdout calibrates a regime reliability gate. The
outer historical selection uses origins 118/131/144, holdout 157, and
selects on training-only official-score aggregation. Phase 1/2 forecasts
must be frozen and SHA-256 pinned BEFORE scoring the later truth.

Research only: Phase 2 is retrospectively known, not a blind leaderboard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from vn1_from_scratch import HORIZON, KEYS, read_sales
from vn1_lgbm_ensemble import _features, read_history
from vn1_amsi_zero_rule_ablation import incumbent, _metric
from vn1_learned_activity import allocate_weekly, predict_activity
from vn1_amsi_specialists import amsi_forecast
from vn1_winner_arima_amsi_research import (
    _mix, _observed, client_autoarima,
)

CHAMPION_NAME = "arima30_client_amsi_last10"
CHAMPION_PHASE2_REFERENCE = 0.4693382738889349
OOF_START = 105
OOF_STEP = 13
TRAIN_EVAL_ORIGINS = (118, 131, 144)
HOLDOUT_ORIGIN = 157
MIN_GAIN = 0.001
MIN_IMPROVED_FOLDS = 2
PREDICTION_NAMES = (
    "champion",
    "residual_uniform20",
    "residual_uniform35",
    "residual_gated20",
    "residual_gated35",
)
PARAMS = dict(
    objective="regression_l1",
    n_estimators=100,
    learning_rate=.05,
    num_leaves=15,
    min_child_samples=80,
    max_bin=127,
    reg_lambda=4.,
    colsample_bytree=.9,
    subsample=.9,
    subsample_freq=1,
    random_state=2026,
    n_jobs=2,
    verbosity=-1,
)


def _history_matrix(history: pd.DataFrame) -> np.ndarray:
    return _observed(history)


def _active_mask(y: np.ndarray) -> np.ndarray:
    recent=y[:,-13:]
    return (recent.mean(axis=1)>=2.) & ((recent>0).mean(axis=1)>=.5)


def _regimes(y: np.ndarray) -> np.ndarray:
    """No future observations; identify stable / volatile / shifting regimes."""
    recent=y[:,-13:]
    prev=y[:,-26:-13]
    level=recent.mean(axis=1)
    cv=recent.std(axis=1)/np.maximum(level,.5)
    trend=(recent.sum(axis=1)+3.)/(prev.sum(axis=1)+3.)
    regime=np.where(cv<=.9,0,1)
    regime=np.where((trend>=1.5)|(trend<=.67),2,regime)
    return regime.astype(np.int8)


def _normalize(proposed: np.ndarray, reference: np.ndarray) -> np.ndarray:
    p=np.asarray(proposed,dtype=np.float64)
    b=np.asarray(reference,dtype=np.float64)
    if p.shape!=b.shape or p.ndim!=2 or p.shape[1]!=HORIZON:
        raise ValueError("Forecast shape mismatch")
    if (not np.isfinite(p).all() or not np.isfinite(b).all()
        or (p<0).any() or (b<0).any()):
        raise ValueError("Forecast values invalid")
    needed=b.sum(axis=0)
    got=p.sum(axis=0)
    if np.any((needed>0)&(got<=0)):
        return b.copy()
    ratio=np.divide(needed,got,out=np.ones_like(needed),where=got>0)
    out=p*ratio[None,:]
    if not np.allclose(out.sum(axis=0),needed,atol=1e-5,rtol=1e-10):
        raise ArithmeticError("Champion aggregate demand changed")
    return out


def champion_forecast(history: pd.DataFrame, *, trees:int=160,
                      activity_trees:int=100, client_cap:int=16
                      ) -> tuple[np.ndarray,dict]:
    """Exactly the preselected PR17 policy, skipping its unused candidate fits."""
    control=incumbent(history,trees=trees)
    probability, pinfo=predict_activity(history,estimators=activity_trees)
    baseline=allocate_weekly(control,probability)
    arima, arima_info=client_autoarima(history,baseline,cap=client_cap)
    amsi=amsi_forecast(history,anchor="last",seasonality=.45,group_shrink=.55)
    output=_mix(baseline,arima,amsi,.30,.10)
    return output,{"activity":pinfo,"clients":arima_info,
                   "policy":CHAMPION_NAME}


def _regression_features(history:pd.DataFrame,
                         forecast:np.ndarray) -> pd.DataFrame:
    y=_history_matrix(history)
    n=len(y)
    if forecast.shape!=(n,HORIZON) or not np.isfinite(forecast).all():
        raise ValueError("Misaligned champion")
    feat=_features(history,y.shape[1])
    x=feat.copy()
    x["champion_prediction"]=forecast.reshape(-1).astype("float32")
    x["champion_log1p"]=np.log1p(forecast.reshape(-1)).astype("float32")
    x["champion_minus_stat"]=(
        forecast.reshape(-1)-feat["statistical_baseline"].to_numpy()).astype("float32")
    last=y[:,-13:]
    last_mean=last.mean(axis=1)
    cv=last.std(axis=1)/np.maximum(last_mean,.5)
    past_mean=y[:,-26:-13].mean(axis=1)
    for k,v in {
        "recent_cv13":cv,
        "last13_mean":last_mean,
        "trend13_vs_prev13":(last_mean+1.)/(past_mean+1.),
        "recent_active_fraction":(last>0).mean(axis=1),
        "recent_regime":_regimes(y).astype(float),
    }.items():
        x[k]=np.repeat(np.asarray(v,dtype=np.float32),HORIZON)
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite residual feature")
    return x


def completed_oof_origins(t:int) -> list[int]:
    if t<OOF_START+HORIZON:
        return []
    return list(range(OOF_START,t-HORIZON+1,OOF_STEP))


def make_oof_record(full_history:pd.DataFrame, origin:int, *,
                    trees:int=160,activity_trees:int=100,
                    client_cap:int=16) -> dict:
    """Feature and label at origin S. No observation after S enters features."""
    y=full_history.iloc[:,3:].to_numpy(dtype=np.float64)
    if origin<OOF_START or origin+HORIZON>y.shape[1]:
        raise ValueError("OOF origin labels are not fully observed")
    past=full_history.iloc[:,:3+origin]
    champ,metadata=champion_forecast(
        past,trees=trees,activity_trees=activity_trees,client_cap=client_cap)
    observed=_history_matrix(past)
    eligible=_active_mask(observed)
    X=_regression_features(past,champ)
    chosen=np.repeat(eligible,HORIZON)
    actual=y[:,origin:origin+HORIZON]
    residual=(actual-champ).reshape(-1)[chosen].astype(np.float32)
    if len(residual)==0:
        raise ValueError("No active series eligible for residual training")
    return {
        "origin":origin,
        "X":X.loc[chosen].reset_index(drop=True),
        "residual":residual,
        "eligible_series":int(eligible.sum()),
        "champion":champ,
        "truth":actual.copy(),
        "regime":_regimes(observed),
        "mask":eligible,
        "full_features":X,
        "meta":metadata,
    }


def _fit(records:list[dict],*,trees:int=100):
    if not records:
        raise ValueError("Need at least one completed out-of-fold origin")
    X=pd.concat([record["X"] for record in records],ignore_index=True)
    y=np.concatenate([record["residual"] for record in records])
    if len(X)!=len(y) or len(y)<100:
        raise ValueError("Too few complete historical residual rows")
    model=lgb.LGBMRegressor(**dict(PARAMS,n_estimators=trees))
    model.fit(X,y,categorical_feature=["Client","Warehouse"])
    return model


def _corrections(model,history:pd.DataFrame,champion:np.ndarray) -> np.ndarray:
    y=_history_matrix(history)
    out=np.asarray(model.predict(_regression_features(history,champion)),
                   dtype=np.float64).reshape(len(y),HORIZON)
    cap=np.maximum(3.,2.0*y[:,-13:].mean(axis=1))
    out=np.clip(out,-cap[:,None],cap[:,None])
    out[~_active_mask(y),:]=0.
    if not np.isfinite(out).all():raise ArithmeticError("Invalid correction")
    return out


def _gate_weights(records:list[dict],*,trees:int=100) -> tuple[np.ndarray,dict]:
    """Forward-only reliability: train earlier folds, calibrate on LAST completed fold.

    0/0.2/0.35 per stable/volatile/trending group. Require >=100 group
    series and >=1% improvement in group absolute errors before trusting
    correction. Never train the calibration model on its own targets.
    """
    weights=np.zeros(3,dtype=np.float64)
    details={"calibration_origin":None,"groups":{},"nested_train_origins":[]}
    if len(records)<2:
        return weights,details
    calib=records[-1]
    previous=records[:-1]
    model=_fit(previous,trees=trees)
    n=len(calib["regime"])
    allcorr=np.asarray(model.predict(calib["full_features"]),
                       dtype=float).reshape(n,HORIZON)
    past_mean=calib["full_features"]["last13_mean"].to_numpy().reshape(n,HORIZON)[:,0]
    allcorr=np.clip(allcorr,-np.maximum(3,2*past_mean)[:,None],
                    np.maximum(3,2*past_mean)[:,None])
    allcorr[~calib["mask"]]=0
    for regime in range(3):
        mask=(calib["regime"]==regime)&calib["mask"]
        n_series=int(mask.sum())
        if n_series<100:
            details["groups"][str(regime)]={"series":n_series,"weight":0.,
                                              "status":"insufficient historical series"}
            continue
        reference=np.abs(calib["champion"][mask]-calib["truth"][mask]).sum()
        best=0.; best_error=reference
        # A conservative margin; no optimization over post-origin labels.
        for weight in (.20,.35):
            adjusted=np.maximum(0,calib["champion"][mask]+weight*allcorr[mask])
            error=np.abs(adjusted-calib["truth"][mask]).sum()
            if error <= .99*reference and error<best_error:
                best=weight;best_error=error
        weights[regime]=best
        details["groups"][str(regime)]={
            "series":n_series,"weight":best,
            "historical_absolute_gain":float(reference-best_error)}
    details["calibration_origin"]=calib["origin"]
    details["nested_train_origins"]=[int(x["origin"]) for x in previous]
    return weights,details


def candidate_forecasts(history:pd.DataFrame, *,
                        oof_records:list[dict],champion:np.ndarray,
                        residual_trees:int=100) -> tuple[dict[str,np.ndarray],dict]:
    """Create five predetermined policies using only completed OOF records."""
    t=_history_matrix(history).shape[1]
    if not oof_records or any(r["origin"]+HORIZON>t for r in oof_records):
        raise ValueError("Residual training would see future labels")
    if len(set(r["origin"] for r in oof_records)) != len(oof_records):
        raise ValueError("Duplicate OOF origin")
    if sorted(r["origin"] for r in oof_records)!=[r["origin"] for r in oof_records]:
        raise ValueError("OOF chronology violated")
    model=_fit(oof_records,trees=residual_trees)
    corrections=_corrections(model,history,champion)
    regime=_regimes(_history_matrix(history))
    gate,details=_gate_weights(oof_records,trees=residual_trees)
    out={"champion":champion.copy()}
    for name,amount in (("residual_uniform20",.20),("residual_uniform35",.35)):
        out[name]=_normalize(np.maximum(0,champion+amount*corrections),champion)
    for name,multiplier in (("residual_gated20",.20),("residual_gated35",.35)):
        # Gate learned 0/.2/.35 from previous completed fold only.
        local=np.minimum(gate[regime],multiplier)
        out[name]=_normalize(np.maximum(0,champion+local[:,None]*corrections),champion)
    if tuple(out)!=PREDICTION_NAMES:raise AssertionError("Candidate list changed")
    return out,{
        "completed_oof_origins":[int(r["origin"]) for r in oof_records],
        "rows_fit":int(sum(len(r["residual"]) for r in oof_records)),
        "active_series":int(_active_mask(_history_matrix(history)).sum()),
        "gate":details,"fitted_model":"LightGBM L1"
    }


def _completed_records(phase0:pd.DataFrame,t:int,cache:dict[int,dict],*,
                       trees:int=160,activity_trees:int=100,client_cap:int=16)->list[dict]:
    result=[]
    for origin in completed_oof_origins(t):
        if origin not in cache:
            cache[origin]=make_oof_record(
                phase0,origin,trees=trees,activity_trees=activity_trees,
                client_cap=client_cap)
        result.append(cache[origin])
    return result


def historical_validation(phase0:pd.DataFrame,*,trees:int=160,
                          activity_trees:int=100,client_cap:int=16,
                          residual_trees:int=100)->dict:
    y=_history_matrix(phase0)
    if y.shape[1] < HOLDOUT_ORIGIN+HORIZON:
        raise ValueError("Full historical holdout missing")
    losses={name:{"abs":0.,"signed":0.} for name in PREDICTION_NAMES}
    totals=0.;fold_scores={};diagnostics={};wins={name:0 for name in PREDICTION_NAMES}
    cache={}
    for t in (*TRAIN_EVAL_ORIGINS,HOLDOUT_ORIGIN):
        past=phase0.iloc[:,:3+t]
        records=_completed_records(phase0,t,cache,trees=trees,
                                   activity_trees=activity_trees,client_cap=client_cap)
        if records[-1]["origin"]+HORIZON>t:
            raise AssertionError("Future label leaked into rolling fit")
        champ,_=champion_forecast(past,trees=trees,
                                 activity_trees=activity_trees,client_cap=client_cap)
        pred,meta=candidate_forecasts(past,oof_records=records,
                                      champion=champ,residual_trees=residual_trees)
        truth=y[:,t:t+HORIZON]
        fold_scores[str(t)]={name:_metric(v,truth) for name,v in pred.items()}
        diagnostics[str(t)]=meta
        if t!=HOLDOUT_ORIGIN:
            totals+=float(truth.sum())
            for name,v in pred.items():
                err=v-truth
                losses[name]["abs"]+=float(np.abs(err).sum())
                losses[name]["signed"]+=float(err.sum())
                if fold_scores[str(t)][name]<fold_scores[str(t)]["champion"]:
                    wins[name]+=1
        print("VN1_CHAMPION_RESIDUAL_FOLD="+str(t),flush=True)
    scores={name:(v["abs"]+abs(v["signed"]))/totals for name,v in losses.items()}
    winner=min(PREDICTION_NAMES,key=lambda n:(scores[n],PREDICTION_NAMES.index(n)))
    holdout=fold_scores[str(HOLDOUT_ORIGIN)]
    accepted=(winner!="champion" and
              scores[winner] <= scores["champion"]-MIN_GAIN and
              holdout[winner] <= holdout["champion"]-MIN_GAIN and
              wins[winner]>=MIN_IMPROVED_FOLDS)
    return {
        "training_origins":list(TRAIN_EVAL_ORIGINS),
        "holdout_origin":HOLDOUT_ORIGIN,
        "training_scores":scores,
        "holdout_scores":holdout,
        "fold_scores":fold_scores,
        "folds_better_than_champion":wins,
        "winner_training_only":winner,
        "selected":winner if accepted else "champion",
        "accepted":bool(accepted),
        "threshold_absolute":MIN_GAIN,
        "minimum_folds_better":MIN_IMPROVED_FOLDS,
        "candidate_names":list(PREDICTION_NAMES),
        "diagnostics":diagnostics,
        "retrospective_not_blind":True
    }


def _sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(data_dir:Path, output_dir:Path, report_path:Path, *,
           phase:int,trees:int=160,activity_trees:int=100,
           client_cap:int=16,residual_trees:int=100)->dict:
    if phase not in (1,2):raise ValueError("Unknown forecast phase")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Never overwrite frozen forecasts")
    receipt=json.loads(report_path.read_text())
    if (receipt.get("training_origins")!=list(TRAIN_EVAL_ORIGINS)
        or receipt.get("holdout_origin")!=HOLDOUT_ORIGIN
        or receipt.get("candidate_names")!=list(PREDICTION_NAMES)):
        raise ValueError("Unrecognized historical selection")
    train=receipt["training_scores"];holdout=receipt["holdout_scores"]
    winner=receipt["winner_training_only"]
    expected=(winner!="champion" and
              train[winner]<=train["champion"]-MIN_GAIN and
              holdout[winner]<=holdout["champion"]-MIN_GAIN and
              receipt["folds_better_than_champion"][winner]>=MIN_IMPROVED_FOLDS)
    if expected!=receipt["accepted"] or receipt["selected"]!=(winner if expected else "champion"):
        raise ValueError("Historical gate integrity check failed")
    history=read_history(phase,data_dir)
    t=_history_matrix(history).shape[1]
    cache={}
    records=_completed_records(history,t,cache,trees=trees,
                               activity_trees=activity_trees,client_cap=client_cap)
    champion,info=champion_forecast(history,trees=trees,
                                   activity_trees=activity_trees,client_cap=client_cap)
    predictions,meta=candidate_forecasts(history,oof_records=records,
                                         champion=champion,residual_trees=residual_trees)
    predictions["historical_selected"]=predictions[receipt["selected"]].copy()
    output_dir.mkdir(parents=True,exist_ok=True)
    dest=output_dir/"frozen.npz"
    np.savez_compressed(dest,**predictions)
    dates=list(pd.date_range(pd.Timestamp(history.columns[-1])+pd.Timedelta(weeks=1),
                             periods=HORIZON,freq="7D").strftime("%Y-%m-%d"))
    manifest={
        "phase":phase, "series":len(history),"dates":dates,
        "champion":CHAMPION_NAME,"selected":receipt["selected"],
        "accepted":receipt["accepted"],
        "candidate_names":list(predictions),"archive_sha256":_sha(dest),
        "phase0_receipt_sha256":_sha(report_path),
        "phase0_sales_sha256":_sha(data_dir/"Phase 0 - Sales.csv"),
        "phase1_sales_sha256":(_sha(data_dir/"Phase 1 - Sales.csv") if phase==2 else None),
        "fitting_diagnostics":meta,
        "no_target_labels_loaded_during_freeze":True,
        "no_private_target_export":True,
        "retrospective_not_blind":True
    }
    (output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("VN1_CHAMPION_RESIDUAL_FROZEN="+json.dumps({
        k:manifest[k] for k in ("phase","archive_sha256","selected","series")
    },sort_keys=True),flush=True)
    return manifest


def score(data_dir:Path, output_dir:Path,truth:Path)->dict:
    manifest=json.loads((output_dir/"manifest.json").read_text())
    path=output_dir/"frozen.npz"
    if _sha(path)!=manifest["archive_sha256"]:
        raise ValueError("Frozen forecast SHA-256 mismatch")
    hist=read_history(manifest["phase"],data_dir)
    actual=read_sales(truth)
    if (len(actual)!=manifest["series"] or
        not actual[KEYS].equals(hist[KEYS]) or
        list(actual.columns[3:])!=manifest["dates"]):
        raise ValueError("Forecast series IDs/dates misaligned")
    y=actual.iloc[:,3:].to_numpy(dtype=float)
    with np.load(path) as bundle:
        scores={name:_metric(np.asarray(bundle[name],dtype=float),y)
                for name in manifest["candidate_names"]}
    result={"phase":manifest["phase"],"scores":scores,
            "historical_selected":manifest["selected"],
            "archived_sha256":manifest["archive_sha256"],
            "retrospective_not_blind":True}
    (output_dir/"score.json").write_text(json.dumps(result,indent=2)+"\n")
    print("VN1_CHAMPION_RESIDUAL_SCORE="+json.dumps(result,sort_keys=True),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("operation",choices=("historical","freeze","score"))
    ap.add_argument("--data-dir",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--report",type=Path)
    ap.add_argument("--truth",type=Path)
    ap.add_argument("--phase",type=int,choices=(1,2),default=2)
    ap.add_argument("--trees",type=int,default=160)
    ap.add_argument("--activity-trees",type=int,default=100)
    ap.add_argument("--client-cap",type=int,default=16)
    ap.add_argument("--residual-trees",type=int,default=100)
    args=ap.parse_args()
    kwargs=dict(trees=args.trees,activity_trees=args.activity_trees,
                client_cap=args.client_cap,residual_trees=args.residual_trees)
    if args.operation=="historical":
        if args.out.exists() and any(args.out.iterdir()):
            raise FileExistsError("Historical receipt already exists")
        original=read_sales(args.data_dir/"Phase 0 - Sales.csv")
        report=historical_validation(original,**kwargs)
        args.out.mkdir(parents=True,exist_ok=True)
        (args.out/"historical.json").write_text(json.dumps(report,indent=2)+"\n")
        print("VN1_CHAMPION_RESIDUAL_HISTORICAL="+json.dumps({
            "training":report["training_scores"],"holdout":report["holdout_scores"],
            "selected":report["selected"],"accepted":report["accepted"]
        },sort_keys=True),flush=True)
    elif args.operation=="freeze":
        if args.report is None:ap.error("freeze requires --report")
        freeze(args.data_dir,args.out,args.report,phase=args.phase,**kwargs)
    else:
        if args.truth is None:ap.error("score requires --truth")
        score(args.data_dir,args.out,args.truth)


if __name__=="__main__":
    main()
