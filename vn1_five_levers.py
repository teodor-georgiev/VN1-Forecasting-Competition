"""Five leak-safe VN1 error-reduction experiments, never active by default.

Families: 1. L1/Huber/quantile objectives, 2. log/sqrt/season-normalized
targets, 3. lagged price features, 4. client-warehouse reconciliation,
5. trimmed leading-zero demand levels.

The control is the frozen learned-activity research recipe, computed using
only sales before each historical prediction origin. Candidate selection
uses fully observed Phase 0 folds 105,118,131,144; 157 is an acceptance
gate and cannot be used for weight/variant search. Phase 1 is retrospective
only. Private price data are optional, NEVER written to output artifacts.
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
from vn1_lgbm_ensemble import (
    LGBM_PARAMS, START_TRAIN_WEEK, TRAIN_STRIDE, _features, read_history,
)
from vn1_amsi_zero_rule_ablation import incumbent, _metric
from vn1_learned_activity import allocate_weekly, predict_activity

ORIGINS = (105, 118, 131, 144, 157)
MIN_GAIN = 0.001
BLEND = 0.30
MODEL_VARIANTS = {
    "loss_l1": ("regression_l1", "raw", False, None),
    "loss_huber": ("huber", "raw", False, None),
    "loss_quantile60": ("quantile", "raw", False, 0.60),
    "target_log1p": ("regression", "log1p", False, None),
    "target_sqrt": ("regression", "sqrt", False, None),
    "target_season_norm": ("regression_l1", "season_norm", False, None),
    "price_tweedie": ("tweedie", "raw", True, None),
    "price_l1": ("regression_l1", "raw", True, None),
}
POLICIES = ("learned_activity", *MODEL_VARIANTS, "hierarchy_client_warehouse",
            "trim_prelaunch_zeros")
FAMILIES = {
    "loss": ["loss_l1","loss_huber","loss_quantile60"],
    "targets": ["target_log1p","target_sqrt","target_season_norm"],
    "price": ["price_tweedie","price_l1"],
    "hierarchy": ["hierarchy_client_warehouse"],
    "trim": ["trim_prelaunch_zeros"],
}


def _sales(history: pd.DataFrame) -> np.ndarray:
    if list(history.columns[:3]) != list(KEYS) or history.duplicated(KEYS).any():
        raise ValueError("Invalid or duplicate VN1 keys")
    y = history.iloc[:, 3:].to_numpy(dtype=np.float64)
    if y.ndim != 2 or y.shape[1] < 65 or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Need finite nonnegative sales history with 65+ weeks")
    return y


def _aligned_prices(history: pd.DataFrame, price: pd.DataFrame | None) -> np.ndarray | None:
    if price is None:
        return None
    if list(price.columns[:3]) != list(KEYS) or not price[KEYS].equals(history[KEYS]):
        raise ValueError("Price keys do not match sales row order")
    if list(price.columns[3:]) != list(history.columns[3:]):
        raise ValueError("Price dates must match sales historical dates")
    numeric=price.iloc[:,3:].apply(pd.to_numeric,errors="coerce")
    return numeric.to_numpy(dtype=np.float32)


def read_price(path:Path, history:pd.DataFrame) -> pd.DataFrame:
    """Read private observed prices with key/date alignment; unknown stays NaN."""
    raw=pd.read_csv(path)
    if not set(KEYS).issubset(raw.columns) or raw.duplicated(KEYS).any():
        raise ValueError("Prices missing VN1 identifiers or duplicate keys")
    idx=pd.MultiIndex.from_frame(history[KEYS])
    table=raw.set_index(KEYS).reindex(idx).reindex(columns=history.columns[3:])
    # Completely missing price series are permissible (absence is a feature).
    return pd.concat([history[KEYS].reset_index(drop=True),
                      table.reset_index(drop=True)],axis=1)


def _price_features(price:pd.DataFrame,origin:int) -> dict[str,np.ndarray]:
    if origin < 52 or origin > len(price.columns)-3:
        raise ValueError("Price origin exceeds available history")
    p=price.iloc[:,3:3+origin].to_numpy(dtype=np.float32)
    good=np.isfinite(p) & (p>0)
    val=np.where(good,p,0)
    last=np.where(good,np.arange(origin)[None,:],-1).max(axis=1)
    recent=val[np.arange(len(p)),np.maximum(last,0)]
    def avg(n:int)->np.ndarray:
        count=good[:,-n:].sum(axis=1)
        return np.divide(val[:,-n:].sum(axis=1),count,
                         out=np.zeros(len(p),dtype=np.float32),where=count>0)
    mean4,mean13,mean52=avg(4),avg(13),avg(52)
    span=p[:,-13:]
    high=np.where(good[:,-13:],span,-np.inf).max(axis=1)
    low=np.where(good[:,-13:],span,np.inf).min(axis=1)
    spread=np.where(np.isfinite(high)&np.isfinite(low),
                    (high-low)/(mean13+1),0)
    return {
        "price_last_known":recent,
        "price_mean4":mean4,
        "price_mean13":mean13,
        "price_mean52":mean52,
        "price_ratio4_13":(mean4+1)/(mean13+1),
        "price_range13":np.clip(spread,0,20),
        "price_coverage13":good[:,-13:].mean(axis=1),
        "price_missing":(last<0).astype(float),
    }


def _X(history:pd.DataFrame, origin:int, prices:pd.DataFrame | None) -> pd.DataFrame:
    out=_features(history,origin)
    if prices is not None:
        for col,values in _price_features(prices,origin).items():
            out[col]=np.repeat(values.astype(np.float32),HORIZON)
    if not np.isfinite(out.to_numpy(dtype=np.float64)).all():
        raise ValueError("Nonfinite feature matrix")
    return out


def _fit_dataset(history:pd.DataFrame,price:pd.DataFrame | None):
    y=_sales(history)
    origins=list(range(START_TRAIN_WEEK,len(y.T)-HORIZON+1,TRAIN_STRIDE))
    if not origins:
        raise ValueError("No completed training origin")
    X=[]; targets=[]
    for origin in origins:
        X.append(_X(history,origin,price))
        targets.append(y[:,origin:origin+HORIZON].reshape(-1).astype(np.float32))
    features=pd.concat(X,ignore_index=True)
    labels=np.concatenate(targets)
    if len(features)!=len(labels) or max(origins)+HORIZON>y.shape[1]:
        raise AssertionError("Incomplete historical training labels")
    return features,labels,origins


def _transform_target(y:np.ndarray,feat:pd.DataFrame,name:str)->np.ndarray:
    if name=="raw": return y.copy()
    if name=="log1p": return np.log1p(y)
    if name=="sqrt": return np.sqrt(y)
    if name=="season_norm":
        return y / np.maximum(1,feat["statistical_baseline"].to_numpy(dtype=float))
    raise ValueError("Unknown target transform")


def _invert_target(values:np.ndarray,features:pd.DataFrame,name:str)->np.ndarray:
    x=np.asarray(values,dtype=float)
    if name=="raw": return np.maximum(0,x)
    if name=="log1p": return np.expm1(np.clip(x,0,16))
    if name=="sqrt": return np.maximum(0,x)**2
    if name=="season_norm":
        return np.maximum(0,x) * np.maximum(1,features["statistical_baseline"].to_numpy(dtype=float))
    raise ValueError("Unknown target transform")


def _train_predict_variants(history:pd.DataFrame,base:np.ndarray,
                            prices:pd.DataFrame | None, *, trees:int=80
                            ) -> dict[str,np.ndarray]:
    X,y,_=_fit_dataset(history,None)
    current=_X(history,_sales(history).shape[1],None)
    px=py=pcurrent=None
    if prices is not None:
        px,py,_=_fit_dataset(history,prices)
        pcurrent=_X(history,_sales(history).shape[1],prices)
    output={}
    for key,(objective,transform,has_price,quantile) in MODEL_VARIANTS.items():
        if has_price and prices is None:
            continue
        a=px if has_price else X
        labels=py if has_price else y
        prediction_features=pcurrent if has_price else current
        transformed=_transform_target(labels,a,transform)
        params={**LGBM_PARAMS,"objective":objective,
                "n_estimators":trees,"num_leaves":15,"min_child_samples":80,
                "learning_rate":0.05}
        params.pop("tweedie_variance_power",None)
        if objective=="tweedie":params["tweedie_variance_power"]=1.3
        if objective in ("quantile","huber"):params["alpha"]=quantile or .90
        reg=lgb.LGBMRegressor(**params)
        reg.fit(a,transformed,categorical_feature=["Client","Warehouse"])
        pred=_invert_target(reg.predict(prediction_features),prediction_features,transform)
        matrix=np.asarray(pred,dtype=np.float64).reshape(len(base),HORIZON)
        # Fixed, pre-origin plausibility cap; avoids log/sqrt instability.
        hist=_sales(history)
        cap=np.maximum(12,np.maximum(hist[:,-52:].max(axis=1)*6,
                                    hist[:,-13:].mean(axis=1)*8))
        matrix=np.clip(matrix,0,cap[:,None])
        output[key]=_preserve_weekly((1-BLEND)*base+BLEND*matrix,base)
    return output


def _preserve_weekly(candidate:np.ndarray,base:np.ndarray)->np.ndarray:
    p=np.asarray(candidate,dtype=np.float64)
    b=np.asarray(base,dtype=np.float64)
    if b.shape!=p.shape or b.ndim!=2 or b.shape[1]!=HORIZON:
        raise ValueError("Incorrect forecast shape")
    if not np.isfinite(p).all() or not np.isfinite(b).all() or (p<0).any() or (b<0).any():
        raise ValueError("Invalid forecast values")
    want=b.sum(axis=0)
    got=p.sum(axis=0)
    if np.any((want>0)&(got<=0)):
        return b.copy()
    factors=np.divide(want,got,out=np.ones_like(want),where=got>0)
    out=p*factors[None,:]
    if not np.allclose(out.sum(axis=0),want,atol=1e-5,rtol=1e-10):
        raise ArithmeticError("Weekly mass drift")
    return out


def _hierarchy(history:pd.DataFrame,base:np.ndarray)->np.ndarray:
    """Reconcile client×warehouse forecasts with pre-origin group annual level."""
    y=_sales(history)
    group=pd.MultiIndex.from_frame(history[["Client","Warehouse"]])
    codes,_=pd.factorize(group,sort=True)
    groups=int(codes.max())+1
    sums=np.zeros((groups,y.shape[1]),dtype=np.float64)
    np.add.at(sums,codes,y)
    matched=sums[:,-52:-39]
    prior=sums[:,-65:-52].mean(axis=1)
    last13=sums[:,-13:].mean(axis=1)
    mean4=sums[:,-4:].mean(axis=1)
    # Historical group expected total, with annual weeks smoothed toward 1.
    seasonal=np.clip((matched+10)/(prior[:,None]+10),.6,1.5)
    level=.7*last13+.3*mean4
    target=level[:,None]*(.65+.35*seasonal)
    out=base.copy()
    baseline_totals=np.zeros((groups,HORIZON),dtype=np.float64)
    np.add.at(baseline_totals,codes,base)
    for g in range(groups):
        idx=np.flatnonzero(codes==g)
        pred=baseline_totals[g]
        factor=np.divide(target[g],pred,out=np.ones_like(pred),where=pred>0)
        out[idx]*=np.clip(factor,.4,2.5)
    return _preserve_weekly(.7*base+.3*out,base)


def _trim_leading_zeros(history:pd.DataFrame,base:np.ndarray)->np.ndarray:
    """Short-lived series use post-launch average instead of pre-launch zeros."""
    y=_sales(history)
    t=y.shape[1]
    seen=y>0
    ever=seen.any(axis=1)
    first=np.where(seen,np.arange(t)[None,:],t).min(axis=1)
    age=np.maximum(1,t-first)
    post_mean=np.divide(y.sum(axis=1),age,
                        out=np.zeros(len(y)),where=ever)
    recent13=y[:,-13:].mean(axis=1)
    recent4=y[:,-4:].mean(axis=1)
    corrected=.5*post_mean+.3*recent13+.2*recent4
    denom=np.maximum(.1,recent13)
    factor=np.clip(corrected/denom,.4,2.0)
    # Only recently launched series: first activity in the last 78 weeks.
    mask=ever&(age<78)&(recent13>0)
    adjusted=base.copy()
    adjusted[mask]*=factor[mask,None]
    return _preserve_weekly(.8*base+.2*adjusted,base)


def candidates(history:pd.DataFrame,control:np.ndarray, *,
               prices:pd.DataFrame|None=None,trees:int=80,
               activity_trees:int=100) -> tuple[dict[str,np.ndarray],dict]:
    y=_sales(history)
    if control.shape!=(len(y),HORIZON):
        raise ValueError("Misaligned baseline")
    if prices is not None:_aligned_prices(history,prices)
    probability,meta=predict_activity(history,estimators=activity_trees)
    baseline=allocate_weekly(np.asarray(control,dtype=np.float64),probability)
    outputs={"learned_activity":baseline}
    outputs.update(_train_predict_variants(history,baseline,prices,trees=trees))
    outputs["hierarchy_client_warehouse"]=_hierarchy(history,baseline)
    outputs["trim_prelaunch_zeros"]=_trim_leading_zeros(history,baseline)
    expected=tuple(x for x in POLICIES if prices is not None or x not in FAMILIES["price"])
    if tuple(outputs)!=expected:
        raise AssertionError("Missing candidate experiment or unexpected ordering")
    for name,v in outputs.items():
        if v.shape!=baseline.shape or not np.isfinite(v).all() or (v<0).any():
            raise ValueError("Invalid output "+name)
    return outputs,{"activity":meta,"prices_available":prices is not None,
                    "policies":list(outputs)}


def _phase0(history:pd.DataFrame,prices:pd.DataFrame|None,*,trees:int,
            activity_trees:int,baseline_trees:int) -> dict:
    y=_sales(history)
    if y.shape[1] < ORIGINS[-1]+HORIZON:
        raise ValueError("Phase 0 holdout not complete")
    loss={}
    scores={}
    total=0.
    summaries={}
    for origin in ORIGINS:
        past=history.iloc[:,:3+origin]
        pp=prices.iloc[:,:3+origin] if prices is not None else None
        control=incumbent(past,trees=baseline_trees)
        preds,meta=candidates(past,control,prices=pp,trees=trees,
                              activity_trees=activity_trees)
        targets=y[:,origin:origin+HORIZON]
        scores[str(origin)]={k:_metric(v,targets) for k,v in preds.items()}
        summaries[str(origin)]=meta
        if origin!=ORIGINS[-1]:
            total+=float(targets.sum())
            for name,pred in preds.items():
                error=pred-targets
                loss.setdefault(name,{"abs":0.,"signed":0.})
                loss[name]["abs"]+=float(np.abs(error).sum())
                loss[name]["signed"]+=float(error.sum())
        print("VN1_FIVE_FOLD="+str(origin),flush=True)
    train={name:(v["abs"]+abs(v["signed"]))/total for name,v in loss.items()}
    winner=min(train,key=lambda n:(train[n],list(train).index(n)))
    holdout=scores[str(ORIGINS[-1])]
    accepted=(winner!="learned_activity" and
              train[winner] <= train["learned_activity"]-MIN_GAIN and
              holdout[winner] <= holdout["learned_activity"]-MIN_GAIN)
    return {"origins_train":list(ORIGINS[:-1]),"holdout_origin":ORIGINS[-1],
            "training_scores":train,"holdout_scores":holdout,
            "all_fold_scores":scores,"experiment_details":summaries,
            "winner_train_only":winner,
            "selected":winner if accepted else "learned_activity",
            "accepted":bool(accepted),"minimum_absolute_gain":MIN_GAIN,
            "families":FAMILIES,"prices_loaded":prices is not None,
            "caveat":"Post-hoc public Phase 1/2 research, not blind competition."}


def historical(data_dir:Path,output_dir:Path,*,price0:Path|None,
               trees:int,activity_trees:int,baseline_trees:int)->dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Do not overwrite existing receipts")
    phase0=read_sales(data_dir/"Phase 0 - Sales.csv")
    prices=read_price(price0,phase0) if price0 is not None else None
    outcome=_phase0(phase0,prices,trees=trees,activity_trees=activity_trees,
                    baseline_trees=baseline_trees)
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/"historical.json").write_text(json.dumps(outcome,indent=2)+"\n")
    print("VN1_FIVE_HISTORICAL="+json.dumps({
        "selected":outcome["selected"],"accepted":outcome["accepted"],
        "training":outcome["training_scores"],"holdout":outcome["holdout_scores"],
        "prices_loaded":outcome["prices_loaded"]
    },sort_keys=True),flush=True)
    return outcome


def phase1(data_dir:Path,output_dir:Path,*,price0:Path|None,
           trees:int,activity_trees:int,baseline_trees:int) -> dict:
    """Frozen Phase 0-selected recipe; score Phase 1 only after prediction archive."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Do not overwrite existing forecasts")
    phase0=read_sales(data_dir/"Phase 0 - Sales.csv")
    prices=read_price(price0,phase0) if price0 is not None else None
    chosen=_phase0(phase0,prices,trees=trees,activity_trees=activity_trees,
                   baseline_trees=baseline_trees)
    control=incumbent(phase0,trees=baseline_trees)
    preds,meta=candidates(phase0,control,prices=prices,trees=trees,
                          activity_trees=activity_trees)
    preds["phase0_selected"]=preds[chosen["selected"]].copy()
    output_dir.mkdir(parents=True,exist_ok=True)
    archived=output_dir/"frozen-phase1.npz"
    np.savez_compressed(archived,**preds)
    manifest={"phase":1,"forecast_names":list(preds),"selected":chosen["selected"],
              "historical_acceptance":chosen["accepted"],
              "sha256":hashlib.sha256(archived.read_bytes()).hexdigest(),
              "prediction_dates":list(pd.date_range(
                  pd.Timestamp(phase0.columns[-1])+pd.Timedelta(weeks=1),
                  periods=HORIZON,freq="7D").strftime("%Y-%m-%d")),
              "num_series":len(phase0),"prices_used":prices is not None,
              "truth_was_not_opened_when_frozen":True}
    (output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("VN1_FIVE_FROZEN="+json.dumps(manifest,sort_keys=True),flush=True)
    phase1_targets=read_sales(data_dir/"Phase 1 - Sales.csv")
    if not phase1_targets[KEYS].equals(phase0[KEYS]) or list(phase1_targets.columns[3:]) != manifest["prediction_dates"]:
        raise ValueError("Invalid Phase 1 target row/dates")
    if hashlib.sha256(archived.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Frozen forecasts changed before scoring")
    y=phase1_targets.iloc[:,3:].to_numpy(dtype=np.float64)
    scores={k:_metric(v,y) for k,v in preds.items()}
    receipt={"phase":1,"series":len(y),"scores":scores,
             "selected":chosen["selected"],
             "accepted":chosen["accepted"],"frozen_sha256":manifest["sha256"],
             "retrospective_not_blind":True}
    (output_dir/"scores-phase1.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("VN1_FIVE_PHASE1="+json.dumps(receipt,sort_keys=True),flush=True)
    return receipt


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command",choices=("historical","phase1"))
    ap.add_argument("--data-dir",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--price0",type=Path)
    ap.add_argument("--trees",type=int,default=80)
    ap.add_argument("--activity-trees",type=int,default=100)
    ap.add_argument("--baseline-trees",type=int,default=160)
    args=ap.parse_args()
    kwargs=dict(price0=args.price0,trees=args.trees,
                activity_trees=args.activity_trees,baseline_trees=args.baseline_trees)
    if args.command=="historical":
        historical(args.data_dir,args.out,**kwargs)
    else:
        phase1(args.data_dir,args.out,**kwargs)


if __name__=="__main__":
    main()
