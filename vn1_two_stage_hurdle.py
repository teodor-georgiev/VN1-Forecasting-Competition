"""Experimental two-stage VN1 hurdle: P(any 13-week sale) × E(units | sale).

Predict demand *occurrence* and conditional positive demand *separately*, then
distribute each series' expected total across its incumbent's 13-week profile.
A weekly, volume-neutral correction isolates the allocation improvement from
aggregate forecast-bias changes. Fit on completed historical origin labels only.

Research-only. Historical Phase 0 acceptance cannot make public Phase 1/2
retrospective scores independent blind evidence.
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
from vn1_lgbm_ensemble import read_history
from vn1_amsi_zero_rule_ablation import incumbent, _metric
from vn1_learned_activity import (
    ORIGINS, MIN_GAIN, activity_features, history_values, predict_activity,
    allocate_weekly, training_origins,
)

# The successful previous research policy is the comparator, not Phase 2 truth.
POLICIES = ("activity_p100", "hurdle25", "hurdle50", "hurdle75", "hurdle100")
PARAMS = dict(
    objective="regression",
    learning_rate=0.05,
    num_leaves=15,
    n_estimators=100,
    min_child_samples=80,
    reg_lambda=4.0,
    colsample_bytree=0.9,
    max_bin=127,
    random_state=2026,
    n_jobs=2,
    verbosity=-1,
)


def fit_conditional(history: pd.DataFrame, *, n_estimators: int = 100):
    """Only positive completed 13-week totals become conditional training labels.

    Return training metadata for auditing chronology and class imbalance.
    """
    y = history_values(history)
    origins = training_origins(y.shape[1])
    if not origins:
        raise ValueError("Insufficient fully observed training origins")
    positive_x, positive_y = [], []
    counts = []
    for origin in origins:
        # No feature at origin may use observations from its 13-week target.
        past = history.iloc[:, :len(KEYS) + origin]
        future_total = y[:, origin:origin+HORIZON].sum(axis=1)
        mask = future_total > 0
        counts.append(int(mask.sum()))
        if mask.any():
            positive_x.append(activity_features(past).loc[mask])
            positive_y.append(np.log1p(future_total[mask]))
    if not positive_x:
        return None, 0.0, {"origins": origins, "positive_counts": counts}
    X = pd.concat(positive_x, ignore_index=True)
    target = np.concatenate(positive_y)
    if X.shape[0] != target.size or not np.isfinite(target).all():
        raise ValueError("Conditional training misaligned or nonfinite")
    fallback = float(np.expm1(target.mean()))
    model = None
    if X.shape[0] >= 10 and np.ptp(target) >= 1e-10:
        model = lgb.LGBMRegressor(**dict(PARAMS, n_estimators=n_estimators))
        model.fit(X, target)
    return model, fallback, {"origins": origins, "positive_counts": counts,
                              "positive_samples": int(len(target))}


def predict_positive_total(history: pd.DataFrame, *, n_estimators: int = 100
                           ) -> tuple[np.ndarray, dict]:
    model, fallback, details = fit_conditional(history, n_estimators=n_estimators)
    if model is None:
        predicted = np.full(len(history), fallback, dtype=np.float64)
    else:
        log_amount = model.predict(activity_features(history))
        # Bound numerics at a generous count level; all clipping is independent
        # of unobserved future targets.
        predicted = np.expm1(np.clip(log_amount, 0, np.log1p(1e9)))
    if predicted.shape != (len(history),) or not np.isfinite(predicted).all():
        raise ValueError("Invalid conditional quantity forecast")
    return np.maximum(predicted, 0), details


def hurdle_matrix(base: np.ndarray, occurrence_p: np.ndarray,
                  conditional_total: np.ndarray) -> np.ndarray:
    """Expected 13-week units = P(any sales) * E(13-week units | any sales).

    The original incumbent supplies ONLY the within-series weekly profile;
    the second stage independently supplies conditional series quantity.
    """
    baseline = np.asarray(base,dtype=np.float64)
    p = np.asarray(occurrence_p,dtype=np.float64)
    mu = np.asarray(conditional_total,dtype=np.float64)
    n = baseline.shape[0]
    if (baseline.ndim != 2 or baseline.shape[1] != HORIZON or
        p.shape != (n,) or mu.shape != (n,) or
        not np.isfinite(baseline).all() or not np.isfinite(p).all() or
        not np.isfinite(mu).all() or (baseline < 0).any() or
        (p < 0).any() or (p > 1).any() or (mu < 0).any()):
        raise ValueError("Invalid hurdle model arrays")
    total = baseline.sum(axis=1)
    profile = np.divide(baseline, total[:,None],
                        out=np.full_like(baseline, 1/HORIZON),
                        where=total[:,None] > 0)
    return np.maximum(0.0, profile * (p*mu)[:,None])


def normalize_weekly(candidate: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Preserve original predicted units in every future week."""
    candidate = np.asarray(candidate,dtype=np.float64)
    baseline = np.asarray(baseline,dtype=np.float64)
    if candidate.shape != baseline.shape or candidate.ndim != 2:
        raise ValueError("Mismatched forecast grids")
    if not np.isfinite(candidate).all() or (candidate < 0).any() or not np.isfinite(baseline).all() or (baseline < 0).any():
        raise ValueError("Invalid forecast values")
    wanted = baseline.sum(axis=0)
    got = candidate.sum(axis=0)
    if np.any((got == 0) & (wanted > 0)):
        return baseline.copy()
    result = candidate * np.divide(wanted,got,out=np.ones_like(wanted),where=got>0)[None,:]
    if not np.allclose(result.sum(axis=0), wanted, atol=1e-5, rtol=1e-10):
        raise ArithmeticError("Week-by-week bias preservation failed")
    return result


def candidates(history: pd.DataFrame, baseline: np.ndarray,
               *, occurrence_trees: int = 100, conditional_trees: int = 100
               ) -> tuple[dict[str,np.ndarray], dict]:
    p, occurrence_meta = predict_activity(history, estimators=occurrence_trees)
    mu, conditional_meta = predict_positive_total(history, n_estimators=conditional_trees)
    base = np.asarray(baseline,dtype=np.float64)
    if base.shape != (len(history), HORIZON):
        raise ValueError("Wrong baseline shape")
    activity = allocate_weekly(base, p)
    hurdle = hurdle_matrix(base, p, mu)
    # Mixing *before* weekly normalization captures conditional quantity
    # information without changing the historical incumbent aggregate bias.
    outputs = {"activity_p100": activity}
    for weight, name in ((0.25,"hurdle25"),(0.5,"hurdle50"),
                         (0.75,"hurdle75"),(1.0,"hurdle100")):
        outputs[name] = normalize_weekly((1-weight)*activity+weight*hurdle,base)
    if tuple(outputs) != POLICIES:
        raise AssertionError("Unexpected candidate registry")
    for name, values in outputs.items():
        if values.shape != base.shape or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Bad forecast from {name}")
    return outputs, {"occurrence":occurrence_meta,
                     "conditional":conditional_meta,
                     "mean_activity_probability":float(p.mean()),
                     "mean_conditional_units":float(mu.mean())}


def _pooled_score(loss: dict[str, dict[str,float]], denominator: float) -> dict[str,float]:
    if denominator <= 0:
        raise ValueError("No observed historical demand")
    return {name:(v["absolute"]+abs(v["signed"]))/denominator
            for name,v in loss.items()}


def select_phase0(phase0: pd.DataFrame, *, trees: int = 160,
                  occurrence_trees: int = 100, conditional_trees: int = 100) -> dict:
    observed = history_values(phase0)
    if observed.shape[1] < ORIGINS[-1]+HORIZON:
        raise ValueError("Last validation horizon not fully observed")
    pool = {name:{"absolute":0.,"signed":0.} for name in POLICIES}
    folds = {}
    heldout = {}
    denominator=0.
    for origin in ORIGINS:
        past=phase0.iloc[:, :len(KEYS)+origin].copy()
        control=incumbent(past,trees=trees)
        preds, metadata = candidates(past,control,occurrence_trees=occurrence_trees,
                                      conditional_trees=conditional_trees)
        actual=observed[:,origin:origin+HORIZON]
        fold_scores={name:_metric(pred,actual) for name,pred in preds.items()}
        folds[str(origin)]={"scores":fold_scores,"models":metadata}
        if origin==ORIGINS[-1]:
            heldout=fold_scores
        else:
            denominator+=float(actual.sum())
            for name,pred in preds.items():
                err=pred-actual
                pool[name]["absolute"]+=float(np.abs(err).sum())
                pool[name]["signed"]+=float(err.sum())
        print(f"VN1_HURDLE_FOLD={origin}",flush=True)
    training = _pooled_score(pool,denominator)
    winner = min(POLICIES,key=lambda name:(training[name],POLICIES.index(name)))
    accepted=(winner!="activity_p100" and
              training[winner] <= training["activity_p100"]-MIN_GAIN and
              heldout[winner] <= heldout["activity_p100"]-MIN_GAIN)
    return {"training_origins":list(ORIGINS[:-1]),
            "holdout_origin":ORIGINS[-1],
            "training_scores":training,"holdout_scores":heldout,
            "winner_on_training_only":winner,
            "selected":winner if accepted else "activity_p100",
            "accepted":bool(accepted),"minimum_gain":MIN_GAIN,"folds":folds}


def freeze(data_dir:Path,output_dir:Path,*,phase:int,trees:int=160,
           occurrence_trees:int=100,conditional_trees:int=100) -> dict:
    if phase not in (1,2):
        raise ValueError("Unknown phase")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Freeze directory must be empty")
    original=read_sales(data_dir/"Phase 0 - Sales.csv")
    selected=select_phase0(original,trees=trees,occurrence_trees=occurrence_trees,
                           conditional_trees=conditional_trees)
    history=read_history(phase,data_dir)
    base=incumbent(history,trees=trees)
    forecasts,details=candidates(history,base,occurrence_trees=occurrence_trees,
                                  conditional_trees=conditional_trees)
    forecasts["original_incumbent"]=base
    forecasts["phase0_selected"]=forecasts[selected["selected"]].copy()
    output_dir.mkdir(parents=True,exist_ok=True)
    dest=output_dir/"frozen.npz"
    np.savez_compressed(dest,**forecasts)
    future=list(pd.date_range(pd.Timestamp(history.columns[-1])+pd.Timedelta(weeks=1),
                              periods=HORIZON,freq="7D").strftime("%Y-%m-%d"))
    manifest={"phase":phase,"series":len(history),"horizon":HORIZON,
              "cutoff":str(history.columns[-1]),"future_dates":future,
              "forecast_names":list(forecasts),"selection":selected,
              "model_diagnostics":details,"frozen_sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),
              "fit_only_historical_observed_targets":True,
              "posthoc_public_phase_results_not_blind":True}
    (output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("VN1_HURDLE_FROZEN="+json.dumps({
        "phase":phase,"selected":selected["selected"],"hash":manifest["frozen_sha256"]
    }),flush=True)
    return manifest


def score(data_dir:Path,output_dir:Path)->dict:
    manifest=json.loads((output_dir/"manifest.json").read_text())
    frozen=output_dir/"frozen.npz"
    if hashlib.sha256(frozen.read_bytes()).hexdigest()!=manifest["frozen_sha256"]:
        raise ValueError("Frozen forecast hash mismatch")
    targets=read_sales(data_dir/f"Phase {manifest['phase']} - Sales.csv")
    hist=read_history(manifest["phase"],data_dir)
    if (len(targets)!=manifest["series"] or not targets[KEYS].equals(hist[KEYS]) or
        list(targets.columns[3:])!=manifest["future_dates"]):
        raise ValueError("Actuals not aligned with frozen predictions")
    y=targets.iloc[:,3:].to_numpy(dtype=np.float64)
    with np.load(frozen) as f:
        results={name:_metric(np.asarray(f[name],dtype=np.float64),y)
                 for name in manifest["forecast_names"]}
    receipt={"phase":manifest["phase"],"scores":results,
             "selected_on_phase0":manifest["selection"]["selected"],
             "accepted_by_historical_gate":manifest["selection"]["accepted"],
             "sha256":manifest["frozen_sha256"],"public_retrospective_not_blind":True}
    (output_dir/"scores.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("VN1_HURDLE_SCORES="+json.dumps(results,sort_keys=True),flush=True)
    return receipt


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("operation",choices=("historical","freeze","score"))
    ap.add_argument("--data-dir",type=Path,default=Path("."))
    ap.add_argument("--output-dir",type=Path,default=Path("results/vn1-hurdle"))
    ap.add_argument("--phase",type=int,choices=(1,2),default=2)
    ap.add_argument("--trees",type=int,default=160)
    ap.add_argument("--occurrence-trees",type=int,default=100)
    ap.add_argument("--conditional-trees",type=int,default=100)
    args=ap.parse_args()
    if args.operation=="historical":
        phase0=read_sales(args.data_dir/"Phase 0 - Sales.csv")
        output=select_phase0(phase0,trees=args.trees,
                             occurrence_trees=args.occurrence_trees,
                             conditional_trees=args.conditional_trees)
        args.output_dir.mkdir(parents=True,exist_ok=True)
        (args.output_dir/"phase0-historical.json").write_text(json.dumps(output,indent=2)+"\n")
        print("VN1_HURDLE_HISTORICAL="+json.dumps({k:output[k] for k in (
            "winner_on_training_only","selected","accepted","training_scores",
            "holdout_scores")},sort_keys=True),flush=True)
    elif args.operation=="freeze":
        freeze(args.data_dir,args.output_dir,phase=args.phase,trees=args.trees,
               occurrence_trees=args.occurrence_trees,
               conditional_trees=args.conditional_trees)
    else:
        score(args.data_dir,args.output_dir)


if __name__=="__main__":
    main()
