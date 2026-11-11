"""Strict freeze-first retrospective evaluator for VN1's five experimental levers.

Uses the Phase 0 historical selection receipt to freeze Phase 1 or Phase 2
forecasts before a separate score command loads labels. Private prices and
truth stay in an authorized runtime, not in the public GitHub artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vn1_from_scratch import HORIZON, KEYS, read_sales
from vn1_lgbm_ensemble import read_history
from vn1_amsi_zero_rule_ablation import incumbent
from vn1_five_levers import (
    POLICIES, FAMILIES, MIN_GAIN, MODEL_VARIANTS, candidates, read_price, _metric,
    _train_predict_variants, _hierarchy, _trim_leading_zeros,
)
from vn1_learned_activity import allocate_weekly, predict_activity


def _digest(path:Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected(report:Path,prices:bool) -> dict:
    historic=json.loads(report.read_text())
    if (historic.get("origins_train")!=[105,118,131,144]
            or historic.get("holdout_origin")!=157
            or historic.get("minimum_absolute_gain")!=MIN_GAIN
            or historic.get("prices_loaded") != prices
            or historic.get("families")!=FAMILIES
            or historic.get("selected") not in POLICIES):
        raise ValueError("Unverified or mismatched Phase 0 selection receipt")
    winner=historic["winner_train_only"]
    train=historic["training_scores"]
    holdout=historic["holdout_scores"]
    expected_accept=(
        winner!="learned_activity" and
        train[winner]<=train["learned_activity"]-MIN_GAIN and
        holdout[winner]<=holdout["learned_activity"]-MIN_GAIN
    )
    if historic.get("accepted")!=expected_accept:
        raise ValueError("Phase 0 acceptance gate not reproducible")
    if historic["selected"]!=(winner if expected_accept else "learned_activity"):
        raise ValueError("Selection does not match gate")
    return historic


def _prices(data_dir:Path, phase:int,
            price0:Path|None,price1:Path|None)->pd.DataFrame|None:
    if price0 is None:
        if price1 is not None:raise ValueError("Price1 without Price0")
        return None
    p0=read_sales(data_dir/"Phase 0 - Sales.csv")
    a=read_price(price0,p0)
    if phase==1:return a
    if price1 is None:raise ValueError("Phase 2 price history requires Phase1 Price")
    p1=read_sales(data_dir/"Phase 1 - Sales.csv")
    b=read_price(price1,p1)
    if not a[KEYS].equals(b[KEYS]):
        raise ValueError("Misordered price histories")
    dates0=pd.to_datetime(a.columns[3:])
    dates1=pd.to_datetime(b.columns[3:])
    if dates1[0]!=dates0[-1]+pd.Timedelta(weeks=1):
        raise ValueError("Price chronology not contiguous")
    return pd.concat([a,b.iloc[:,3:]],axis=1)


def freeze(data_dir:Path,out_dir:Path,report:Path,*,phase:int,
           price0:Path|None=None,price1:Path|None=None,trees:int=80,
           activity_trees:int=100,baseline_trees:int=160,
           selected_only:bool=False)->dict:
    if phase not in (1,2):raise ValueError("Bad phase")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("Freeze directory must be new or empty")
    chosen=_selected(report,price0 is not None)
    history=read_history(phase,data_dir)
    prices=_prices(data_dir,phase,price0,price1)
    control=incumbent(history,trees=baseline_trees)
    if selected_only:
        probabilities, probability_meta = predict_activity(history,estimators=activity_trees)
        base=allocate_weekly(control,probabilities)
        forecast={"learned_activity":base}
        selected=chosen["selected"]
        if selected in MODEL_VARIANTS:
            forecast.update(_train_predict_variants(history,base,prices,
                            trees=trees,only={selected}))
        elif selected=="hierarchy_client_warehouse":
            forecast[selected]=_hierarchy(history,base)
        elif selected=="trim_prelaunch_zeros":
            forecast[selected]=_trim_leading_zeros(history,base)
        elif selected!="learned_activity":
            raise ValueError("Unknown selected policy")
        meta={"policies":list(forecast),"prices_available":prices is not None,
              "activity":probability_meta}
    else:
        forecast,meta=candidates(history,control,prices=prices,
                                 trees=trees,activity_trees=activity_trees)
    if chosen["selected"] not in forecast:
        raise AssertionError("Selected model missing from candidate set")
    forecast["historical_selected"]=forecast[chosen["selected"]].copy()
    out_dir.mkdir(parents=True,exist_ok=True)
    archive=out_dir/"frozen.npz"
    np.savez_compressed(archive,**forecast)
    future=list(pd.date_range(pd.Timestamp(history.columns[-1])+pd.Timedelta(weeks=1),
                              periods=HORIZON,freq="7D").strftime("%Y-%m-%d"))
    manifest={
        "phase":phase,"selection":chosen["selected"],
        "accepted":chosen["accepted"],"series":len(history),
        "history_cutoff":str(history.columns[-1]),
        "dates":future,"candidate_names":list(forecast),
        "forecast_sha256":_digest(archive),"phase0_receipt_sha256":_digest(report),
        "phase0_sales_sha256":_digest(data_dir/"Phase 0 - Sales.csv"),
        "phase1_sales_sha256":_digest(data_dir/"Phase 1 - Sales.csv") if phase==2 else None,
        "price0_sha256":_digest(price0) if price0 is not None else None,
        "price1_sha256":_digest(price1) if price1 is not None else None,
        "price_data_embedded_in_archive":False,
        "only_historically_selected_policy_fit":selected_only,
        "phase_target_labels_not_loaded_in_freeze":True,
        "caveat":"Previously analyzed retrospective phases, not independent blind validation.",
        "fit_summary":{"policies":meta["policies"],"prices_available":meta["prices_available"]}
    }
    (out_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("VN1_FIVE_FREEZE="+json.dumps({
        "phase":phase,"selected":manifest["selection"],"sha256":manifest["forecast_sha256"],
        "prices":meta["prices_available"]},sort_keys=True),flush=True)
    return manifest


def score(data_dir:Path,out_dir:Path,*,phase:int,truth:Path)->dict:
    manifest=json.loads((out_dir/"manifest.json").read_text())
    if manifest["phase"]!=phase:raise ValueError("Wrong phase")
    bundle=out_dir/"frozen.npz"
    if _digest(bundle)!=manifest["forecast_sha256"]:
        raise ValueError("Frozen forecast has been modified")
    history=read_history(phase,data_dir)
    observed=read_sales(truth)
    if (len(observed)!=manifest["series"] or not observed[KEYS].equals(history[KEYS])
            or list(observed.columns[3:])!=manifest["dates"]):
        raise ValueError("Target IDs or forecast weeks misaligned")
    y=observed.iloc[:,3:].to_numpy(dtype=np.float64)
    with np.load(bundle) as archive:
        results={name:_metric(np.asarray(archive[name],dtype=np.float64),y)
                 for name in manifest["candidate_names"]}
    answer={
        "phase":phase,"series":len(y),"horizon":HORIZON,"scores":results,
        "phase0_selected":manifest["selection"],
        "historical_accepted":manifest["accepted"],
        "frozen_sha256":manifest["forecast_sha256"],
        "retrospective_not_blind":True,
    }
    (out_dir/"score.json").write_text(json.dumps(answer,indent=2)+"\n")
    print("VN1_FIVE_SCORE="+json.dumps(answer,sort_keys=True),flush=True)
    return answer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation",choices=("freeze","score"))
    p.add_argument("--data-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--report",type=Path)
    p.add_argument("--phase",type=int,choices=(1,2),required=True)
    p.add_argument("--price0",type=Path)
    p.add_argument("--price1",type=Path)
    p.add_argument("--truth",type=Path)
    p.add_argument("--trees",type=int,default=80)
    p.add_argument("--activity-trees",type=int,default=100)
    p.add_argument("--baseline-trees",type=int,default=160)
    p.add_argument("--selected-only",action="store_true",
                   help="Fit only the preselected Phase 0 candidate plus control")
    args=p.parse_args()
    if args.operation=="freeze":
        if args.report is None:p.error("freeze requires historical --report")
        freeze(args.data_dir,args.out,args.report,phase=args.phase,
               price0=args.price0,price1=args.price1,trees=args.trees,
               activity_trees=args.activity_trees,baseline_trees=args.baseline_trees,
               selected_only=args.selected_only)
    else:
        if args.truth is None:p.error("score requires --truth")
        score(args.data_dir,args.out,phase=args.phase,truth=args.truth)


if __name__=="__main__":
    main()
