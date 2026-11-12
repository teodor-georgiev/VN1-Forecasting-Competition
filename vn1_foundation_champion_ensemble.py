"""Leakage-controlled blend of SAVED foundation forecasts with VN1 champion.

Only uses previously generated product-keyed CSV predictions, so optimizing
weights never reruns Chronos/TimesFM. Phase 1 may select a conservative simplex
weight, Phase 2 is frozen with no target reads, then separately scored.
The current best champion is passed explicitly; it is never silently replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from vn1_from_scratch import HORIZON, KEYS, read_sales

MAX_MODELS = 4
MAX_FOUNDATION_WEIGHT = 0.30
STEP = 0.05
MIN_GAIN = 0.001


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_json(path: Path, receipt: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def aligned_forecast_frame(path: Path, reference: pd.DataFrame) -> pd.DataFrame:
    """Require an exact series key/order and complete 13 future week alignment."""
    path = Path(path)
    frame = pd.read_csv(path)
    if list(frame.columns[:len(KEYS)]) != list(KEYS):
        raise ValueError(f"Forecast key columns differ: {path}")
    if len(frame) != len(reference) or frame.duplicated(KEYS).any():
        raise ValueError(f"Forecast key count or duplicate keys: {path}")
    if not frame[KEYS].equals(reference[KEYS]):
        raise ValueError(f"Forecast key or series ordering mismatch: {path}")
    if list(frame.columns[len(KEYS):]) != list(reference.columns[len(KEYS):]):
        raise ValueError(f"Forecast dates or horizon mismatch: {path}")
    if frame.shape[1] != len(KEYS) + HORIZON:
        raise ValueError("Forecast must contain exactly 13 weekly horizons")
    values = frame.iloc[:,len(KEYS):].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"Nonfinite or negative forecast: {path}")
    return frame


def load_aligned(path: Path, reference: pd.DataFrame) -> np.ndarray:
    return aligned_forecast_frame(path, reference).iloc[:,len(KEYS):].to_numpy(
        dtype=np.float64)


def metric(pred: np.ndarray, truth: np.ndarray) -> dict:
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim != 2 or pred.shape[1] != HORIZON:
        raise ValueError("Metric shape mismatch")
    if (not np.isfinite(pred).all() or not np.isfinite(truth).all()
            or (pred < 0).any() or (truth < 0).any() or truth.sum() <= 0):
        raise ValueError("Invalid forecast/target data")
    error = pred-truth
    total = float(truth.sum())
    absolute = float(np.abs(error).sum()/total)
    signed = float(error.sum()/total)
    return {"score": absolute+abs(signed), "absolute": absolute,
            "signed": signed, "bias_penalty": abs(signed)}


def model_grid(n: int, *, step: float = STEP,
               max_foundation: float = MAX_FOUNDATION_WEIGHT):
    """Fixed 5%-increment simplex grid, champion occupies >=70% by default."""
    if not 1 <= n <= MAX_MODELS:
        raise ValueError("Must supply 1 to 4 foundation model forecasts")
    if step <= 0 or max_foundation < 0 or max_foundation > 1:
        raise ValueError("Invalid weight grid")
    steps = int(round(1 / step))
    cap = int(round(max_foundation / step))
    if abs(steps*step-1) > 1e-8 or abs(cap*step-max_foundation) > 1e-8:
        raise ValueError("Grid must divide full and maximum weight exactly")
    for parts in itertools.product(range(cap + 1), repeat=n):
        total = sum(parts)
        if total <= cap:
            yield (round(1-total*step, 10),
                   *(round(p*step, 10) for p in parts))


def _models(foundation: dict[str, Path]) -> tuple[str, ...]:
    if not foundation or len(foundation)>MAX_MODELS:
        raise ValueError("Supply 1 to 4 named foundation models")
    for model in foundation:
        if not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", model):
            raise ValueError("Model identifiers must be simple lowercase names")
        if model=="champion":
            raise ValueError("Champion must be a separate source")
    return tuple(sorted(foundation))


def optimize(target_phase1: Path, champion_phase1: Path,
             foundation_phase1: dict[str,Path], output: Path,
             *, min_gain: float = MIN_GAIN) -> dict:
    """Select weight only on Phase 1 labels. NEVER inspect Phase 2 targets."""
    names = _models(foundation_phase1)
    target = read_sales(Path(target_phase1))
    if target.shape[1] != len(KEYS)+HORIZON:
        raise ValueError("Phase 1 target requires 13 dates")
    truth = target.iloc[:,len(KEYS):].to_numpy(dtype=np.float64)
    champion = load_aligned(champion_phase1,target)
    models = [load_aligned(foundation_phase1[name],target) for name in names]
    incumbent = metric(champion,truth)
    champion_best_score = incumbent["score"]
    selected_weights=(1.,*(0. for _ in names))
    best = incumbent
    best_grid = None
    for weights in model_grid(len(names)):
        prediction = weights[0]*champion
        for w,model in zip(weights[1:],models):
            if w:
                prediction = prediction + w*model
        result = metric(prediction,truth)
        # Strict best; tie stays with champion-only first as generated by grid.
        if best_grid is None or result["score"] < best_grid["score"]-1e-12:
            best_grid = result
            best_grid_weights = weights
    if best_grid["score"] <= champion_best_score-min_gain:
        selected_weights=best_grid_weights
        best=best_grid
    selected=dict(zip(("champion",*names),selected_weights))
    receipt={
        "selection_phase":1,
        "method":"chronological_phase1_conservative_5pct_simplex",
        "full_catalogue_rows":int(len(target)),
        "horizon":HORIZON,
        "model_order":list(names),
        "step":STEP,
        "max_total_foundation_weight":MAX_FOUNDATION_WEIGHT,
        "min_gain":min_gain,
        "champion_score":champion_best_score,
        "phase1_candidate_best_score":best_grid["score"],
        "phase1_candidate_best_weights":dict(
            zip(("champion",*names),best_grid_weights)),
        "selected_score":best["score"],
        "selected_weights":selected,
        "accepted":selected["champion"] < 1,
        "source_sha256":{
            "champion":digest(champion_phase1),
            **{name:digest(foundation_phase1[name]) for name in names},
        },
        "target_phase1_sha256":digest(target_phase1),
        "phase2_labels_read":False,
        "retrospective_selection_not_blind":True,
        "pretraining_overlap_unknown":True,
        "caution":"Phase 1 has been inspected in earlier research; do not treat a Phase 2 comparison as an unseen blind test.",
    }
    safe_json(Path(output),receipt)
    return receipt


def _phase2_reference(history: pd.DataFrame) -> pd.DataFrame:
    if list(history.columns[:len(KEYS)])!=list(KEYS) or history.duplicated(KEYS).any():
        raise ValueError("Invalid historical keys")
    if len(history.columns)<=len(KEYS):
        raise ValueError("Historical weekly dates missing")
    last=pd.Timestamp(history.columns[-1])
    dates=pd.date_range(last+pd.Timedelta(weeks=1),periods=HORIZON,freq="7D")
    reference=history[KEYS].copy()
    for d in dates:
        reference[d.strftime("%Y-%m-%d")]=0.0
    return reference


def freeze(history_phase1: Path, champion_phase2: Path,
           foundation_phase2: dict[str,Path], weights_path: Path,
           output_dir: Path) -> dict:
    """Freeze Phase 2 without loading ANY Phase 2 targets or selecting weights."""
    weightfile=Path(weights_path)
    prior=json.loads(weightfile.read_text(encoding="utf-8"))
    names=_models(foundation_phase2)
    if prior.get("selection_phase")!=1 or names!=tuple(prior.get("model_order", [])):
        raise ValueError("Phase 2 models differ from Phase 1 selected models")
    weights=prior["selected_weights"]
    if (set(weights)!={"champion",*names}
            or abs(sum(weights.values())-1.)>1e-10
            or min(weights.values()) < 0
            or weights["champion"] < 1-MAX_FOUNDATION_WEIGHT-1e-10):
        raise ValueError("Invalid preselected simplex weights")
    history=read_sales(Path(history_phase1))
    reference=_phase2_reference(history)
    champion=load_aligned(champion_phase2,reference)
    mixed=weights["champion"]*champion
    for name in names:
        x=load_aligned(foundation_phase2[name],reference)
        if weights[name]:
            mixed=mixed+weights[name]*x
    if not np.isfinite(mixed).all() or (mixed < 0).any():
        raise ValueError("Mixed forecasts invalid")
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    frozen=output_dir/"ensemble-phase2.csv"
    existing=output_dir/"champion-phase2.csv"
    manifest=output_dir/"frozen-manifest.json"
    if any(p.exists() for p in (frozen,existing,manifest)):
        raise FileExistsError("Frozen output exists; never overwrite a Phase 2 forecast")
    out=reference.copy()
    out.iloc[:,len(KEYS):]=mixed
    incumbent=reference.copy()
    incumbent.iloc[:,len(KEYS):]=champion
    out.to_csv(frozen,index=False)
    incumbent.to_csv(existing,index=False)
    receipt={
        "selection_phase":1,"frozen_phase":2,
        "weights_sha256":digest(weightfile),
        "history_phase1_sha256":digest(history_phase1),
        "source_sha256":{
            "champion":digest(champion_phase2),
            **{name:digest(foundation_phase2[name]) for name in names}},
        "output_sha256":{"ensemble":digest(frozen),"champion":digest(existing)},
        "product_count":len(history),"horizon":HORIZON,
        "selected_weights":weights,
        "model_order":list(names),
        "phase2_labels_read":False,
        "retrospective_not_blind":True,
    }
    safe_json(manifest,receipt)
    return receipt


def score_frozen(output_dir: Path, target_phase2: Path) -> dict:
    """Validate immutable frozen Phase 2 forecast hashes before scoring labels."""
    folder=Path(output_dir)
    manifest=json.loads((folder/"frozen-manifest.json").read_text(encoding="utf-8"))
    en=folder/"ensemble-phase2.csv"
    ch=folder/"champion-phase2.csv"
    if digest(en)!=manifest["output_sha256"]["ensemble"] or (
            digest(ch)!=manifest["output_sha256"]["champion"]):
        raise ValueError("Frozen forecast SHA-256 mismatch")
    objective=read_sales(Path(target_phase2))
    incumbent=load_aligned(ch,objective)
    mixed=load_aligned(en,objective)
    truth=objective.iloc[:,len(KEYS):].to_numpy(dtype=np.float64)
    scores={
        "rows":len(truth),"horizon":HORIZON,
        "champion":metric(incumbent,truth),
        "ensemble":metric(mixed,truth),
        "improvement":metric(incumbent,truth)["score"]-metric(mixed,truth)["score"],
        "retrospective_not_blind":True,
        "selection_phase":1,
        "score_phase":2,
    }
    safe_json(folder/"score-phase2.json",scores)
    return scores


def _arguments(raw: list[str]) -> dict[str, Path]:
    result={}
    for spec in raw:
        if "=" not in spec:
            raise ValueError("Use model=path for each --foundation")
        model,path=spec.split("=",1)
        if model in result:
            raise ValueError("Duplicate model")
        result[model]=Path(path)
    _models(result)
    return result


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest="operation",required=True)
    o=sub.add_parser("optimize",help="Choose weights from Phase 1; no Phase 2 access")
    o.add_argument("--phase1-target",type=Path,required=True)
    o.add_argument("--champion",type=Path,required=True)
    o.add_argument("--foundation",action="append",required=True)
    o.add_argument("--output",type=Path,required=True)
    f=sub.add_parser("freeze",help="Produce immutable Phase 2 submission")
    f.add_argument("--history",type=Path,required=True,
                   help="Phase 1 sales CSV; used for product IDs and forecast dates")
    f.add_argument("--champion",type=Path,required=True)
    f.add_argument("--foundation",action="append",required=True)
    f.add_argument("--weights",type=Path,required=True)
    f.add_argument("--output-dir",type=Path,required=True)
    s=sub.add_parser("score",help="Score frozen prediction, never select on Phase 2")
    s.add_argument("--output-dir",type=Path,required=True)
    s.add_argument("--phase2-target",type=Path,required=True)
    args=ap.parse_args()
    if args.operation=="optimize":
        result=optimize(args.phase1_target,args.champion,_arguments(args.foundation),args.output)
    elif args.operation=="freeze":
        result=freeze(args.history,args.champion,_arguments(args.foundation),
                      args.weights,args.output_dir)
    else:
        result=score_frozen(args.output_dir,args.phase2_target)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
