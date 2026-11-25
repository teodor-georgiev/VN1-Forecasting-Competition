"""Saved Chronos calibration: preserve incumbent volume, route by past regime.

Phase 1 selection is retrospective. Freeze Phase 2 before scoring it, using
exact keyed forecast alignment and the existing archive scoring protocol.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import vn1_foundation_champion_ensemble as saved
import vn1_champion_residual_oof as base
from vn1_from_scratch import KEYS, HORIZON, read_sales

POLICIES = ("incumbent", "raw30", "calibrated15", "calibrated30", "regime_calibrated")


def _regimes(history):
    y = base._history_matrix(history)
    return np.where(base._active_mask(y), base._regimes(y), 3)


def _calibrated(champion, foundation, weights, regime):
    """Match blend to champion total within each regime, across 13 weeks."""
    out = champion.copy()
    for k in range(4):
        m = regime == k
        w = float(weights[k])
        if not 0 <= w <= .30:
            raise ValueError("Invalid weight")
        if not m.any() or not w:
            continue
        c, f = champion[m], foundation[m]
        total, raw_total = c.sum(), f.sum()
        if total <= 0 or raw_total <= 0:
            continue
        # No target-informed scaling: anchor to the incumbent's forecast.
        calibrated = (1-w)*c + w*f*(total/raw_total)
        if not np.isfinite(calibrated).all():
            raise ValueError("Invalid calibrated blend")
        out[m] = calibrated
    if not np.allclose(out.sum(), champion.sum(), rtol=1e-10, atol=1e-6):
        raise ArithmeticError("Total demand changed")
    return out


def _predict(champion, foundation, regime, policy, weights):
    if policy == "incumbent":
        return champion.copy()
    if policy == "raw30":
        return .70*champion + .30*foundation
    return _calibrated(champion, foundation, weights, regime)


def select(history, target, champion, foundation):
    """Choose conservative weights on Phase 1 only; no Phase 2 inputs."""
    reference = saved._phase2_reference(history)
    if not reference[KEYS].equals(target[KEYS]) or list(reference.columns) != list(target.columns):
        raise ValueError("History and target alignment mismatch")
    y = target.iloc[:,len(KEYS):].to_numpy(dtype=float)
    regime = _regimes(history)
    weights = {"incumbent":[0.]*4,"raw30":[.30]*4,
               "calibrated15":[.15]*4,"calibrated30":[.30]*4}
    route = []
    for k in range(4):
        mask = regime == k
        if mask.sum() < 100 or y[mask].sum() <= 0:
            route.append(0.)
            continue
        losses = []
        for w in (0.,.15,.30):
            pred = _calibrated(champion,foundation,[w]*4,regime)
            losses.append(np.abs(pred[mask]-y[mask]).sum())
        winner = int(np.argmin(losses))
        route.append((0.,.15,.30)[winner] if losses[winner]<.99*losses[0] else 0.)
    weights["regime_calibrated"] = route
    scores = {name:saved.metric(_predict(champion,foundation,regime,name,weights[name]),y)
              for name in POLICIES}
    best = min(POLICIES,key=lambda name:scores[name]["score"])
    selected = best if scores[best]["score"] <= scores["incumbent"]["score"]-.001 else "incumbent"
    return {"selection_phase":1,"selected":selected,"weights":weights[selected],
            "scores":scores,"candidate_weights":weights,"rows":len(y),
            "retrospective_not_blind":True,"phase2_labels_read":False}


def optimize(history_path, target_path, champion_path, foundation_path, output):
    if output.exists():
        raise FileExistsError("Never overwrite selection")
    h, y = read_sales(history_path), read_sales(target_path)
    c, f = saved.load_aligned(champion_path,y), saved.load_aligned(foundation_path,y)
    receipt = select(h,y,c,f)
    receipt["source_sha256"] = {k:saved.digest(p) for k,p in
        (("history",history_path),("target",target_path),("champion",champion_path),("chronos",foundation_path))}
    saved.safe_json(output,receipt)
    return receipt


def freeze(history_path, champion_path, foundation_path, selection_path, output):
    prior = json.loads(selection_path.read_text())
    policy, weights = prior["selected"], prior["weights"]
    if prior.get("selection_phase") != 1 or policy not in POLICIES or len(weights)!=4:
        raise ValueError("Invalid selection")
    if weights != prior["candidate_weights"][policy] or any(w not in (0.,.15,.30) for w in weights):
        raise ValueError("Invalid selected weights")
    best = min(POLICIES,key=lambda name:prior["scores"][name]["score"])
    expected = best if prior["scores"][best]["score"] <= prior["scores"]["incumbent"]["score"]-.001 else "incumbent"
    if policy != expected:
        raise ValueError("Selection gate mismatch")
    h = read_sales(history_path)
    ref = saved._phase2_reference(h)
    c,f = saved.load_aligned(champion_path,ref),saved.load_aligned(foundation_path,ref)
    mix = _predict(c,f,_regimes(h),policy,weights)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Frozen archive already exists")
    output.mkdir(parents=True,exist_ok=True)
    en,ch = output/"ensemble-phase2.csv",output/"champion-phase2.csv"
    for path, values in ((en,mix),(ch,c)):
        frame=ref.copy()
        frame.iloc[:,len(KEYS):]=values
        frame.to_csv(path,index=False)
    receipt={"selection_phase":1,"frozen_phase":2,"selected":policy,"weights":weights,
             "output_sha256":{"ensemble":saved.digest(en),"champion":saved.digest(ch)},
             "selection_sha256":saved.digest(selection_path),"history_sha256":saved.digest(history_path),
             "source_sha256":{"champion":saved.digest(champion_path),"chronos":saved.digest(foundation_path)},
             "product_count":len(h),"horizon":HORIZON,"phase2_labels_read":False,
             "retrospective_not_blind":True}
    saved.safe_json(output/"frozen-manifest.json",receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="op",required=True)
    for op in ("optimize","freeze"):
        s=sub.add_parser(op)
        for field in ("history","champion","foundation","output"):
            s.add_argument("--"+field,type=Path,required=True)
        s.add_argument("--"+("target" if op=="optimize" else "selection"),type=Path,required=True)
    s=sub.add_parser("score")
    s.add_argument("--output",type=Path,required=True)
    s.add_argument("--target",type=Path,required=True)
    a=p.parse_args()
    if a.op=="optimize":
        result=optimize(a.history,a.target,a.champion,a.foundation,a.output)
    elif a.op=="freeze":
        result=freeze(a.history,a.champion,a.foundation,a.selection,a.output)
    else:
        result=saved.score_frozen(a.output,a.target)
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
