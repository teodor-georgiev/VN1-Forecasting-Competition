"""Exploratory VN1 soft dormancy and intermittent-demand allocation benchmark.

Forecasts/acceptance are frozen from Phase 0 history. Retrospective Phase 1
and Phase 2 actuals are opened *only* in a separate scoring process. This work
is post-hoc motivated by Phase 2 research and is not independent blind testing.
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
from vn1_amsi_zero_rule_ablation import incumbent, rules, _metric
from vn1_adaptive import intermittent_forecasts

ORIGINS = (105, 118, 131, 144, 157)  # Last validation target ends at Phase 0 week 170.
MIN_GAIN = 0.001  # The existing historical acceptance gate is unchanged.
CANDIDATES = (
    "incumbent",
    "quiet26_soft50_global",
    "quiet26_soft75_global",
    "quiet26_hard_global",
    "quiet26_hard_weekly",
    "quiet26_hard_recent",
    "quiet26_hard_client",
    "quiet13_soft50_global",
    "quiet13_hard_weekly",
    "quiet52_hard_weekly",
    "tsb_10_volume_preserving",
    "tsb_25_volume_preserving",
    "tsb10_plus_quiet26_weekly",
)


def _observed(history: pd.DataFrame) -> np.ndarray:
    if list(history.columns[:len(KEYS)]) != list(KEYS) or history.duplicated(KEYS).any():
        raise ValueError("Misaligned VN1 keys")
    y = history.iloc[:, len(KEYS):].to_numpy(dtype=np.float64)
    if y.ndim != 2 or y.shape[1] < 65 or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Invalid historical demand")
    return y


def redistribute(
    forecast: np.ndarray, dormant: np.ndarray, strength: float = 1.,
    method: str = "global", history: pd.DataFrame | None = None,
) -> np.ndarray:
    """Shrink dormant forecast mass and reassign it to active series.

    The total of the forecast remains EXACTLY unchanged up to floating-point
    precision. Never distribute mass back onto suppressed products.
    """
    base = np.asarray(forecast, dtype=np.float64)
    mask = np.asarray(dormant, dtype=bool)
    if base.ndim != 2 or base.shape[1] != HORIZON or mask.shape != (base.shape[0],):
        raise ValueError("Wrong shape")
    if not 0 <= strength <= 1 or not np.isfinite(base).all() or (base < 0).any():
        raise ValueError("Invalid forecast or shrinkage")
    if method not in {"global", "weekly", "recent", "client"}:
        raise ValueError("Unrecognized allocation")
    if method in {"recent", "client"} and history is None:
        raise ValueError("History required")
    if history is not None and len(history) != len(base):
        raise ValueError("History length mismatch")
    if not mask.any() or strength == 0:
        return base.copy()
    recipients = ~mask
    if not recipients.any() or base[recipients].sum() == 0:
        return base.copy()
    result = base.copy()
    result[mask] *= (1. - strength)
    removed = strength * base[mask].sum()
    if removed == 0:
        return result

    if method == "weekly":
        original = base[recipients]
        per_week = strength * base[mask].sum(axis=0)
        divisor = original.sum(axis=0)
        weekly_delta = np.zeros_like(original)
        positive = divisor > 0
        weekly_delta[:, positive] = original[:, positive] * (
            per_week[positive] / divisor[positive])
        result[recipients] += weekly_delta
        leftover = per_week[~positive].sum()
        if leftover:
            result[recipients] += leftover * original / original.sum()
    elif method == "recent":
        y = _observed(history)
        recent = y[recipients, -13:].mean(axis=1)
        weight = 0.5 * base[recipients] + 0.5 * recent[:, None]
        divisor = weight.sum()
        if divisor == 0:
            return base.copy()
        result[recipients] += removed * weight / divisor
    elif method == "client":
        codes, _ = pd.factorize(history["Client"].to_numpy(), sort=True)
        stranded = 0.0
        for client in np.unique(codes[mask]):
            removed_client = mask & (codes == client)
            available = recipients & (codes == client)
            mass = float(strength * base[removed_client].sum())
            divisor = float(base[available].sum())
            if divisor > 0:
                result[available] += mass * base[available] / divisor
            else:
                stranded += mass
        if stranded > 0:
            result[recipients] += stranded * base[recipients] / base[recipients].sum()
    else:
        weight = base[recipients]
        result[recipients] += removed * weight / weight.sum()
    if not np.isfinite(result).all() or (result < 0).any():
        raise ArithmeticError("Invalid redistributed predictions")
    if not np.isclose(result.sum(), base.sum(), rtol=1e-10, atol=1e-5):
        raise ArithmeticError("Allocation changed total forecast, invalid bias control")
    return result


def _volume_preserving_mix(base: np.ndarray, other: np.ndarray, weight: float) -> np.ndarray:
    if base.shape != other.shape or not np.isfinite(other).all() or (other < 0).any():
        raise ValueError("Invalid specialist forecast")
    mixed = (1. - weight) * base + weight * other
    if mixed.sum() <= 0 or base.sum() <= 0:
        return base.copy()
    return mixed * (base.sum() / mixed.sum())


def candidate_forecasts(history: pd.DataFrame, base: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    y = _observed(history)
    existing = rules(history)
    mask26 = existing["inactive26_annual_zero"]
    mask52 = existing["inactive52"]
    mask13 = (y[:, -13:].sum(axis=1) == 0) & (y[:, -52:-39].sum(axis=1) == 0)
    _, tsb = intermittent_forecasts(history, alpha=0.1, beta=0.1)
    if not isinstance(base, np.ndarray):
        base = np.asarray(base, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    if base.shape != (len(y), HORIZON):
        raise ValueError("Incorrect baseline dimensions")
    tsb10 = _volume_preserving_mix(base, tsb, 0.10)
    tsb25 = _volume_preserving_mix(base, tsb, 0.25)
    out = {
        "incumbent": base.copy(),
        "quiet26_soft50_global": redistribute(base,mask26,0.50),
        "quiet26_soft75_global": redistribute(base,mask26,0.75),
        "quiet26_hard_global": redistribute(base,mask26,1.00),
        "quiet26_hard_weekly": redistribute(base,mask26,1.00,"weekly"),
        "quiet26_hard_recent": redistribute(base,mask26,1.00,"recent",history),
        "quiet26_hard_client": redistribute(base,mask26,1.00,"client",history),
        "quiet13_soft50_global": redistribute(base,mask13,0.50),
        "quiet13_hard_weekly": redistribute(base,mask13,1.00,"weekly"),
        "quiet52_hard_weekly": redistribute(base,mask52,1.00,"weekly"),
        "tsb_10_volume_preserving": tsb10,
        "tsb_25_volume_preserving": tsb25,
        "tsb10_plus_quiet26_weekly": redistribute(tsb10,mask26,1.00,"weekly"),
    }
    if tuple(out) != CANDIDATES:
        raise AssertionError("Unexpected candidate ordering")
    for name, values in out.items():
        if not np.isfinite(values).all() or (values < 0).any() or values.shape != base.shape:
            raise ValueError(f"Invalid output for {name}")
    return out, {
        "quiet26_annual":int(mask26.sum()),
        "quiet52":int(mask52.sum()),
        "quiet13_annual":int(mask13.sum()),
    }


def select_on_phase0(phase0: pd.DataFrame, trees: int) -> dict:
    y = _observed(phase0)
    if len(y.T) < ORIGINS[-1] + HORIZON:
        raise ValueError("Phase 0 does not contain validation targets")
    aggregates = {k: {"l1": 0.0, "signed": 0.0} for k in CANDIDATES}
    denominator = 0.0
    holdout = {}
    fold_scores = {}
    masks_by_origin = {}
    for origin in ORIGINS:
        past = phase0.iloc[:, :len(KEYS) + origin].copy()
        base = incumbent(past, trees=trees)
        preds, counts = candidate_forecasts(past,base)
        actual = y[:, origin:origin + HORIZON]
        masks_by_origin[str(origin)] = counts
        fold_scores[str(origin)] = {name:_metric(forecast,actual) for name,forecast in preds.items()}
        if origin != ORIGINS[-1]:
            denominator += actual.sum()
            for name,forecast in preds.items():
                delta = forecast - actual
                aggregates[name]["l1"] += float(np.abs(delta).sum())
                aggregates[name]["signed"] += float(delta.sum())
        else:
            holdout = fold_scores[str(origin)]
        print("VN1_SOFT_DORMANCY_FOLD="+str(origin),flush=True)
    if denominator <= 0:
        raise ValueError("Historical targets contain no sales")
    train_scores = {k:(v["l1"]+abs(v["signed"]))/denominator for k,v in aggregates.items()}
    winner = min(CANDIDATES, key=lambda k:(train_scores[k], CANDIDATES.index(k)))
    accepted = (winner != "incumbent" and
                train_scores[winner] <= train_scores["incumbent"] - MIN_GAIN and
                holdout[winner] <= holdout["incumbent"] - MIN_GAIN)
    return {
        "origins_train":list(ORIGINS[:-1]),
        "origin_holdout":ORIGINS[-1],
        "scores_train":train_scores,
        "scores_holdout":holdout,
        "per_origin_scores":fold_scores,
        "mask_counts":masks_by_origin,
        "winner_train_only":winner,
        "selected": winner if accepted else "incumbent",
        "accepted":bool(accepted),
        "minimum_absolute_gain":MIN_GAIN,
    }


def freeze(data_dir:Path, output_dir:Path, phase:int, trees:int=160) -> dict:
    if phase not in (1,2):
        raise ValueError("Phase must be 1 or 2")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Do not overwrite an existing frozen experiment")
    phase0 = read_sales(data_dir/"Phase 0 - Sales.csv")
    selection = select_on_phase0(phase0,trees)
    history = read_history(phase,data_dir)
    base = incumbent(history,trees=trees)
    forecasts, masks = candidate_forecasts(history,base)
    forecasts["phase0_selected"] = forecasts[selection["selected"]].copy()
    output_dir.mkdir(parents=True,exist_ok=True)
    dest = output_dir/"forecast-frozen.npz"
    np.savez_compressed(dest,**forecasts) # float64 so exact score matches original.
    date_columns=list(pd.date_range(pd.to_datetime(history.columns[-1])+pd.Timedelta(weeks=1),
                                    periods=HORIZON,freq="7D").strftime("%Y-%m-%d"))
    manifest = {
        "phase":phase, "n_series":len(history),"horizon":HORIZON,
        "prediction_dates":date_columns,"forecast_candidates":list(forecasts),
        "selection":selection,"mask_counts_phase":masks,
        "data_sources_allowed_for_selection":"Phase 0 sales and previous phase sales only",
        "phase_truth_was_not_loaded_in_freeze":True,
        "blob_sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),
        "disclosure":"Post-hoc Phase 2 research hypothesis; Phase 2 is NOT blind."
    }
    (output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("VN1_SOFT_DORMANCY_FROZEN="+json.dumps({
        "phase":phase,"selected":selection["selected"],"sha256":manifest["blob_sha256"],
        "masks":masks
    }),flush=True)
    return manifest


def score(data_dir:Path, output_dir:Path)->dict:
    manifest=json.loads((output_dir/"manifest.json").read_text())
    target=data_dir/f"Phase {manifest['phase']} - Sales.csv"
    frozen=output_dir/"forecast-frozen.npz"
    if hashlib.sha256(frozen.read_bytes()).hexdigest()!=manifest["blob_sha256"]:
        raise ValueError("Frozen predictions changed since selection")
    y=read_sales(target)
    h=read_history(manifest["phase"],data_dir)
    if not y[KEYS].equals(h[KEYS]) or len(y)!=manifest["n_series"] or list(y.columns[3:])!=manifest["prediction_dates"]:
        raise ValueError("Retrospective targets misaligned with frozen forecast")
    actual=y.iloc[:,len(KEYS):].to_numpy(dtype=np.float64)
    scores={}
    with np.load(frozen) as fp:
        for name in manifest["forecast_candidates"]:
            scores[name]=_metric(np.asarray(fp[name],dtype=np.float64),actual)
    out={
        "phase":manifest["phase"],"score_name":"official_VN1_lower_better",
        "scores":scores, "selected":manifest["selection"]["selected"],
        "historical_accepted":manifest["selection"]["accepted"],
        "frozen_sha256":manifest["blob_sha256"],
        "retrospective_not_blind":True,
    }
    (output_dir/"score-receipt.json").write_text(json.dumps(out,indent=2)+"\n")
    print("VN1_SOFT_DORMANCY_SCORES="+json.dumps(scores,sort_keys=True),flush=True)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=("freeze","score"))
    parser.add_argument("--phase",type=int,choices=(1,2),default=2)
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--trees",type=int,default=160)
    args=parser.parse_args()
    if args.operation=="freeze":
        freeze(args.data_dir,args.output_dir,args.phase,args.trees)
    else:
        score(args.data_dir,args.output_dir)


if __name__=="__main__":
    main()
