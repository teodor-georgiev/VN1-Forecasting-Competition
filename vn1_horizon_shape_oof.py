"""Research-only horizon-shape correction against the PR21 incumbent.

Estimate timing errors from completed OOF forecasts, not Phase 1/2 targets.
Each active series retains its own 13-week total; only timing can change.
The reused historical validation is exploratory, never a blind test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import vn1_champion_residual_oof as base
import vn1_persistent_bias_specialist as persistent
from vn1_from_scratch import HORIZON, KEYS, read_sales

POLICIES = ("incumbent", "global15", "group15", "group30")
MIN_GAIN = 0.001


def _shape_rate(pred, actual, codes, prior):
    """Pool group lead-time shares; remove each completed fold's level bias."""
    groups = int(codes.max()) + 1
    p = np.zeros((groups, HORIZON))
    a = np.zeros_like(p)
    for f, t in zip(pred, actual):
        fp = np.zeros_like(p)
        ft = np.zeros_like(p)
        np.add.at(fp, codes, f)
        np.add.at(ft, codes, t)
        total_p = fp.sum(axis=1, keepdims=True)
        total_t = ft.sum(axis=1, keepdims=True)
        valid = (total_p > 0) & (total_t > 0)
        scale = np.divide(total_p, total_t, out=np.zeros_like(total_p), where=valid)
        p += np.where(valid, fp, 0.0)
        a += ft * scale
    return np.clip((a - p) / (p + prior), -0.30, 0.30)[codes]


def candidate_forecasts(history, incumbent, records):
    y = base._history_matrix(history)
    n, cutoff = y.shape
    control = np.asarray(incumbent, dtype=np.float64)
    if not n or control.shape != (n, HORIZON) or not np.isfinite(control).all() or (control < 0).any():
        raise ValueError("Invalid incumbent forecast")
    origins = [int(r["origin"]) for r in records]
    if not origins or origins != sorted(set(origins)):
        raise ValueError("Need unique chronological completed OOF records")
    if any(o < base.OOF_START or o + HORIZON > cutoff for o in origins):
        raise ValueError("OOF contains future labels")
    preds, actuals = [], []
    for record in records[-3:]:
        p, a = (np.asarray(record[k], dtype=np.float64) for k in ("champion", "truth"))
        if any(x.shape != control.shape or not np.isfinite(x).all() or (x < 0).any() for x in (p, a)):
            raise ValueError("Invalid OOF forecast or truth")
        # Check completed truth against the observed history, including row alignment.
        o = int(record["origin"])
        if not np.array_equal(a, y[:, o:o+HORIZON]):
            raise ValueError("OOF truth differs from completed history")
        preds.append(p)
        actuals.append(a)
    global_rate = _shape_rate(preds, actuals, np.zeros(n, dtype=int), 2500.0)
    group_rates = []
    for key in ("Client", "Warehouse"):
        codes, _ = pd.factorize(history[key], sort=True)
        if (codes < 0).any():
            raise ValueError("Missing group key")
        group_rates.append(_shape_rate(preds, actuals, codes, 2500.0))
    group_rate = 0.5 * global_rate + 0.25 * (group_rates[0] + group_rates[1])
    active = base._active_mask(y)
    choices = {"incumbent": control.copy()}
    for name, rate, weight in (("global15", global_rate, .15),
                                ("group15", group_rate, .15),
                                ("group30", group_rate, .30)):
        candidate = control.copy()
        modified = control[active] * (1.0 + weight * rate[active])
        before = control[active].sum(axis=1, keepdims=True)
        after = modified.sum(axis=1, keepdims=True)
        candidate[active] = modified * np.divide(before, after, out=np.ones_like(before), where=after > 0)
        if not np.isfinite(candidate).all() or (candidate < 0).any():
            raise ArithmeticError("Invalid candidate")
        if not np.allclose(candidate.sum(axis=1), control.sum(axis=1), atol=1e-7, rtol=1e-10):
            raise ArithmeticError("Series horizon totals changed")
        if not np.array_equal(candidate[~active], control[~active]):
            raise ArithmeticError("Inactive forecast changed")
        choices[name] = candidate
    return choices


def historical_validation(phase0, *, trees=160, activity_trees=100, client_cap=16):
    y = base._history_matrix(phase0)
    if y.shape[1] < base.HOLDOUT_ORIGIN + HORIZON:
        raise ValueError("Incomplete historical holdout")
    cache, fold_scores = {}, {}
    totals = {k: np.zeros(2) for k in POLICIES}
    wins = dict.fromkeys(POLICIES, 0)
    demand = 0.0
    for origin in (*base.TRAIN_EVAL_ORIGINS, base.HOLDOUT_ORIGIN):
        past = phase0.iloc[:, :len(KEYS)+origin].copy()
        options = dict(trees=trees, activity_trees=activity_trees, client_cap=client_cap)
        records = base._completed_records(phase0, origin, cache, **options)
        original, _ = base.champion_forecast(past, **options)
        incumbent = persistent.candidate_forecasts(past, original, records)[0]["group_unscaled25"]
        forecasts = candidate_forecasts(past, incumbent, records)
        truth = y[:, origin:origin+HORIZON]
        volume = float(truth.sum())
        if volume <= 0:
            raise ValueError("Zero validation demand")
        components = {}
        for name, forecast in forecasts.items():
            diff = forecast - truth
            components[name] = np.array([np.abs(diff).sum(), diff.sum()])
        scores = {k: float((v[0]+abs(v[1]))/volume) for k, v in components.items()}
        fold_scores[str(origin)] = scores
        if origin != base.HOLDOUT_ORIGIN:
            demand += volume
            for k, v in components.items():
                totals[k] += v
                wins[k] += int(scores[k] < scores["incumbent"])
        print("VN1_HORIZON_SHAPE_FOLD=" + str(origin), flush=True)
    train = {k: float((v[0]+abs(v[1]))/demand) for k, v in totals.items()}
    holdout = fold_scores[str(base.HOLDOUT_ORIGIN)]
    winner = min(POLICIES, key=train.__getitem__)
    accepted = bool(winner != "incumbent" and
                    train[winner] <= train["incumbent"] - MIN_GAIN and
                    holdout[winner] <= holdout["incumbent"] - MIN_GAIN and
                    wins[winner] >= 2)
    return {"incumbent": "PR21 group_unscaled25", "series": len(y), "horizon": HORIZON,
            "training_origins": list(base.TRAIN_EVAL_ORIGINS), "holdout_origin": base.HOLDOUT_ORIGIN,
            "training_scores": train, "holdout_scores": holdout, "fold_scores": fold_scores,
            "training_winner": winner, "train_fold_wins": wins, "accepted": accepted,
            "selected": winner if accepted else "incumbent",
            "used_phase1_labels": False, "used_phase2_labels": False,
            "reused_historical_holdout_exploratory": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("."))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a research receipt")
    phase0 = read_sales(args.data_dir / "Phase 0 - Sales.csv")
    if len(phase0) != 15053:
        raise ValueError("Full VN1 catalogue required")
    receipt = historical_validation(phase0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
    print("VN1_HORIZON_SHAPE_RESULT=" + json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
