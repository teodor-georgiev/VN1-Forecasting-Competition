"""Pre-origin counterfactual SKU expert selection against PR21 persistent incumbent.

Never compare candidates to future targets during training or inference. Select
a bounded expert blend only when the expert repeatedly beats original champion
on *completed* out-of-fold 13-week windows. Research only, never promoted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import vn1_champion_residual_oof as base
import vn1_persistent_bias_specialist as persistent
from vn1_amsi_zero_rule_ablation import _metric
from vn1_from_scratch import KEYS, HORIZON, read_sales

POLICIES = ("incumbent", "oof_recent15", "oof_annual15", "oof_annual30")
MIN_GAIN = 0.001
MIN_WINS = 2


def simpler_experts(values: np.ndarray) -> dict[str, np.ndarray]:
    """Shape-preserving annual matched weeks versus recent-volume constant."""
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] < 65 or (y < 0).any() or not np.isfinite(y).all():
        raise ValueError("Need >=65 finite nonnegative observed weekly sales")
    recent = y[:, -13:].mean(axis=1, keepdims=True)
    recent_forecast = np.repeat(recent, HORIZON, axis=1)
    annual = y[:, -52:-39]
    annual_blend = 0.5 * annual + 0.5 * recent_forecast
    return {"recent": recent_forecast, "annual": annual_blend}


def oof_trust_masks(
    history: pd.DataFrame,
    records: list[dict],
) -> tuple[dict[str, np.ndarray], dict]:
    """Only completed, chronological OOF records can select product experts."""
    y = base._history_matrix(history)
    n, cutoff = y.shape
    if not records:
        raise ValueError("Need at least one completed champion OOF window")
    origins = [int(r["origin"]) for r in records]
    if origins != sorted(set(origins)):
        raise ValueError("OOF origins must be unique and chronological")
    if any(o < 105 or o + HORIZON > cutoff for o in origins):
        raise ValueError("OOF contains future labels or invalid origin")
    before_errors = {"recent": [], "annual": []}
    improvement = {"recent": [], "annual": []}
    for r in records[-2:]:
        o = int(r["origin"])
        actual = np.asarray(r["truth"], dtype=np.float64)
        champion = np.asarray(r["champion"], dtype=np.float64)
        if (actual.shape != (n, HORIZON) or champion.shape != actual.shape
                or not np.isfinite(actual).all() or not np.isfinite(champion).all()
                or (actual < 0).any() or (champion < 0).any()):
            raise ValueError("OOF champion/truth shape or values invalid")
        expert = simpler_experts(y[:, :o])
        champion_l1 = np.abs(champion - actual).sum(axis=1)
        for key, forecast in expert.items():
            alt_l1 = np.abs(forecast - actual).sum(axis=1)
            # Demand threshold comes from historical observations at origin,
            # not from the OOF target that is evaluated here.
            prior_volume = y[:, o-13:o].sum(axis=1)
            qualified = (prior_volume >= 26.) & (champion_l1 >= 1.)
            improvement[key].append(qualified & (alt_l1 <= champion_l1 * .92))
            before_errors[key].append(float(np.count_nonzero(qualified)))
    # OOF reliability: when two windows exist require wins in BOTH.
    trusted = {name: np.logical_and.reduce(bits) for name, bits in improvement.items()}
    diagnostics = {
        "origins_checked": origins[-2:],
        "trusted_recent": int(trusted["recent"].sum()),
        "trusted_annual": int(trusted["annual"].sum()),
        "validated_on_completed_oof_only": True,
    }
    return trusted, diagnostics


def candidate_forecasts(
    history: pd.DataFrame,
    existing: np.ndarray,
    records: list[dict],
) -> tuple[dict[str, np.ndarray], dict]:
    """Controlled top-volume SKU blending; keep each week's total unchanged."""
    y = base._history_matrix(history)
    n = len(y)
    incumbent = np.asarray(existing, dtype=np.float64)
    if (incumbent.shape != (n, HORIZON) or not np.isfinite(incumbent).all()
            or (incumbent < 0).any()):
        raise ValueError("Invalid existing specialist forecast")
    trusted, report = oof_trust_masks(history, records)
    experts = simpler_experts(y)
    active = base._active_mask(y)
    ranked = np.argsort(-y[:, -13:].mean(axis=1), kind="stable")
    top = np.zeros(n, dtype=bool)
    top[ranked[:max(1, int(np.ceil(n * 0.10)))]] = True
    eligible = top & active
    policies = {"incumbent": incumbent.copy()}
    options = (
        ("oof_recent15", "recent", 0.15),
        ("oof_annual15", "annual", 0.15),
        ("oof_annual30", "annual", 0.30),
    )
    for name, expert, weight in options:
        mask = eligible & trusted[expert]
        proposed = incumbent.copy()
        proposed[mask] = ((1. - weight) * incumbent[mask]
                          + weight * experts[expert][mask])
        # Preserve PR21's calibrated total units in EVERY forecast week.
        if active.any():
            before = incumbent[active].sum(axis=0)
            after = proposed[active].sum(axis=0)
            if np.any((before > 0) & (after <= 0)):
                raise ArithmeticError("Blend removed all weekly active demand")
            scale = np.divide(before, after, out=np.ones_like(before), where=after > 0)
            proposed[active] *= scale[None, :]
        if (not np.isfinite(proposed).all() or (proposed < 0).any()
                or not np.allclose(proposed.sum(axis=0), incumbent.sum(axis=0),
                                   atol=1e-6, rtol=1e-10)):
            raise ArithmeticError("Candidate altered total forecast or went invalid")
        if not np.array_equal(proposed[~active], incumbent[~active]):
            raise ArithmeticError("Inactive forecasts were changed")
        policies[name] = proposed
        report[name + "_affected_series"] = int(mask.sum())
    if tuple(policies) != POLICIES:
        raise AssertionError("Unexpected fixed candidate list")
    report["top_decile_series"] = int(eligible.sum())
    return policies, report


def historical_validation(
    phase0: pd.DataFrame,
    *,
    trees: int = 160,
    activity_trees: int = 100,
    client_cap: int = 16,
) -> dict:
    """Training-only choice and strictly later Phase 0 historical veto."""
    y = base._history_matrix(phase0)
    if y.shape[1] < base.HOLDOUT_ORIGIN + HORIZON:
        raise ValueError("Historical origin 157 truth unavailable")
    losses = {k: {"l1": 0., "signed": 0.} for k in POLICIES}
    wins = {k: 0 for k in POLICIES}
    per_fold, diagnostics = {}, {}
    total = 0.
    cache: dict[int, dict] = {}
    for origin in (*base.TRAIN_EVAL_ORIGINS, base.HOLDOUT_ORIGIN):
        past = phase0.iloc[:, :len(KEYS)+origin].copy()
        records = base._completed_records(
            phase0, origin, cache, trees=trees, activity_trees=activity_trees,
            client_cap=client_cap)
        original, _ = base.champion_forecast(
            past, trees=trees, activity_trees=activity_trees, client_cap=client_cap)
        current = persistent.candidate_forecasts(past, original, records)[0]["group_unscaled25"]
        alternatives, details = candidate_forecasts(past, current, records)
        truth = y[:, origin:origin+HORIZON]
        scores = {name: float(_metric(pred, truth))
                  for name, pred in alternatives.items()}
        per_fold[str(origin)] = scores
        diagnostics[str(origin)] = details
        if origin != base.HOLDOUT_ORIGIN:
            total += float(truth.sum())
            for name, pred in alternatives.items():
                err = pred - truth
                losses[name]["l1"] += float(np.abs(err).sum())
                losses[name]["signed"] += float(err.sum())
                if scores[name] < scores["incumbent"]:
                    wins[name] += 1
        print("VN1_COUNTERFACTUAL_FOLD=" + str(origin), flush=True)
    training = {k: (v["l1"] + abs(v["signed"])) / total for k, v in losses.items()}
    winner = min(POLICIES, key=lambda name: (training[name], POLICIES.index(name)))
    holdout = per_fold[str(base.HOLDOUT_ORIGIN)]
    accepted = bool(
        winner != "incumbent"
        and training[winner] <= training["incumbent"]-MIN_GAIN
        and holdout[winner] <= holdout["incumbent"]-MIN_GAIN
        and wins[winner] >= MIN_WINS
    )
    return {
        "benchmark": "full VN1 Phase 0 historical; no phase 1 or 2",
        "incumbent": "PR21 group_unscaled25",
        "training_origins": list(base.TRAIN_EVAL_ORIGINS),
        "holdout_origin": base.HOLDOUT_ORIGIN,
        "policies": list(POLICIES),
        "fold_scores": per_fold,
        "training_scores": training,
        "holdout_scores": holdout,
        "train_fold_wins": wins,
        "training_winner": winner,
        "accepted": accepted,
        "selected": winner if accepted else "incumbent",
        "diagnostics": diagnostics,
        "used_phase1_labels": False,
        "used_phase2_labels": False,
        "holdout_reused_exploratory": True,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("."))
    p.add_argument("--output", type=Path, default=Path("results/vn1-counterfactual.json"))
    p.add_argument("--trees", type=int, default=160)
    p.add_argument("--activity-trees", type=int, default=100)
    p.add_argument("--client-cap", type=int, default=16)
    args = p.parse_args()
    phase0 = read_sales(args.data_dir/"Phase 0 - Sales.csv")
    receipt = historical_validation(
        phase0, trees=args.trees,
        activity_trees=args.activity_trees, client_cap=args.client_cap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Refuse to overwrite prior experiment receipt")
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
    print("VN1_COUNTERFACTUAL_RESULT=" + json.dumps({
        "training": receipt["training_scores"],
        "holdout": receipt["holdout_scores"],
        "selected": receipt["selected"],
        "accepted": receipt["accepted"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
