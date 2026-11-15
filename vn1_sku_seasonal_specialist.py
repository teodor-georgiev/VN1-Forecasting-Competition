"""Research-only seasonal top-SKU correction above VN1 persistent-bias incumbent.

No target after the forecast origin enters features. Candidate family and
acceptance rule are fixed before looking at any Phase 1 or Phase 2 target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import vn1_champion_residual_oof as base
import vn1_persistent_bias_specialist as incumbent
from vn1_amsi_zero_rule_ablation import _metric
from vn1_from_scratch import HORIZON, KEYS, read_sales

CANDIDATES = ("incumbent", "sku_annual10", "sku_peer10", "sku_hybrid15")
MIN_GAIN = 0.001
MIN_TRAIN_FOLD_WINS = 2


def annual_signals(history: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """History-only 13-week forward annual seasonality; leave-one-out peers.

    At origin T, previous-year future covers [T-52, T-39), compared with
    previous-year preceding weeks [T-65, T-52). This never touches [T, T+13).
    """
    y = base._history_matrix(history)
    if y.shape[1] < 65:
        raise ValueError("Need 65 observed weeks for annual future comparison")
    old_before = y[:, -65:-52].sum(axis=1)
    old_future = y[:, -52:-39].sum(axis=1)
    own = np.log((old_future + 5.0) / (old_before + 5.0))
    own = np.clip(own, -0.5, 0.5)

    peer_components = []
    for col in ("Client", "Warehouse", "Product"):
        codes, _ = pd.factorize(history[col], sort=True)
        count = np.bincount(codes)
        before_sum = np.bincount(codes, weights=old_before)
        future_sum = np.bincount(codes, weights=old_future)
        # Exclude this series: the peer factor cannot simply echo its signal.
        before_peer = np.maximum(0., before_sum[codes] - old_before)
        future_peer = np.maximum(0., future_sum[codes] - old_future)
        ratio = np.log((future_peer + 50.0) / (before_peer + 50.0))
        ratio[count[codes] < 2] = 0.0
        peer_components.append(np.clip(ratio, -0.5, 0.5))

    # Product peers span client/warehouse; the three signals are bounded.
    peer = (0.4 * peer_components[0] + 0.2 * peer_components[1]
            + 0.4 * peer_components[2])
    if not np.isfinite(own).all() or not np.isfinite(peer).all():
        raise ArithmeticError("Invalid annual history feature")
    return own, peer


def make_candidates(
    history: pd.DataFrame,
    existing: np.ndarray,
    records: list[dict],
) -> tuple[dict[str, np.ndarray], dict]:
    """Top-volume-only adjustments with weekly total sales conservation."""
    y = base._history_matrix(history)
    control = np.asarray(existing, dtype=np.float64)
    if (control.shape != (len(y), HORIZON)
            or not np.isfinite(control).all() or (control < 0).any()):
        raise ValueError("Invalid current specialist forecast")
    effects = incumbent.estimate_effects(history, records)
    active = effects["active"]
    top = effects["top_volume"] & active
    own, peer = annual_signals(history)
    trusted_sku = effects["series"]  # Completed OOF residual-sign confirmation.

    factors = {
        "sku_annual10": 0.10 * own,
        "sku_peer10": 0.10 * peer,
        "sku_hybrid15": 0.10 * own + 0.10 * peer + 0.15 * trusted_sku,
    }
    results = {"incumbent": control.copy()}
    for name, raw in factors.items():
        rate = np.zeros(len(y), dtype=np.float64)
        rate[top] = np.clip(raw[top], -0.15, 0.15)
        candidate = control.copy()
        candidate[active] = np.maximum(
            0., candidate[active] * (1. + rate[active, None]))
        # Correct relative allocations, not global forecast volume: VN1
        # separately penalizes global bias and the incumbent already calibrates it.
        desired = control[active].sum(axis=0)
        observed = candidate[active].sum(axis=0)
        if np.any((desired > 0) & (observed <= 0)):
            candidate = control.copy()
        else:
            multiplier = np.divide(
                desired, observed, out=np.ones_like(desired), where=observed > 0)
            candidate[active] *= multiplier[None, :]
        if (not np.isfinite(candidate).all() or (candidate < 0).any()
                or not np.allclose(candidate.sum(axis=0), control.sum(axis=0),
                                   atol=1e-6, rtol=1e-10)):
            raise ArithmeticError("Candidate changed global demand or invalid values")
        np.testing.assert_array_equal(candidate[~active], control[~active])
        results[name] = candidate
    if tuple(results) != CANDIDATES:
        raise AssertionError("Unexpected predeclared candidates")
    return results, {
        "top_volume": int(top.sum()),
        "historically_active": int(active.sum()),
        "trusted_persistent_sku": int(np.count_nonzero(trusted_sku)),
        "cutoff": int(y.shape[1]),
        "completed_oof_origins": [int(r["origin"]) for r in records],
    }


def historical_validation(
    phase0: pd.DataFrame,
    *,
    trees: int = 160,
    activity_trees: int = 100,
    client_cap: int = 16,
) -> dict:
    """Compare against the existing PR21 persistent-bias incumbent, never PR17."""
    y = base._history_matrix(phase0)
    if y.shape[1] < base.HOLDOUT_ORIGIN + HORIZON:
        raise ValueError("Incomplete validation data")
    losses = {name: {"absolute": 0., "signed": 0.} for name in CANDIDATES}
    wins = {name: 0 for name in CANDIDATES}
    folds, details = {}, {}
    total = 0.
    cache: dict[int, dict] = {}
    for origin in (*base.TRAIN_EVAL_ORIGINS, base.HOLDOUT_ORIGIN):
        past = phase0.iloc[:, :len(KEYS) + origin].copy()
        records = base._completed_records(
            phase0, origin, cache, trees=trees,
            activity_trees=activity_trees, client_cap=client_cap)
        original, _ = base.champion_forecast(
            past, trees=trees, activity_trees=activity_trees, client_cap=client_cap)
        incumbent_choices, _ = incumbent.candidate_forecasts(
            past, original, records)
        current = incumbent_choices["group_unscaled25"]
        candidates, diagnostic = make_candidates(past, current, records)
        truth = y[:, origin:origin + HORIZON]
        if truth.shape != current.shape:
            raise ValueError("Historical target misalignment")
        scores = {name: float(_metric(pred, truth))
                  for name, pred in candidates.items()}
        folds[str(origin)] = scores
        details[str(origin)] = diagnostic
        if origin != base.HOLDOUT_ORIGIN:
            total += float(truth.sum())
            for name, pred in candidates.items():
                diff = pred - truth
                losses[name]["absolute"] += float(np.abs(diff).sum())
                losses[name]["signed"] += float(diff.sum())
                if scores[name] < scores["incumbent"]:
                    wins[name] += 1
        print("VN1_SKU_SEASON_FOLD=" + str(origin), flush=True)

    if total <= 0:
        raise ValueError("Historical sales volume must be positive")
    train = {name: (value["absolute"] + abs(value["signed"])) / total
             for name, value in losses.items()}
    winner = min(CANDIDATES, key=lambda x: (train[x], CANDIDATES.index(x)))
    holdout = folds[str(base.HOLDOUT_ORIGIN)]
    accepted = (
        winner != "incumbent"
        and train[winner] <= train["incumbent"] - MIN_GAIN
        and holdout[winner] <= holdout["incumbent"] - MIN_GAIN
        and wins[winner] >= MIN_TRAIN_FOLD_WINS
    )
    return {
        "baseline": "PR21 group_unscaled25 persistent-bias specialist",
        "candidate_policies": list(CANDIDATES),
        "training_origins": list(base.TRAIN_EVAL_ORIGINS),
        "holdout_origin": base.HOLDOUT_ORIGIN,
        "training_scores": train,
        "holdout_scores": holdout,
        "fold_scores": folds,
        "training_fold_wins": wins,
        "winner_training_only": winner,
        "selected": winner if accepted else "incumbent",
        "accepted": bool(accepted),
        "min_gain": MIN_GAIN,
        "diagnostics": details,
        "used_phase1_labels": False,
        "used_phase2_labels": False,
        "retrospective_not_blind": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("results/vn1-sku-seasonal.json"))
    parser.add_argument("--trees", type=int, default=160)
    parser.add_argument("--activity-trees", type=int, default=100)
    parser.add_argument("--client-cap", type=int, default=16)
    args = parser.parse_args()
    if args.trees < 1 or args.activity_trees < 1 or not 0 <= args.client_cap <= 512:
        parser.error("Invalid model capacity")
    phase0 = read_sales(args.data_dir / "Phase 0 - Sales.csv")
    receipt = historical_validation(
        phase0, trees=args.trees, activity_trees=args.activity_trees,
        client_cap=args.client_cap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Do not overwrite an existing benchmark receipt")
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("VN1_SKU_SEASON_RESULT=" + json.dumps({
        "selected": receipt["selected"],
        "accepted": receipt["accepted"],
        "train": receipt["training_scores"],
        "holdout": receipt["holdout_scores"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
