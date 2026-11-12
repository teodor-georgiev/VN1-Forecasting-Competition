"""Causal hierarchical VN1 demand correction atop the immutable ARIMA+AMSI policy.

Research only. Estimate global/client/warehouse signed demand biases from fully
completed champion OOF windows, then separate client total demand forecasting
from active product allocation. Select on Phase 0; gate on a held-out Phase 0
origin and the separately observed Phase 1 outcome. No Phase 2 labels are read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import vn1_champion_residual_oof as base
import vn1_persistent_bias_specialist as previous
from vn1_from_scratch import HORIZON, KEYS, read_sales

POLICIES = (
    "champion",
    "global_bias20",
    "clientwarehouse25",
    "hier_clientwarehouse25",
    "hier_sku25",
)
TRAIN_ORIGINS = tuple(base.TRAIN_EVAL_ORIGINS)
HOLDOUT_ORIGIN = base.HOLDOUT_ORIGIN
MIN_TRAIN_GAIN = 0.001
MIN_HOLDOUT_GAIN = 0.001
MIN_PHASE1_GAIN = 0.0002
MIN_TRAIN_WINS = 2


def components(forecast: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Exact official VN1 score split into absolute and global signed error."""
    p = np.asarray(forecast, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if p.shape != y.shape or p.ndim != 2 or p.shape[1] < 1:
        raise ValueError("Forecast/target shape mismatch")
    if (not np.isfinite(p).all() or not np.isfinite(y).all()
            or (p < 0).any() or (y < 0).any()):
        raise ValueError("Nonfinite or negative demand in metric")
    denominator = float(y.sum())
    if denominator <= 0:
        raise ValueError("No positive observed demand")
    error = p-y
    absolute = float(np.abs(error).sum()/denominator)
    signed = float(error.sum()/denominator)
    return {"score": absolute+abs(signed), "absolute": absolute,
            "signed": signed, "bias_penalty": abs(signed)}


def historical_bias(history: pd.DataFrame, records: list[dict]) -> dict:
    """History-only smoothed bias at three demand levels.

    The existing specialist validates OOF chronology, completed horizons,
    series alignment and product sign stability. The correction is estimated
    from the latest three *completed* historical champion residual windows.
    """
    existing = previous.estimate_effects(history, records)
    observed = base._history_matrix(history)
    n, _ = observed.shape
    p = np.zeros(n, dtype=np.float64)
    e = np.zeros(n, dtype=np.float64)
    for record in records[-3:]:
        forecast = np.asarray(record["champion"], dtype=np.float64)
        actual = np.asarray(record["truth"], dtype=np.float64)
        p += forecast.sum(axis=1)
        e += (actual-forecast).sum(axis=1)
    global_rate = float(np.clip(e.sum()/(p.sum()+10000.0), -0.25, 0.25))
    group_rates = {}
    for name in ("Client", "Warehouse"):
        indices, _ = pd.factorize(history[name], sort=True)
        residual = np.bincount(indices, weights=e)
        demand = np.bincount(indices, weights=p)
        # Shrink toward the aggregate prior; 2500 pseudo-units prevents a
        # small historical group from producing extreme percentage changes.
        rate = np.clip(residual/(demand+2500.0), -0.25, 0.25)
        group_rates[name.lower()] = rate[indices]
    if not all(len(x) == n for x in group_rates.values()):
        raise AssertionError("Historical grouping changed series order")
    return {
        "global": global_rate,
        "client": group_rates["client"],
        "warehouse": group_rates["warehouse"],
        "series": existing["series"],
        "active": existing["active"],
        "top_volume": existing["top_volume"],
    }


def candidates(history: pd.DataFrame, champion: np.ndarray,
               records: list[dict]) -> tuple[dict[str, np.ndarray], dict]:
    """Fixed global/direct/hierarchical policies. Inactive rows never change."""
    n = len(history)
    control = np.asarray(champion, dtype=np.float64)
    if (control.shape != (n, HORIZON) or not np.isfinite(control).all()
            or (control < 0).any()):
        raise ValueError("Invalid champion matrix")
    x = historical_bias(history, records)
    active = x["active"]
    client = x["client"]
    warehouse = x["warehouse"]
    sku = x["series"]
    global_rate = x["global"]

    def direct(rate: np.ndarray) -> np.ndarray:
        p = control.copy()
        p[active] *= np.maximum(0, 1.0+rate[active, None])
        return p

    def hierarchy(*, sku_allocation: bool) -> np.ndarray:
        result = control.copy()
        labels, _ = pd.factorize(history["Client"], sort=True)
        for cid in np.unique(labels):
            members = (labels == cid) & active
            if not members.any():
                continue
            original = control[members]
            source_total = original.sum(axis=0)
            # Stage 1: client demand corrected by a smoothed global prior.
            adjustment = 0.25*(0.75*client[members][0]+0.25*global_rate)
            target_total = source_total*(1.0+adjustment)
            # Stage 2: reallocate among active products within each client.
            # Warehouse and persistent product signal change shares, not the
            # client's fitted total. No future product data is required.
            redistribution = 0.20*warehouse[members]
            if sku_allocation:
                redistribution = redistribution + 0.20*sku[members]
            weights = original*np.maximum(0.5, 1.0+redistribution[:, None])
            available = weights.sum(axis=0)
            scale = np.divide(target_total, available,
                              out=np.ones_like(target_total), where=available>0)
            result[members] = weights*scale[None, :]
            if np.any((available <= 0)&(target_total > 0)):
                raise ArithmeticError("Cannot allocate positive client demand")
            if not np.allclose(result[members].sum(axis=0), target_total,
                               rtol=0, atol=1e-7):
                raise ArithmeticError("Client total demand changed after allocation")
        return result

    policies = {
        "champion": control.copy(),
        "global_bias20": direct(np.full(n, 0.20*global_rate)),
        "clientwarehouse25": direct(0.25*(client+warehouse)*0.5),
        "hier_clientwarehouse25": hierarchy(sku_allocation=False),
        "hier_sku25": hierarchy(sku_allocation=True),
    }
    assert tuple(policies) == POLICIES
    for name, forecast in policies.items():
        if not np.isfinite(forecast).all() or (forecast < 0).any():
            raise ArithmeticError(f"{name} produced invalid forecast")
        if not np.array_equal(forecast[~active], control[~active]):
            raise ArithmeticError(f"{name} modified inactive champion fallback")
    return policies, {
        "oof_origins": [int(record["origin"]) for record in records],
        "active_series": int(active.sum()),
        "top_volume": int(x["top_volume"].sum()),
        "stable_sku_count": int(np.count_nonzero(sku)),
        "global_bias": global_rate,
    }


def _expected_phase1_dates(phase0: pd.DataFrame) -> list[str]:
    return list(pd.date_range(
        pd.Timestamp(phase0.columns[-1])+pd.Timedelta(weeks=1),
        periods=HORIZON, freq="7D").strftime("%Y-%m-%d"))


def _check_phase1(phase0: pd.DataFrame, phase1: pd.DataFrame) -> None:
    if (not phase0[KEYS].equals(phase1[KEYS])
            or list(phase1.columns[:len(KEYS)]) != list(KEYS)
            or list(phase1.columns[len(KEYS):]) != _expected_phase1_dates(phase0)):
        raise ValueError("Phase 1 keys, order or 13-week dates do not match Phase 0")
    y = phase1.iloc[:, len(KEYS):].to_numpy(dtype=np.float64)
    if y.shape != (len(phase0), HORIZON) or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Phase 1 targets invalid")


def validate(phase0: pd.DataFrame, phase1: pd.DataFrame, *,
             trees: int = 160, activity_trees: int = 100,
             client_cap: int = 16) -> dict:
    """Chronological Phase 0 selection + independent Phase 1 acceptance gate.

    The phase1 DataFrame is not accessed until the Phase 1 predictions have
    been assembled. No label from any evaluation horizon participates in that
    horizon's fit, and Phase 1 only gates the Phase-0-selected winner.
    """
    _check_phase1(phase0, phase1)
    observed = base._history_matrix(phase0)
    if observed.shape[1] < HOLDOUT_ORIGIN+HORIZON:
        raise ValueError("Not enough completed Phase 0 validation periods")

    totals = {name: {"abs": 0.0, "signed": 0.0} for name in POLICIES}
    fold_scores = {}
    fold_wins = {name: 0 for name in POLICIES}
    diagnostics = {}
    sum_targets = 0.0
    cache: dict[int, dict] = {}
    for origin in (*TRAIN_ORIGINS, HOLDOUT_ORIGIN):
        past = phase0.iloc[:, :len(KEYS)+origin].copy()
        recs = base._completed_records(
            phase0, origin, cache, trees=trees,
            activity_trees=activity_trees, client_cap=client_cap)
        champion, _ = base.champion_forecast(
            past, trees=trees,
            activity_trees=activity_trees, client_cap=client_cap)
        variants, details = candidates(past, champion, recs)
        truth = observed[:, origin:origin+HORIZON]
        fold_scores[str(origin)] = {name: components(pred, truth)
                                    for name, pred in variants.items()}
        diagnostics[str(origin)] = details
        if origin != HOLDOUT_ORIGIN:
            sum_targets += float(truth.sum())
            for name, pred in variants.items():
                err = pred-truth
                totals[name]["abs"] += float(np.abs(err).sum())
                totals[name]["signed"] += float(err.sum())
                if fold_scores[str(origin)][name]["score"] < (
                        fold_scores[str(origin)]["champion"]["score"]):
                    fold_wins[name] += 1
        print("VN1_HIERARCHY_PHASE0_FOLD="+str(origin), flush=True)

    if sum_targets <= 0:
        raise ValueError("No training demand for selection")
    train = {name: {"score": (v["abs"]+abs(v["signed"]))/sum_targets,
                    "absolute": v["abs"]/sum_targets,
                    "signed": v["signed"]/sum_targets,
                    "bias_penalty": abs(v["signed"])/sum_targets}
             for name, v in totals.items()}
    # Winner is determined ONLY using Phase 0 completed training origins.
    winner = min(POLICIES, key=lambda name: (train[name]["score"], POLICIES.index(name)))
    holdout = fold_scores[str(HOLDOUT_ORIGIN)]

    # Phase 1 is a dedicated future test. Its candidate forecast is generated
    # entirely from Phase 0 before its observed sales enter the evaluator.
    cutoff = observed.shape[1]
    recs = base._completed_records(
        phase0, cutoff, cache, trees=trees,
        activity_trees=activity_trees, client_cap=client_cap)
    champion, _ = base.champion_forecast(
        phase0, trees=trees,
        activity_trees=activity_trees, client_cap=client_cap)
    variants, future_details = candidates(phase0, champion, recs)
    y1 = phase1.iloc[:, len(KEYS):].to_numpy(dtype=np.float64)
    phase1_metrics = {name: components(pred, y1) for name, pred in variants.items()}

    accepted = (
        winner != "champion"
        and train[winner]["score"] <= train["champion"]["score"]-MIN_TRAIN_GAIN
        and holdout[winner]["score"] <= holdout["champion"]["score"]-MIN_HOLDOUT_GAIN
        and phase1_metrics[winner]["score"] <= (
            phase1_metrics["champion"]["score"]-MIN_PHASE1_GAIN)
        and fold_wins[winner] >= MIN_TRAIN_WINS
    )
    return {
        "training_origins": list(TRAIN_ORIGINS),
        "holdout_origin": HOLDOUT_ORIGIN,
        "training_scores": train,
        "holdout_scores": holdout,
        "phase1_scores": phase1_metrics,
        "train_fold_wins": fold_wins,
        "winner_training_only": winner,
        "selected": winner if accepted else "champion",
        "accepted": bool(accepted),
        "policy_names": list(POLICIES),
        "thresholds": {"train": MIN_TRAIN_GAIN, "holdout": MIN_HOLDOUT_GAIN,
                       "phase1": MIN_PHASE1_GAIN, "train_wins": MIN_TRAIN_WINS},
        "diagnostics": diagnostics,
        "phase1_features": future_details,
        "phase2_targets_read": False,
        "retrospective_not_blind": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path,
                        default=Path("results/hierarchical/historical-phase1.json"))
    parser.add_argument("--trees", type=int, default=160)
    parser.add_argument("--activity-trees", type=int, default=100)
    parser.add_argument("--client-cap", type=int, default=16)
    args = parser.parse_args()
    if args.trees < 1 or args.activity_trees < 1 or not 0 <= args.client_cap <= 512:
        parser.error("Invalid model fit parameters")
    phase0 = read_sales(args.data_dir / "Phase 0 - Sales.csv")
    phase1 = read_sales(args.data_dir / "Phase 1 - Sales.csv")
    report = validate(phase0, phase1, trees=args.trees,
                      activity_trees=args.activity_trees,
                      client_cap=args.client_cap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print("VN1_HIERARCHY_RESULT="+json.dumps({
        "accepted": report["accepted"],
        "selected": report["selected"],
        "winner_training_only": report["winner_training_only"],
        "train": report["training_scores"],
        "holdout": report["holdout_scores"],
        "phase1": report["phase1_scores"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
