"""Retrospective VN1 AMSI + history-only zero-rule experiment.

Two-stage design:
  freeze: choose the policy ONLY on historical Phase-0 rolling origins, and
          save all forecasts before accessing Phase-1/2 target labels.
  score:  evaluate already-frozen candidates against published targets.

No arbitrary product identifiers, inventory assumptions or future price data.
The AMSI approximation is NOT the private competition winner's algorithm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vn1_amsi_specialists import AMSI_CONFIGS, amsi_forecast
from vn1_from_scratch import HORIZON, KEYS, read_sales
from vn1_lgbm_ensemble import choose_weight, ensemble_forecast, read_history

ORIGINS = (131, 144, 157)  # All horizon targets are in Phase 0 (170 weeks).
POLICIES = (
    "incumbent",
    "amsi_short30",
    "amsi_last30",
    "amsi_mean13_30",
    "all_history_zero",
    "inactive52",
    "inactive26_annual_zero",
    "warehouse_closed26",
    "amsi_short30_and_inactive26",
    "inactive52_reallocate_global",
    "inactive26_reallocate_global",
    "inactive52_reallocate_client",
    "warehouse_closed26_reallocate_client",
)
MIN_GAIN = 0.001


def _values(history: pd.DataFrame) -> np.ndarray:
    if list(history.columns[:3]) != list(KEYS) or history.duplicated(KEYS).any():
        raise ValueError("Missing or duplicate series keys")
    y = history.iloc[:, 3:].to_numpy(dtype=np.float64)
    if y.ndim != 2 or y.shape[1] < 65 or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Invalid historical sales")
    return y


def rules(history: pd.DataFrame) -> dict[str, np.ndarray]:
    """Use sales at or before the prediction origin, never evaluation labels."""
    y = _values(history)
    n = len(y)
    never_sold = y.sum(axis=1) == 0
    quiet52 = y[:, -52:].sum(axis=1) == 0
    quiet26 = y[:, -26:].sum(axis=1) == 0
    annual_target_zero = y[:, -52:-39].sum(axis=1) == 0

    # A warehouse identifier alone is NOT globally unique. Group by client+warehouse.
    # Require >=3 observed SKUs in the group and past activity before declaring closed.
    groups = pd.MultiIndex.from_frame(history[["Client", "Warehouse"]])
    grp, _ = pd.factorize(groups, sort=True)
    counts = np.bincount(grp)
    group_recent = np.bincount(grp, weights=y[:, -26:].sum(axis=1))
    group_previous = np.bincount(grp, weights=y[:, -78:-26].sum(axis=1)) if y.shape[1] >= 78 else np.zeros(len(counts))
    warehouse_closed = ((counts[grp] >= 3) &
                        (group_recent[grp] == 0) &
                        (group_previous[grp] >= 12))

    return {
        "all_history_zero": never_sold,
        "inactive52": quiet52,
        "inactive26_annual_zero": quiet26 & annual_target_zero,
        "warehouse_closed26": warehouse_closed,
    }


def incumbent(history: pd.DataFrame, *, trees: int = 160) -> np.ndarray:
    weight = choose_weight(history, n_estimators=trees)["weight_lgbm"]
    pred = ensemble_forecast(history, weight_lgbm=weight,
                             n_estimators=trees).iloc[:, 3:].to_numpy(dtype=np.float64)
    if pred.shape != (len(history), HORIZON):
        raise AssertionError("Champion shape mismatch")
    return np.maximum(pred, 0)


def rebalance_predictions(baseline: np.ndarray, mask: np.ndarray,
                          group: np.ndarray | None = None) -> np.ndarray:
    """Suppress likely dormant series, redistribute their mass without future data.

    Preserve the *global total* of the original forecast so the VN1 bias
    penalty remains unchanged. Distribution is proportional to non-suppressed
    forecast mass, either globally or within the same client. If a client has
    no recipient, its forecast mass is redistributed globally.
    This is an exploratory post-Phase-2-diagnostic candidate, NOT blind tuning.
    """
    baseline = np.asarray(baseline, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if baseline.ndim != 2 or baseline.shape[1] != HORIZON or mask.shape != (len(baseline),):
        raise ValueError("Invalid rebalancing inputs")
    if not np.isfinite(baseline).all() or (baseline < 0).any():
        raise ValueError("Invalid baseline")
    if not mask.any():
        return baseline.copy()
    recipient = ~mask
    if not recipient.any() or baseline[recipient].sum() <= 0:
        return baseline.copy()
    result = baseline.copy()
    result[mask] = 0
    leftover = 0.0
    if group is not None:
        group = np.asarray(group)
        if group.shape != mask.shape:
            raise ValueError("Client groups must align to series")
        codes, _ = pd.factorize(group, sort=True)
        for g in np.unique(codes[mask]):
            dropped = mask & (codes == g)
            kept = recipient & (codes == g)
            mass = float(baseline[dropped].sum())
            denominator = float(result[kept].sum())
            if mass == 0:
                continue
            if denominator > 0:
                result[kept] *= 1 + mass / denominator
            else:
                leftover += mass
    else:
        leftover = float(baseline[mask].sum())
    if leftover > 0:
        denominator = float(result[recipient].sum())
        result[recipient] *= 1 + leftover / denominator
    if not np.isclose(result.sum(), baseline.sum(), rtol=1e-10, atol=1e-5):
        raise ArithmeticError("Bias-neutral reallocation failed to conserve forecast volume")
    return result


def candidates(history: pd.DataFrame, baseline: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    if baseline.shape != (len(history), HORIZON):
        raise ValueError("Incorrect incumbent shape")
    masks = rules(history)
    short = amsi_forecast(history, anchor="short", seasonality=0.60, group_shrink=0.55)
    last = amsi_forecast(history, anchor="last", seasonality=0.45, group_shrink=0.55)
    mean13 = amsi_forecast(history, anchor="mean13", seasonality=0.65, group_shrink=0.70)
    def zero_where(matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
        copy = matrix.copy()
        copy[mask] = 0.0
        return copy
    short30 = 0.7 * baseline + 0.3 * short
    out = {
        "incumbent": baseline,
        "amsi_short30": short30,
        "amsi_last30": 0.7 * baseline + 0.3 * last,
        "amsi_mean13_30": 0.7 * baseline + 0.3 * mean13,
        "all_history_zero": zero_where(baseline, masks["all_history_zero"]),
        "inactive52": zero_where(baseline, masks["inactive52"]),
        "inactive26_annual_zero": zero_where(baseline, masks["inactive26_annual_zero"]),
        "warehouse_closed26": zero_where(baseline, masks["warehouse_closed26"]),
        "amsi_short30_and_inactive26": zero_where(short30, masks["inactive26_annual_zero"]),
        "inactive52_reallocate_global": rebalance_predictions(baseline, masks["inactive52"]),
        "inactive26_reallocate_global": rebalance_predictions(baseline, masks["inactive26_annual_zero"]),
        "inactive52_reallocate_client": rebalance_predictions(
            baseline, masks["inactive52"], history["Client"].to_numpy()),
        "warehouse_closed26_reallocate_client": rebalance_predictions(
            baseline, masks["warehouse_closed26"], history["Client"].to_numpy()),
    }
    if tuple(out) != POLICIES:
        raise AssertionError("Candidate registry changed")
    for name, values in out.items():
        if values.shape != baseline.shape or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid forecast from {name}")
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    return out, counts


def _metric(yhat: np.ndarray, y: np.ndarray) -> float:
    if yhat.shape != y.shape or y.sum() <= 0:
        raise ValueError("Bad forecast/target alignment")
    error = yhat - y
    return float((np.abs(error).sum() + abs(error.sum())) / y.sum())


def select_phase0(phase0: pd.DataFrame, trees: int) -> dict:
    """Train on origins 131 and 144; accept only on distinct origin 157."""
    y = _values(phase0)
    fold_outputs = []
    for origin in ORIGINS:
        past = phase0.iloc[:, :3 + origin].copy()
        base = incumbent(past, trees=trees)
        preds, masks = candidates(past, base)
        target = y[:, origin:origin + HORIZON]
        if target.shape != base.shape or origin + HORIZON > len(y.T):
            raise AssertionError("Validation leakage")
        fold_outputs.append((origin, preds, target, masks))
        print("VN1_RULE_FOLD_COMPLETE=" + str(origin), flush=True)
    total = sum(row[2].sum() for row in fold_outputs[:-1])
    if total <= 0:
        raise ValueError("No historical demand in calibration folds")
    scores = {}
    for name in POLICIES:
        err = np.concatenate([(p[name] - t).reshape(-1) for _, p, t, _ in fold_outputs[:-1]])
        scores[name] = float((np.abs(err).sum() + abs(err.sum())) / total)
    ordered = sorted(POLICIES, key=lambda name: (scores[name], POLICIES.index(name)))
    chosen = ordered[0]
    final_origin, final_preds, final_truth, final_masks = fold_outputs[-1]
    holdout = {name: _metric(p, final_truth) for name,p in final_preds.items()}
    accepted = (
        chosen != "incumbent" and
        scores[chosen] <= scores["incumbent"] - MIN_GAIN and
        holdout[chosen] <= holdout["incumbent"] - MIN_GAIN
    )
    return {
        "train_origins": list(ORIGINS[:-1]),
        "holdout_origin": final_origin,
        "train_scores": scores,
        "holdout_scores": holdout,
        "train_mask_counts": {str(o): counts for o, _, _, counts in fold_outputs},
        "winner_on_train_only": chosen,
        "accepted": bool(accepted),
        "selected": chosen if accepted else "incumbent",
        "min_gain": MIN_GAIN,
    }


def freeze(data_dir: Path, output_dir: Path, *, phase: int, trees: int = 160) -> dict:
    if phase not in (1, 2):
        raise ValueError("Phase must be 1 or 2")
    if output_dir.exists() and list(output_dir.glob("*")):
        raise FileExistsError("Freeze directory must be empty; do not overwrite predictions")
    phase0 = read_sales(data_dir / "Phase 0 - Sales.csv")
    selection = select_phase0(phase0, trees)
    history = read_history(phase, data_dir)
    base = incumbent(history, trees=trees)
    preds, mask_counts = candidates(history, base)
    names = list(preds)
    selected = selection["selected"]
    # Selection is frozen using only Phase 0; never optimized on target phase.
    preds["phase0_selected"] = preds[selected].copy()
    names.append("phase0_selected")

    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / "frozen.npz"
    np.savez_compressed(dest, **{name:preds[name].astype(np.float32) for name in names})
    horizon_dates = list(pd.date_range(pd.to_datetime(history.columns[-1]) + pd.Timedelta(weeks=1),
                                       periods=HORIZON, freq="7D").strftime("%Y-%m-%d"))
    recent_dates = list(history.columns[-52:])
    recent_sales = history.iloc[:, -52:].sum(axis=0).astype(float).tolist()
    receipt = {
        "phase":phase, "series":len(history), "horizon": HORIZON,
        "source": "VN1-Forecasting-Competition; immutable Phase 0 selection",
        "no_future_labels_for_fit_or_selection": True,
        "candidate_names":names, "selection":selection, "mask_counts":mask_counts,
        "future_dates":horizon_dates,
        "recent_52_weeks_dates":recent_dates, "recent_52_weeks_actual_total":recent_sales,
        "freeze_sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),
    }
    (output_dir/"manifest.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("VN1_AMSI_RULES_FROZEN=" + json.dumps({
        "phase": phase, "selected":selected,"sha256":receipt["freeze_sha256"],
        "masks":mask_counts}),flush=True)
    return receipt


def score(data_dir: Path, output_dir: Path) -> dict:
    """Read target truth only AFTER all forecasts and selection were frozen."""
    receipt = json.loads((output_dir/"manifest.json").read_text())
    frozen = output_dir/"frozen.npz"
    if hashlib.sha256(frozen.read_bytes()).hexdigest() != receipt["freeze_sha256"]:
        raise ValueError("Frozen candidate hash mismatch")
    actual = read_sales(data_dir / f"Phase {receipt['phase']} - Sales.csv")
    if len(actual) != receipt["series"]:
        raise ValueError("Unaligned actual rows")
    history = read_history(receipt["phase"], data_dir)
    if not actual[KEYS].equals(history[KEYS]):
        raise ValueError("Phase targets do not align to frozen series keys")
    dates = list(actual.columns[3:])
    if dates != receipt["future_dates"]:
        raise ValueError("Target horizon mismatch")
    y = actual.iloc[:, 3:].to_numpy(dtype=np.float64)
    with np.load(frozen) as data:
        measures = {}
        weekly = {"dates":dates,"actual":y.sum(axis=0).tolist()}
        for name in receipt["candidate_names"]:
            pred = np.asarray(data[name],dtype=np.float64)
            measures[name] = _metric(pred,y)
            if name in ("incumbent","amsi_short30","amsi_last30","phase0_selected","inactive26_annual_zero","inactive26_reallocate_global","inactive52_reallocate_client"):
                weekly[name] = pred.sum(axis=0).tolist()
    result = {
        "phase": receipt["phase"],
        "official_vn1_scores": measures,
        "selected_before_target_scoring": receipt["selection"]["selected"],
        "selection_used_phase0_only": True,
        "weekly_aggregates": weekly,
        "mask_counts": receipt["mask_counts"],
        "reference_incumbent": measures["incumbent"],
        "result_is_retrospective_not_blind": True,
    }
    (output_dir/"official-score-receipt.json").write_text(json.dumps(result,indent=2)+"\n")
    print("VN1_AMSI_RULES_OFFICIAL_SCORES="+json.dumps({k:round(v,10) for k,v in measures.items()},sort_keys=True),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=("freeze","score"))
    parser.add_argument("--phase",type=int,choices=(1,2),default=2)
    parser.add_argument("--data-dir",type=Path,default=Path("."))
    parser.add_argument("--output-dir",type=Path,default=Path("results/amsi-rules"))
    parser.add_argument("--trees",type=int,default=160)
    args=parser.parse_args()
    if args.operation=="freeze":
        freeze(args.data_dir,args.output_dir,phase=args.phase,trees=args.trees)
    else:
        score(args.data_dir,args.output_dir)


if __name__=="__main__":
    main()
