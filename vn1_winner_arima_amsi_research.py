"""History-only VN1 30% AutoARIMA + anchored-seasonality research.

A bounded, explicitly *partial* approximation to the reported winning
architecture, not an exact reproduction of that team's private model:
- 30% ARIMA blend only for series/groups with successful fitted models;
- optional AMSI 10% or 30% additive mix;
- historical learned-activity model is the control, NOT automatically replaced.

Historical selection uses Phase 0 completed forecast origins only.
Scoring released Phase 1 or private Phase 2 targets is a distinct invocation
AFTER forecasts and selection are persisted with SHA-256.
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
from vn1_amsi_zero_rule_ablation import incumbent, _metric
from vn1_learned_activity import (
    ORIGINS, MIN_GAIN, allocate_weekly, predict_activity,
)
from vn1_amsi_specialists import (
    amsi_forecast, make_stats_model, statsforecast_specialist,
)

CANDIDATES = (
    "learned_activity",
    "arima30_series",
    "arima30_client",
    "amsi_last10",
    "amsi_last30",
    "amsi_short10",
    "arima30_series_amsi_last10",
    "arima30_client_amsi_last10",
)
ARIMA_WEIGHT = 0.30
AMSI_SMALL_WEIGHT = 0.10


def _observed(history: pd.DataFrame) -> np.ndarray:
    if list(history.columns[:len(KEYS)]) != list(KEYS) or history.duplicated(KEYS).any():
        raise ValueError("Missing or duplicate VN1 series identifiers")
    values = history.iloc[:, len(KEYS):].to_numpy(dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] < 105 or
        not np.isfinite(values).all() or (values < 0).any()):
        raise ValueError("Invalid historical observations")
    return values


def client_autoarima(
    history: pd.DataFrame, fallback: np.ndarray, *, cap: int = 16,
) -> tuple[np.ndarray, dict]:
    """AutoARIMA on up to cap eligible high-volume client aggregates.

    Each forecast is disaggregated across that client's products in
    proportion to the pre-origin fallback. Never use future observed sales.
    Caps control the compute budget; unfitted clients retain the fallback.
    """
    y = _observed(history)
    control = np.asarray(fallback, dtype=np.float64)
    if (control.shape != (len(y), HORIZON) or not np.isfinite(control).all()
        or (control < 0).any() or not 0 <= cap <= 512):
        raise ValueError("Invalid client AutoARIMA inputs")
    result = control.copy()
    codes, clients = pd.factorize(history["Client"], sort=True)
    groups = len(clients)
    aggregated = np.zeros((groups, y.shape[1]), dtype=np.float64)
    np.add.at(aggregated, codes, y)
    group_previous = np.zeros((groups, HORIZON), dtype=np.float64)
    np.add.at(group_previous, codes, control)
    trailing = aggregated[:, -52:]
    daily_volume = trailing.mean(axis=1)
    activity = (trailing > 0).mean(axis=1)
    eligible = np.flatnonzero((daily_volume >= 5) & (activity >= 0.6))
    order = eligible[np.argsort(-daily_volume[eligible], kind="stable")[:cap]]
    accepted = []
    failed = []
    for group in order:
        try:
            model = make_stats_model("autoarima52")
            forecast = model.forecast(y=np.ascontiguousarray(aggregated[group]), h=HORIZON)
            point = np.asarray(forecast["mean"], dtype=np.float64)
            if point.shape != (HORIZON,) or not np.isfinite(point).all():
                failed.append(int(group))
                continue
            prev = group_previous[group]
            # Predeclared guard against ARIMA implausible seasonal explosions.
            clipped = np.where(prev > 0,
                               np.clip(point, 0.25 * prev, 4.0 * prev), prev)
            rows = np.flatnonzero(codes == group)
            factors = np.divide(clipped, prev, out=np.ones_like(prev), where=prev > 0)
            result[rows] = control[rows] * factors[None, :]
            accepted.append(int(group))
        except (ValueError, ArithmeticError, FloatingPointError, TypeError,
                ZeroDivisionError, np.linalg.LinAlgError):
            failed.append(int(group))
            continue
    if not np.isfinite(result).all() or (result < 0).any():
        raise ArithmeticError("Invalid client aggregate ARIMA allocation")
    return result, {
        "eligible_groups": int(len(eligible)),
        "tried_groups": int(len(order)),
        "fitted_groups": len(accepted),
        "failed_groups": len(failed),
        "affected_series": int(np.isin(codes, accepted).sum()),
        "cap": cap,
    }


def _normalize(candidate: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Reallocate forecast volume while keeping each week's original total."""
    b = np.asarray(base, dtype=np.float64)
    p = np.asarray(candidate, dtype=np.float64)
    if b.shape != p.shape or b.ndim != 2 or b.shape[1] != HORIZON:
        raise ValueError("Mismatched 13-week forecast grids")
    if not np.isfinite(p).all() or (p < 0).any() or not np.isfinite(b).all() or (b < 0).any():
        raise ValueError("Invalid candidate or baseline")
    desired = b.sum(axis=0)
    existing = p.sum(axis=0)
    if np.any((desired > 0) & (existing <= 0)):
        return b.copy()
    ratios = np.divide(desired, existing, out=np.ones_like(desired), where=existing > 0)
    out = p * ratios[None, :]
    if not np.allclose(out.sum(axis=0), desired, atol=1e-5, rtol=1e-10):
        raise ArithmeticError("Week-by-week total changed")
    return out


def _mix(control: np.ndarray, arima: np.ndarray | None = None,
         amsi: np.ndarray | None = None, arima_weight: float = 0.,
         amsi_weight: float = 0.) -> np.ndarray:
    if arima_weight < 0 or amsi_weight < 0 or arima_weight+amsi_weight > 1:
        raise ValueError("Mix weights out of range")
    result = (1 - arima_weight - amsi_weight) * control
    if arima_weight:
        if arima is None or arima.shape != control.shape:
            raise ValueError("ARIMA forecast missing")
        result = result + arima_weight * arima
    if amsi_weight:
        if amsi is None or amsi.shape != control.shape:
            raise ValueError("AMSI forecast missing")
        result = result + amsi_weight * amsi
    return _normalize(result, control)


def candidates(history: pd.DataFrame, control: np.ndarray, *,
               cap: int = 16, activity_trees: int = 100
               ) -> tuple[dict[str, np.ndarray], dict]:
    """Generate fixed policies from pre-origin history, no later labels."""
    y = _observed(history)
    if control.shape != (len(y), HORIZON) or not np.isfinite(control).all() or (control < 0).any():
        raise ValueError("Invalid original incumbent forecast")
    p, pmeta = predict_activity(history, estimators=activity_trees)
    baseline = allocate_weekly(control, p)
    series_arima, fitted = statsforecast_specialist(
        history, "autoarima52", cap=cap, fallback=baseline)
    client_arima, client_meta = client_autoarima(history, baseline, cap=cap)
    amsi_last = amsi_forecast(history, anchor="last", seasonality=0.45, group_shrink=0.55)
    amsi_short = amsi_forecast(history, anchor="short", seasonality=0.60, group_shrink=0.55)
    forecasts = {
        "learned_activity": baseline,
        "arima30_series": _mix(baseline, series_arima, arima_weight=ARIMA_WEIGHT),
        "arima30_client": _mix(baseline, client_arima, arima_weight=ARIMA_WEIGHT),
        "amsi_last10": _mix(baseline, amsi=amsi_last, amsi_weight=AMSI_SMALL_WEIGHT),
        "amsi_last30": _mix(baseline, amsi=amsi_last, amsi_weight=0.30),
        "amsi_short10": _mix(baseline, amsi=amsi_short, amsi_weight=AMSI_SMALL_WEIGHT),
        "arima30_series_amsi_last10": _mix(
            baseline, series_arima, amsi_last, ARIMA_WEIGHT, AMSI_SMALL_WEIGHT),
        "arima30_client_amsi_last10": _mix(
            baseline, client_arima, amsi_last, ARIMA_WEIGHT, AMSI_SMALL_WEIGHT),
    }
    if tuple(forecasts) != CANDIDATES:
        raise AssertionError("Unexpected predeclared policy set")
    for name, x in forecasts.items():
        if x.shape != baseline.shape or not np.isfinite(x).all() or (x < 0).any():
            raise ValueError(f"Invalid forecast for {name}")
    return forecasts, {
        "arima_series_fitted": int(fitted.sum()),
        "arima_client": client_meta,
        "learned_activity": pmeta,
        "sku_cap": cap,
    }


def select_phase0(phase0: pd.DataFrame, *, cap: int = 16,
                  trees: int = 160, activity_trees: int = 100) -> dict:
    y = _observed(phase0)
    if y.shape[1] < ORIGINS[-1]+HORIZON:
        raise ValueError("Incomplete final historical validation horizon")
    losses = {name: {"absolute": 0., "signed": 0.} for name in CANDIDATES}
    total = 0.
    scores_by_origin = {}
    fit_diagnostics = {}
    for origin in ORIGINS:
        past = phase0.iloc[:, :len(KEYS)+origin].copy()
        control = incumbent(past, trees=trees)
        forecast, diagnostics = candidates(past, control, cap=cap, activity_trees=activity_trees)
        target = y[:, origin:origin+HORIZON]
        scores_by_origin[str(origin)] = {name: _metric(pred, target) for name, pred in forecast.items()}
        fit_diagnostics[str(origin)] = {
            "series_fitted": diagnostics["arima_series_fitted"],
            "clients_fitted": diagnostics["arima_client"]["fitted_groups"],
            "client_affected_series": diagnostics["arima_client"]["affected_series"],
        }
        if origin != ORIGINS[-1]:
            total += float(target.sum())
            for name, pred in forecast.items():
                error = pred - target
                losses[name]["absolute"] += float(np.abs(error).sum())
                losses[name]["signed"] += float(error.sum())
        print("VN1_ARIMA_AMSI_FOLD=" + str(origin), flush=True)
    if total <= 0:
        raise ValueError("Phase 0 origin windows have no demand")
    pooled = {name: (v["absolute"]+abs(v["signed"]))/total for name, v in losses.items()}
    winner = min(CANDIDATES, key=lambda n: (pooled[n], CANDIDATES.index(n)))
    holdout = scores_by_origin[str(ORIGINS[-1])]
    accepted = (
        winner != "learned_activity" and
        pooled[winner] <= pooled["learned_activity"]-MIN_GAIN and
        holdout[winner] <= holdout["learned_activity"]-MIN_GAIN
    )
    return {
        "training_origins": list(ORIGINS[:-1]),
        "holdout_origin": ORIGINS[-1],
        "training_scores": pooled,
        "holdout_scores": holdout,
        "winner_on_train_only": winner,
        "selected": winner if accepted else "learned_activity",
        "accepted": bool(accepted),
        "acceptance_absolute_gain": MIN_GAIN,
        "origin_scores": scores_by_origin,
        "arima_fit_counts": fit_diagnostics,
        "caveat": "ARIMA capped by SKU or client; no claim of winner replication",
    }


def freeze(data_dir: Path, output_dir: Path, *, cap: int = 16,
           trees: int = 160, activity_trees: int = 100) -> dict:
    """Persist forecasts for BOTH phases before loading any Phase 2 targets."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Output must be absent or empty")
    phase0 = read_sales(data_dir/"Phase 0 - Sales.csv")
    selection = select_phase0(phase0, cap=cap, trees=trees, activity_trees=activity_trees)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for phase in (1, 2):
        hist = read_history(phase, data_dir)
        control = incumbent(hist, trees=trees)
        preds, details = candidates(hist, control, cap=cap, activity_trees=activity_trees)
        preds["historical_selected"] = preds[selection["selected"]].copy()
        dest = output_dir/f"frozen-phase{phase}.npz"
        np.savez_compressed(dest, **preds)
        future = list(pd.date_range(
            pd.Timestamp(hist.columns[-1])+pd.Timedelta(weeks=1),
            periods=HORIZON, freq="7D").strftime("%Y-%m-%d"))
        manifests[str(phase)] = {
            "series": len(hist), "history_cutoff": str(hist.columns[-1]),
            "dates": future,
            "candidate_names": list(preds),
            "frozen_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            "fit_diagnostics": details,
        }
        print("VN1_ARIMA_AMSI_FROZEN=" + json.dumps({
            "phase": phase, "sha256": manifests[str(phase)]["frozen_sha256"],
            "arima_series_fitted": details["arima_series_fitted"],
            "arima_clients_fitted": details["arima_client"]["fitted_groups"],
        }), flush=True)
    receipt = {
        "history_only_selection": selection,
        "phase_outputs": manifests,
        "cap": cap, "trees": trees, "activity_trees": activity_trees,
        "no_target_phase_labels_loaded_in_freeze": True,
        "retrospective_public_scores_are_not_blind": True,
    }
    (output_dir/"manifest.json").write_text(json.dumps(receipt,indent=2)+"\n")
    return receipt


def score(data_dir: Path, output_dir: Path, *, phase: int,
          truth: Path) -> dict:
    """Read target ONLY after verifying frozen archive integrity."""
    if phase not in (1, 2):
        raise ValueError("Unknown phase")
    receipt = json.loads((output_dir/"manifest.json").read_text())
    manifest = receipt["phase_outputs"][str(phase)]
    bundle = output_dir/f"frozen-phase{phase}.npz"
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != manifest["frozen_sha256"]:
        raise ValueError("Frozen candidate archive was modified")
    actual = read_sales(truth)
    history = read_history(phase, data_dir)
    if (len(actual) != manifest["series"] or
        not actual[KEYS].equals(history[KEYS]) or
        list(actual.columns[3:]) != manifest["dates"]):
        raise ValueError("Phase labels, dates or product keys differ from frozen forecast")
    y = actual.iloc[:,3:].to_numpy(dtype=np.float64)
    with np.load(bundle) as data:
        result = {name: _metric(np.asarray(data[name],dtype=np.float64),y)
                  for name in manifest["candidate_names"]}
    out = {
        "phase": phase, "series": len(y), "horizon": HORIZON,
        "scores": result,
        "selection": receipt["history_only_selection"]["selected"],
        "accepted": receipt["history_only_selection"]["accepted"],
        "frozen_sha256": manifest["frozen_sha256"],
        "source_target": truth.name,
        "retrospective_not_blind": True,
    }
    (output_dir/f"score-phase{phase}.json").write_text(json.dumps(out,indent=2)+"\n")
    print("VN1_ARIMA_AMSI_SCORES="+json.dumps(out,sort_keys=True), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("operation",choices=("freeze","score"))
    ap.add_argument("--data-dir",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--cap",type=int,default=16)
    ap.add_argument("--trees",type=int,default=160)
    ap.add_argument("--activity-trees",type=int,default=100)
    ap.add_argument("--phase",type=int,choices=(1,2),default=1)
    ap.add_argument("--truth",type=Path,default=None)
    args=ap.parse_args()
    if args.operation == "freeze":
        freeze(args.data_dir,args.output_dir,cap=args.cap,trees=args.trees,
               activity_trees=args.activity_trees)
    else:
        if args.truth is None:
            ap.error("--truth is required for scoring")
        score(args.data_dir,args.output_dir,phase=args.phase,truth=args.truth)


if __name__ == "__main__":
    main()
