"""Historically fitted persistent-bias specialist for the VN1 ARIMA+AMSI champion.

This is a *separate* research candidate, not a replacement of the champion.
Corrections are estimated only from completed out-of-fold champion errors,
then evaluated on later nonoverlapping origins. Phase 1/2 targets are never
read here; selection uses Phase 0 training origins and a later holdout only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import vn1_champion_residual_oof as base
from vn1_amsi_zero_rule_ablation import _metric
from vn1_from_scratch import HORIZON, KEYS, read_sales

POLICIES = (
    "champion",
    "group_active20",
    "group_active35",
    "sku_stable35",
    "hybrid20",
    "group_unscaled25",
)
MIN_GAIN = 0.001
MIN_TRAIN_FOLD_WINS = 2


def estimate_effects(history: pd.DataFrame, records: list[dict]) -> dict[str, np.ndarray]:
    """Estimate group and persistent per-SKU errors without later observations.

    The newest three completed OOF windows are sufficient for this deliberately
    low-capacity correction. Group estimates are pooled by client and warehouse;
    per-SKU estimates require matching residual direction in the last two folds.
    Eligibility and top-volume selection use only sales observed at cutoff.
    """
    y = base._history_matrix(history)
    n, cutoff = y.shape
    if not records:
        raise ValueError("At least one completed OOF forecast is required")
    origins = [int(r["origin"]) for r in records]
    if origins != sorted(set(origins)):
        raise ValueError("OOF records must be in unique chronological order")
    if any(origin < base.OOF_START or origin + HORIZON > cutoff for origin in origins):
        raise ValueError("OOF record would contain future target labels")
    panel_p, panel_e = [], []
    for record in records[-3:]:
        pred = np.asarray(record["champion"], dtype=np.float64)
        truth = np.asarray(record["truth"], dtype=np.float64)
        if pred.shape != (n, HORIZON) or truth.shape != (n, HORIZON):
            raise ValueError("OOF champion/truth shape differs from history")
        if not np.isfinite(pred).all() or not np.isfinite(truth).all():
            raise ValueError("OOF contains nonfinite values")
        if (pred < 0).any() or (truth < 0).any():
            raise ValueError("OOF requires nonnegative sales")
        panel_p.append(pred.sum(axis=1))
        panel_e.append((truth - pred).sum(axis=1))

    predicted = np.stack(panel_p)
    error = np.stack(panel_e)
    sum_pred = predicted.sum(axis=0)
    sum_error = error.sum(axis=0)
    volume = y[:, -13:].mean(axis=1)
    active = base._active_mask(y)

    top_volume = np.zeros(n, dtype=bool)
    ranked = np.argsort(-volume, kind="stable")
    top_volume[ranked[:max(1, int(np.ceil(0.05 * n)))]] = True
    top_volume &= active

    # Large group pseudo-volume keeps small groups near zero correction.
    group_effects = []
    for key in ("Client", "Warehouse"):
        codes, _ = pd.factorize(history[key], sort=True)
        grouped_error = np.bincount(codes, weights=sum_error)
        grouped_forecast = np.bincount(codes, weights=sum_pred)
        grouped_rate = np.clip(
            grouped_error / (grouped_forecast + 2500.0), -0.25, 0.25)
        group_effects.append(grouped_rate[codes])
    group = (group_effects[0] + group_effects[1]) / 2.0

    # Demand changes sign between completed origins are not a reliable
    # per-product bias. Shrink pooled residuals by two future horizons.
    series = np.zeros(n, dtype=np.float64)
    if len(error) >= 2:
        last, previous = error[-1], error[-2]
        sign_stable = (last * previous > 0)
        material_last = np.abs(last) >= 0.025 * np.maximum(predicted[-1], 1.0)
        material_prev = np.abs(previous) >= 0.025 * np.maximum(predicted[-2], 1.0)
        trusted = sign_stable & material_last & material_prev
        prior_volume = 2.0 * HORIZON * np.maximum(volume, 1.0)
        series = np.clip(sum_error / (sum_pred + prior_volume), -0.25, 0.25)
        series[~trusted] = 0.0

    group[~active] = 0.0
    series[~top_volume] = 0.0
    return {
        "group": group,
        "series": series,
        "active": active,
        "top_volume": top_volume,
    }


def candidate_forecasts(
    history: pd.DataFrame,
    champion: np.ndarray,
    records: list[dict],
) -> tuple[dict[str, np.ndarray], dict]:
    """Predeclared bounded correction policies, champion retained unchanged."""
    y = base._history_matrix(history)
    control = np.asarray(champion, dtype=np.float64)
    if control.shape != (len(y), HORIZON):
        raise ValueError("Champion shape differs from forecast grid")
    if not np.isfinite(control).all() or (control < 0).any():
        raise ValueError("Champion contains invalid forecasts")
    effects = estimate_effects(history, records)
    group = effects["group"]
    series = effects["series"]

    def adjusted(rate: np.ndarray, *, conserve: bool = True) -> np.ndarray:
        # Renormalize ONLY the active pool. Inactive products retain exactly
        # their original forecasts, even after volume conservation.
        modifiable = effects["active"]
        candidate = control.copy()
        candidate[modifiable] = np.maximum(
            0.0, control[modifiable] * (1.0 + rate[modifiable, None]))
        if conserve and modifiable.any():
            desired = control[modifiable].sum(axis=0)
            current = candidate[modifiable].sum(axis=0)
            if np.any((desired > 0) & (current <= 0)):
                return control.copy()
            scale = np.divide(
                desired, current, out=np.ones_like(current), where=current > 0)
            candidate[modifiable] *= scale[None, :]
            if not np.allclose(candidate.sum(axis=0), control.sum(axis=0),
                               atol=1e-7, rtol=1e-10):
                raise ArithmeticError("Champion aggregate demand changed")
        return candidate

    choices = {
        "champion": control.copy(),
        "group_active20": adjusted(0.20 * group),
        "group_active35": adjusted(0.35 * group),
        "sku_stable35": adjusted(0.35 * series),
        "hybrid20": adjusted(0.20 * group + 0.20 * series),
        # Let a consistent group bias also compete on the official bias penalty.
        "group_unscaled25": adjusted(0.25 * group, conserve=False),
    }
    if tuple(choices) != POLICIES:
        raise AssertionError("Candidate set changed")
    return choices, {
        "oof_origins": [int(record["origin"]) for record in records],
        "active_series": int(effects["active"].sum()),
        "top_volume_series": int(effects["top_volume"].sum()),
        "persistent_sku_series": int(np.count_nonzero(series)),
        "group_effect_nonzero": int(np.count_nonzero(group)),
    }


def historical_validation(
    phase0: pd.DataFrame,
    *,
    trees: int = 160,
    activity_trees: int = 100,
    client_cap: int = 16,
) -> dict:
    """Train on rolling Phase-0 origins; accept only if later holdout wins."""
    y = base._history_matrix(phase0)
    if y.shape[1] < base.HOLDOUT_ORIGIN + HORIZON:
        raise ValueError("Incomplete historical holdout")
    losses = {name: {"absolute": 0.0, "signed": 0.0} for name in POLICIES}
    win_counts = {name: 0 for name in POLICIES}
    fold_scores, diagnostics = {}, {}
    total = 0.0
    cache: dict[int, dict] = {}
    for origin in (*base.TRAIN_EVAL_ORIGINS, base.HOLDOUT_ORIGIN):
        past = phase0.iloc[:, :len(KEYS) + origin].copy()
        records = base._completed_records(
            phase0, origin, cache, trees=trees,
            activity_trees=activity_trees, client_cap=client_cap)
        champ, _ = base.champion_forecast(
            past, trees=trees, activity_trees=activity_trees, client_cap=client_cap)
        forecasts, details = candidate_forecasts(past, champ, records)
        truth = y[:, origin:origin + HORIZON]
        if truth.shape != (len(y), HORIZON):
            raise ValueError("Incomplete validation truth")
        fold_scores[str(origin)] = {name: _metric(pred, truth)
                                    for name, pred in forecasts.items()}
        diagnostics[str(origin)] = details
        if origin != base.HOLDOUT_ORIGIN:
            total += float(truth.sum())
            for name, forecast in forecasts.items():
                diff = forecast - truth
                losses[name]["absolute"] += float(np.abs(diff).sum())
                losses[name]["signed"] += float(diff.sum())
                if fold_scores[str(origin)][name] < fold_scores[str(origin)]["champion"]:
                    win_counts[name] += 1
        print("VN1_PERSISTENT_BIAS_FOLD=" + str(origin), flush=True)
    if total <= 0:
        raise ValueError("Historical training demand is zero")
    scores = {name: (v["absolute"] + abs(v["signed"])) / total
              for name, v in losses.items()}
    winner = min(POLICIES, key=lambda name: (scores[name], POLICIES.index(name)))
    holdout = fold_scores[str(base.HOLDOUT_ORIGIN)]
    accepted = (
        winner != "champion"
        and scores[winner] <= scores["champion"] - MIN_GAIN
        and holdout[winner] <= holdout["champion"] - MIN_GAIN
        and win_counts[winner] >= MIN_TRAIN_FOLD_WINS
    )
    return {
        "training_origins": list(base.TRAIN_EVAL_ORIGINS),
        "holdout_origin": base.HOLDOUT_ORIGIN,
        "training_scores": scores,
        "holdout_scores": holdout,
        "fold_scores": fold_scores,
        "train_fold_wins": win_counts,
        "winner_training_only": winner,
        "selected": winner if accepted else "champion",
        "accepted": bool(accepted),
        "min_gain": MIN_GAIN,
        "candidate_policies": list(POLICIES),
        "diagnostics": diagnostics,
        "target_phase2_seen": False,
        "retrospective_not_blind": True,
    }



def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_selection(report_path: Path) -> dict:
    receipt = json.loads(report_path.read_text())
    if (receipt.get("training_origins") != list(base.TRAIN_EVAL_ORIGINS)
            or receipt.get("holdout_origin") != base.HOLDOUT_ORIGIN
            or receipt.get("candidate_policies") != list(POLICIES)):
        raise ValueError("Unexpected historical selection lineage")
    winner = receipt.get("winner_training_only")
    if winner not in POLICIES:
        raise ValueError("Unrecognized historical winner")
    train, holdout, wins = (
        receipt["training_scores"], receipt["holdout_scores"],
        receipt["train_fold_wins"])
    expected = (
        winner != "champion"
        and train[winner] <= train["champion"] - MIN_GAIN
        and holdout[winner] <= holdout["champion"] - MIN_GAIN
        and wins[winner] >= MIN_TRAIN_FOLD_WINS
    )
    if receipt.get("accepted") is not bool(expected):
        raise ValueError("Historical acceptance gate does not match metrics")
    if receipt.get("selected") != (winner if expected else "champion"):
        raise ValueError("Historical selection inconsistent with gate")
    return receipt


def freeze(
    data_dir: Path,
    output_dir: Path,
    receipt_path: Path,
    *,
    trees: int = 160,
    activity_trees: int = 100,
    client_cap: int = 16,
) -> dict:
    """Freeze one historically selected Phase 2 challenger BEFORE reading targets."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Never overwrite a frozen forecast")
    selection = _verified_selection(receipt_path)
    from vn1_lgbm_ensemble import read_history

    # read_history(2) reads ONLY Phase 0 and Phase 1 sales.
    history = read_history(2, data_dir)
    cutoff = base._history_matrix(history).shape[1]
    records = base._completed_records(
        history, cutoff, {}, trees=trees,
        activity_trees=activity_trees, client_cap=client_cap)
    champion, details = base.champion_forecast(
        history, trees=trees,
        activity_trees=activity_trees, client_cap=client_cap)
    policies, diagnostic = candidate_forecasts(history, champion, records)
    picked = policies[selection["selected"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "forecast.npz"
    np.savez_compressed(archive, champion=champion, historical_selected=picked)
    dates = list(pd.date_range(
        pd.Timestamp(history.columns[-1]) + pd.Timedelta(weeks=1),
        periods=HORIZON, freq="7D").strftime("%Y-%m-%d"))
    manifest = {
        "phase": 2,
        "selected": selection["selected"],
        "accepted": selection["accepted"],
        "series": len(history),
        "horizon": HORIZON,
        "dates": dates,
        "forecast_sha256": _sha256(archive),
        "selection_sha256": _sha256(receipt_path),
        "phase0_sales_sha256": _sha256(data_dir / "Phase 0 - Sales.csv"),
        "phase1_sales_sha256": _sha256(data_dir / "Phase 1 - Sales.csv"),
        "source_sha256": _sha256(Path(__file__)),
        "completed_oof_origins": diagnostic["oof_origins"],
        "champion": base.CHAMPION_NAME,
        "no_target_labels_loaded_during_freeze": True,
        "retrospective_not_blind": True,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("VN1_PERSISTENT_BIAS_FROZEN=" + json.dumps({
        "selected": manifest["selected"],
        "forecast_sha256": manifest["forecast_sha256"],
        "completed_oof_origins": manifest["completed_oof_origins"],
    }, sort_keys=True), flush=True)
    return manifest


def score(data_dir: Path, output_dir: Path, truth_path: Path) -> dict:
    """Verify frozen forecasts before opening Phase 2 labels for final scoring."""
    manifest = json.loads((output_dir / "manifest.json").read_text())
    archive = output_dir / "forecast.npz"
    if _sha256(archive) != manifest["forecast_sha256"]:
        raise ValueError("Frozen forecast SHA-256 mismatch")
    if (_sha256(data_dir / "Phase 0 - Sales.csv") != manifest["phase0_sales_sha256"]
            or _sha256(data_dir / "Phase 1 - Sales.csv") != manifest["phase1_sales_sha256"]):
        raise ValueError("Historical input SHA-256 mismatch")
    from vn1_lgbm_ensemble import read_history

    historical = read_history(2, data_dir)
    actual = read_sales(truth_path)  # first and only access to Phase 2 labels
    if (manifest["phase"] != 2
            or len(actual) != manifest["series"]
            or not historical[KEYS].equals(actual[KEYS])
            or list(actual.columns[3:]) != manifest["dates"]):
        raise ValueError("Phase 2 target alignment, IDs or dates differ")
    target = actual.iloc[:, 3:].to_numpy(dtype=np.float64)
    with np.load(archive) as frozen:
        scores = {
            name: float(_metric(np.asarray(frozen[name], dtype=np.float64), target))
            for name in ("champion", "historical_selected")
        }
    receipt = {
        "phase": 2,
        "series": manifest["series"],
        "horizon": HORIZON,
        "selection": manifest["selected"],
        "scores": scores,
        "improvement": scores["champion"] - scores["historical_selected"],
        "forecast_sha256": manifest["forecast_sha256"],
        "retrospective_not_blind": True,
    }
    (output_dir / "score-phase2.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("VN1_PERSISTENT_BIAS_PHASE2_SCORES=" +
          json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=("historical", "freeze", "score"),
                        default="historical")
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("results/persistent-bias.json"))
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--trees", type=int, default=160)
    parser.add_argument("--activity-trees", type=int, default=100)
    parser.add_argument("--client-cap", type=int, default=16)
    args = parser.parse_args()
    if args.trees < 1 or args.activity_trees < 1 or not 0 <= args.client_cap <= 512:
        parser.error("Invalid model capacity")
    if args.operation == "freeze":
        if args.selection is None:
            parser.error("--selection is required when freezing")
        freeze(args.data_dir, args.output, args.selection,
               trees=args.trees, activity_trees=args.activity_trees,
               client_cap=args.client_cap)
        return
    if args.operation == "score":
        if args.truth is None:
            parser.error("--truth is required when scoring")
        score(args.data_dir, args.output, args.truth)
        return
    phase0 = read_sales(args.data_dir / "Phase 0 - Sales.csv")
    receipt = historical_validation(
        phase0, trees=args.trees,
        activity_trees=args.activity_trees, client_cap=args.client_cap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("VN1_PERSISTENT_BIAS_RESULT=" + json.dumps({
        "selected": receipt["selected"],
        "accepted": receipt["accepted"],
        "training_scores": receipt["training_scores"],
        "holdout_scores": receipt["holdout_scores"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
