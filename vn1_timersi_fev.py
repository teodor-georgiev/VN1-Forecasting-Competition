"""Prepare VN1 Phase-0 history for the real TimeRSI FEV task runner.

This is NOT a TimeRSI reimplementation. The optional 'run' operation calls
upstream TimeRSI; its CUDA, Docker, Claude Code, and Vertex requirements must
already be satisfied on an approved research host.

Only public Phase-0 historical weeks are exported. Phase-1/2 labels are never
opened by this adapter. This is an in-sample historical research exercise,
not a blind competition submission or automatic champion promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ("Client", "Warehouse", "Product")
HORIZON = 13
FOLDS = (92, 105, 118, 131, 144, 157)


def read_phase0(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if tuple(frame.columns[:3]) != KEYS or frame.duplicated(list(KEYS)).any():
        raise ValueError("VN1 identifiers must be unique and in official order")
    dates = pd.DatetimeIndex(pd.to_datetime(frame.columns[3:], errors="raise"))
    if len(dates) < FOLDS[-1] + HORIZON:
        raise ValueError("Phase 0 is too short for the reserved historical holdout")
    if not (dates.dayofweek == 0).all() or not (dates[1:] - dates[:-1] == pd.Timedelta(days=7)).all():
        raise ValueError("Expected consecutive Monday sales weeks")
    values = frame.iloc[:, 3:].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Historical sales must be finite and nonnegative")
    return frame


def to_fev_long(frame: pd.DataFrame) -> pd.DataFrame:
    """Long-format FEV data, one observation per series and week, preserving order."""
    dates = pd.DatetimeIndex(pd.to_datetime(frame.columns[3:]))
    n, t = len(frame), len(dates)
    ids = np.array([f"{i:08d}" for i in range(n)], dtype=object)
    values = frame.iloc[:, 3:].to_numpy(dtype=np.float64)
    if t < FOLDS[-1] + HORIZON or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid historical sales")
    return pd.DataFrame({
        "id": np.repeat(ids, t),
        "timestamp": np.tile(dates.to_numpy(), n),
        "target": values.reshape(-1),
    })


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_phase0(data_dir: Path, parquet: Path) -> dict:
    """Create fresh, immutable-by-convention Phase-0 input; refuse overwrites."""
    source = data_dir / "Phase 0 - Sales.csv"
    manifest_path = parquet.with_suffix(".manifest.json")
    if parquet.exists() or manifest_path.exists():
        raise FileExistsError("Research input or manifest already exists; use a new output path")
    frame = read_phase0(source)
    data = to_fev_long(frame)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    # TimeRSI's pinned 'datasets' dependency includes the required PyArrow backend.
    data.to_parquet(parquet, index=False)
    receipt = {
        "experiment": "TimeRSI genuine upstream FEV adapter",
        "scope": "Phase 0 historical only",
        "series": len(frame),
        "weeks": frame.shape[1] - len(KEYS),
        "origins": list(FOLDS),
        "horizon": HORIZON,
        "source_sha256": sha256(source),
        "dataset_sha256": sha256(parquet),
        "phase1_or_phase2_labels_loaded": False,
        "retrospective_not_blind": True,
    }
    manifest_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def build_task(parquet: Path):
    """FEV's native local-parquet Task, with only completed Phase-0 windows."""
    import fev

    return fev.Task(
        dataset_path=str(parquet.resolve()),
        horizon=HORIZON,
        num_windows=len(FOLDS),
        initial_cutoff=FOLDS[0],
        window_step_size=HORIZON,
        min_context_length=65,
        seasonality=52,
        id_column="id",
        timestamp_column="timestamp",
        target="target",
        task_name="vn1-phase0-timersi-historical",
        eval_metric="MASE",  # TimeRSI selection uses its own objective.
    )


def official_fold_components(prediction: np.ndarray, actual: np.ndarray) -> dict:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(actual, dtype=np.float64)
    if (pred.shape != truth.shape or pred.ndim != 2
            or pred.shape[1] != HORIZON or not np.isfinite(pred).all()
            or not np.isfinite(truth).all() or (truth < 0).any()
            or truth.sum() <= 0):
        raise ValueError("Invalid VN1 prediction/actual shape or values")
    # Organizer metric uses global signed error, not per-item bias.
    err = np.maximum(pred, 0.0) - truth
    return {
        "absolute_error": float(np.abs(err).sum()),
        "signed_error": float(err.sum()),
        "actual_units": float(truth.sum()),
        "vn1": float((np.abs(err).sum() + abs(err.sum())) / truth.sum()),
    }


def run_timersi(parquet: Path, checkpoint: Path, output_dir: Path) -> dict:
    """Execute upstream TimeRSI; no provider secrets or checkpoints are stored."""
    manifest = json.loads(parquet.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("origins") != list(FOLDS)
            or manifest.get("dataset_sha256") != sha256(parquet)
            or manifest.get("scope") != "Phase 0 historical only"):
        raise ValueError("Unrecognized or modified historical input")
    project = os.environ.get("TIMERSI_VERTEX_PROJECT")
    if not project:
        raise RuntimeError("TIMERSI_VERTEX_PROJECT must be configured on the approved host")
    if not checkpoint.is_dir():
        raise FileNotFoundError("TimeRSI requires an existing local TimesFM-3 checkpoint")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("TimeRSI output directory must be empty")
    # Deliberately import only inside the optional run operation.
    from timersi_u1.fev_adapter import TimeRSIConfig
    from timersi_u1.task_process import fit_predict

    task = build_task(parquet)
    forecasts, timings, run_dir = fit_predict(
        task,
        TimeRSIConfig(
            checkpoint_path=str(checkpoint.resolve()),
            vertex_project=project,
            run_root=str(output_dir.resolve()),
        ),
    )
    if len(forecasts) != len(FOLDS):
        raise ValueError("TimeRSI did not return every declared historical fold")
    fold_scores = []
    abs_total = bias_total = volume_total = 0.0
    for index, origin in enumerate(FOLDS):
        window = task.get_window(index)
        actual = np.asarray(window.get_ground_truth()["target"], dtype=np.float64)
        predicted = np.asarray(forecasts[index]["target"]["predictions"], dtype=np.float64)
        score = official_fold_components(predicted, actual)
        score["origin"] = origin
        fold_scores.append(score)
        abs_total += score["absolute_error"]
        bias_total += score["signed_error"]
        volume_total += score["actual_units"]
    receipt = {
        "model": "TimeRSI upstream, frozen TimesFM-3 with agent research",
        "scope": "Phase 0 historical only",
        "fold_scores": fold_scores,
        "pooled_vn1": (abs_total + abs(bias_total)) / volume_total,
        "timings": timings,
        "run_directory": str(run_dir),
        "dataset_sha256": manifest["dataset_sha256"],
        "retrospective_not_blind": True,
        "comparable_to_phase2_champion": False,
    }
    path = output_dir / "vn1_official_historical_scores.json"
    if path.exists():
        raise FileExistsError("Official historical receipt already exists")
    path.write_text(json.dumps(receipt, indent=2, default=str) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("export", "run"))
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.operation == "export":
        result = export_phase0(args.data_dir, args.parquet)
    else:
        if args.checkpoint is None or args.out is None:
            parser.error("run requires --checkpoint and --out")
        result = run_timersi(args.parquet, args.checkpoint, args.out)
    print(json.dumps(result, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
