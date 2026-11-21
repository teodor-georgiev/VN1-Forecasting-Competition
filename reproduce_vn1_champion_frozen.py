"""Independently score the immutable VN1 ARIMA+AMSI champion Phase 2 archive.

READ-ONLY: no model training, tuning, writing predictions, external calls, or
retrospective parameter changes. This script never serializes actual sales,
series identifiers or per-series forecasts into its aggregate report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_CHAMPION_SHA256 = (
    "33cdb93ccc2415e9c6697e1074a0e4ecb2ad6066d473480b79a3582aa1122101"
)
EXPECTED_PHASE2_TRUTH_GIT_BLOB = "714205238d3fcebe8a82b0cec97b28f375914d3b"
EXPECTED_CHAMPION_SCORE = 0.4693382738889349
EXPECTED_SELECTED = "arima30_client_amsi_last10"
EXPECTED_SERIES = 15053
EXPECTED_HORIZON = 13
EXPECTED_TOTAL_UNITS = 3242739
KEYS = ("Client", "Warehouse", "Product")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validated_sales(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if tuple(frame.columns[:3]) != KEYS or frame.duplicated(list(KEYS)).any():
        raise ValueError("Missing, reordered or duplicate source identifiers")
    if frame.shape[0] != EXPECTED_SERIES:
        raise ValueError("Unexpected VN1 series count")
    values = frame.iloc[:, 3:].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Sales contain invalid or negative quantities")
    return frame


def require_aligned_labels(
    phase0: pd.DataFrame, phase1: pd.DataFrame, phase2: pd.DataFrame,
    expected_dates: list[str],
) -> None:
    for frame in (phase1, phase2):
        if not phase0[list(KEYS)].equals(frame[list(KEYS)]):
            raise ValueError("Phase 0/1/2 series identifiers or order disagree")
    if (len(expected_dates) != EXPECTED_HORIZON
            or list(phase2.columns[3:]) != expected_dates):
        raise ValueError("Published truth and frozen forecast dates do not align")
    history_dates = pd.to_datetime([*phase0.columns[3:], *phase1.columns[3:]])
    if not history_dates.is_monotonic_increasing or history_dates.has_duplicates:
        raise ValueError("History timestamps duplicate or are out of order")
    if pd.Timestamp(expected_dates[0]) != history_dates[-1] + pd.Timedelta(weeks=1):
        raise ValueError("Frozen target begins at an incorrect forecast origin")


def score_official(actual: np.ndarray, prediction: np.ndarray) -> dict:
    """Official VN1: WAPE + absolute GLOBAL signed bias, not mean per-series bias."""
    actual = np.asarray(actual, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if (actual.ndim != 2 or actual.shape != prediction.shape
            or not np.isfinite(actual).all() or not np.isfinite(prediction).all()
            or (actual < 0).any() or (prediction < 0).any()):
        raise ValueError("Invalid score matrices")
    denominator = float(np.sum(actual, dtype=np.float64))
    if denominator <= 0:
        raise ValueError("Cannot score zero total actual demand")
    error = prediction - actual
    absolute = float(np.sum(np.abs(error), dtype=np.float64))
    signed = float(np.sum(error, dtype=np.float64))
    return {
        "wape": absolute / denominator,
        "pbias": signed / denominator,
        "vn1": (absolute + abs(signed)) / denominator,
        "observations": int(actual.size),
        "actual_units": denominator,
    }


def replay(
    frozen_dir: Path,
    truth: Path,
    phase0_path: Path,
    phase1_path: Path,
    *,
    expected_frozen_sha256: str = EXPECTED_CHAMPION_SHA256,
) -> dict:
    manifest = json.loads((frozen_dir / "manifest.json").read_text(encoding="utf-8"))
    selection = manifest["history_only_selection"]
    frozen = manifest["phase_outputs"]["2"]
    if (selection["selected"] != EXPECTED_SELECTED
            or frozen["series"] != EXPECTED_SERIES
            or len(frozen["dates"]) != EXPECTED_HORIZON
            or EXPECTED_SELECTED not in frozen["candidate_names"]
            or "historical_selected" not in frozen["candidate_names"]):
        raise ValueError("Manifest is not the reviewed VN1 champion selection")
    path = frozen_dir / "frozen-phase2.npz"
    sha = sha256_file(path)
    if sha != expected_frozen_sha256 or sha != frozen["frozen_sha256"]:
        raise ValueError("Champion archive hash differs from pinned reviewed source")
    phase0 = validated_sales(phase0_path)
    phase1 = validated_sales(phase1_path)
    phase2 = validated_sales(truth)
    require_aligned_labels(phase0, phase1, phase2, frozen["dates"])
    actual = phase2.iloc[:, 3:].to_numpy(dtype=np.float64)
    if actual.shape != (EXPECTED_SERIES, EXPECTED_HORIZON):
        raise ValueError("Phase 2 actuals have incorrect shape")
    if float(np.sum(actual)) != EXPECTED_TOTAL_UNITS:
        raise ValueError("Official Phase 2 actual demand checksum differs")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(frozen["candidate_names"]):
            raise ValueError("Frozen candidate names differ from manifest")
        chosen = np.asarray(archive["historical_selected"], dtype=np.float64)
        exact = np.asarray(archive[EXPECTED_SELECTED], dtype=np.float64)
        if not np.array_equal(chosen, exact):
            raise ValueError("Historically selected forecast differs from named champion")
        values = score_official(actual, chosen)
    if abs(values["vn1"] - EXPECTED_CHAMPION_SCORE) > 1e-10:
        raise AssertionError(
            "Original retrospective champion did not reproduce: "
            + str(values["vn1"])
        )
    return {
        "status": "retrospective_reproduction_passed",
        "model": EXPECTED_SELECTED,
        "score": values,
        "target": "official_published_phase2",
        "frozen_sha256": sha,
        "truth_expected_git_blob": EXPECTED_PHASE2_TRUTH_GIT_BLOB,
        "series": EXPECTED_SERIES,
        "horizon": EXPECTED_HORIZON,
        "source": "PR17 frozen 2026-11-11; pinned published SupChains VN1 Phase 2",
        "no_retraining": True,
        "no_phase2_model_selection": True,
        "is_blind_competition_submission": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--phase0", type=Path, required=True)
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = replay(
        args.frozen_dir, args.truth, args.phase0, args.phase1,
    )
    if args.report.exists():
        raise FileExistsError("Refusing to overwrite existing VN1 score evidence")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("VN1_CHAMPION_REPRODUCTION=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
