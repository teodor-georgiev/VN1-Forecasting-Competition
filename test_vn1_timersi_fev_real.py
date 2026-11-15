"""Full VN1 Phase-0 -> actual pinned FEV integration, without TimeRSI credentials.

Run on a fresh GitHub-hosted CPU runner: this uses the real 15,053-series
dataset and real fev.Task implementation, rather than mocked unit objects.
Only Phase-0 data is opened. No forecasts from TimeRSI are claimed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vn1_timersi_fev import FOLDS, HORIZON, build_task, export_phase0, official_fold_components, read_phase0


def test_full_phase0_fev_windows_align_to_all_official_series(tmp_path: Path):
    original = read_phase0(Path("Phase 0 - Sales.csv"))
    parquet = tmp_path / "vn1-phase0.parquet"
    manifest = export_phase0(Path("."), parquet)

    assert manifest["series"] == len(original) == 15_053
    assert manifest["weeks"] == 170
    assert manifest["origins"] == list(FOLDS)
    assert manifest["phase1_or_phase2_labels_loaded"] is False
    assert json.loads(parquet.with_suffix(".manifest.json").read_text()) == manifest

    task = build_task(parquet)
    assert tuple(task.cutoffs) == FOLDS
    expected_ids = [f"{i:08d}" for i in range(len(original))]
    volume_by_fold = {}
    for index, origin in enumerate(FOLDS):
        window = task.get_window(index)
        truth = window.get_ground_truth()
        assert list(truth["id"]) == expected_ids, f"FEV series order changed in fold {index}"
        actual = np.asarray(truth["target"], dtype=np.float64)
        expected = original.iloc[:, 3 + origin:3 + origin + HORIZON].to_numpy(dtype=np.float64)
        assert actual.shape == expected.shape == (15_053, HORIZON)
        np.testing.assert_array_equal(actual, expected, err_msg=f"FEV target mismatch in fold {index}")
        score = official_fold_components(actual, expected)
        assert score["vn1"] == 0.0
        volume_by_fold[str(origin)] = score["actual_units"]

    assert all(value > 0 for value in volume_by_fold.values())
    print("VN1_TIMERSI_REAL_FEV_PHASE0_PARITY=PASS")
    print(json.dumps({"series": len(original), "origins": list(FOLDS),
                      "horizon": HORIZON, "observed_units_by_origin": volume_by_fold},
                     sort_keys=True))
