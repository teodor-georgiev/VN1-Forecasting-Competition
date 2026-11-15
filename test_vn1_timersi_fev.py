"""Dependency-light contracts for the isolated VN1 -> TimeRSI/FEV adapter."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from vn1_timersi_fev import (
    FOLDS,
    HORIZON,
    build_task,
    export_phase0,
    official_fold_components,
    read_phase0,
    to_fev_long,
)


def phase0_fixture(n: int = 2) -> pd.DataFrame:
    dates = pd.date_range("2020-07-06", periods=170, freq="W-MON")
    keys = pd.DataFrame({
        "Client": np.arange(1, n + 1),
        "Warehouse": np.ones(n, dtype=int),
        "Product": np.arange(10, 10 + n),
    })
    values = pd.DataFrame(
        np.arange(n * 170, dtype=float).reshape(n, 170),
        columns=dates.strftime("%Y-%m-%d"),
    )
    return pd.concat([keys, values], axis=1)


def test_phase0_long_format_preserves_all_observations_and_series_order():
    frame = phase0_fixture()
    long = to_fev_long(frame)
    assert list(long.columns) == ["id", "timestamp", "target"]
    assert len(long) == 2 * 170
    assert long["id"].iloc[0] == "00000000"
    assert long["id"].iloc[170] == "00000001"
    assert long["timestamp"].iloc[0] == pd.Timestamp("2020-07-06")
    assert long["timestamp"].iloc[170] == pd.Timestamp("2020-07-06")
    assert long["target"].iloc[0] == 0
    assert long["target"].iloc[170] == 170
    assert FOLDS == (92, 105, 118, 131, 144, 157)
    assert FOLDS[-1] + HORIZON == 170


def test_reject_missing_week_and_duplicate_ids(tmp_path):
    frame = phase0_fixture()
    frame.iloc[:, 3] = 0
    frame = frame.rename(columns={"2020-07-13": "2020-07-14"})
    file = tmp_path / "Phase 0 - Sales.csv"
    frame.to_csv(file, index=False)
    with pytest.raises(ValueError, match="consecutive Monday"):
        read_phase0(file)
    dup = phase0_fixture()
    dup.loc[1, ["Client", "Warehouse", "Product"]] = dup.loc[
        0, ["Client", "Warehouse", "Product"]
    ].to_numpy()
    dup.to_csv(file, index=False)
    with pytest.raises(ValueError, match="unique"):
        read_phase0(file)


def test_organizer_vn1_formula_penalizes_aggregate_signed_bias():
    actual = np.ones((2, HORIZON), dtype=float)
    pred = actual.copy()
    pred[0, 0] += 2
    pred[1, 0] -= 1
    score = official_fold_components(pred, actual)
    assert score["absolute_error"] == pytest.approx(3)
    assert score["signed_error"] == pytest.approx(1)
    assert score["vn1"] == pytest.approx(4 / 26)
    with pytest.raises(ValueError):
        official_fold_components(pred[:, :-1], actual)


def test_export_never_reads_phase1_or_phase2(tmp_path, monkeypatch):
    phase0_fixture().to_csv(tmp_path / "Phase 0 - Sales.csv", index=False)
    (tmp_path / "Phase 2 - Sales.csv").write_text("unusable confidential labels")
    original_read = pd.read_csv

    def guarded_read(path, *args, **kwargs):
        if "Phase 1" in str(path) or "Phase 2" in str(path):
            raise AssertionError("future labels must never be opened")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded_read)
    # Keep this unit test independent of the optional PyArrow writer.
    def fake_parquet(self, path, index=False):
        assert index is False
        assert len(self) == 340
        Path(path).write_bytes(b"synthetic parquet fixture")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_parquet)
    parquet = tmp_path / "historical.parquet"
    receipt = export_phase0(tmp_path, parquet)
    assert receipt["phase1_or_phase2_labels_loaded"] is False
    assert receipt["origins"] == list(FOLDS)
    assert receipt["weeks"] == 170
    assert receipt["series"] == 2
    manifest = json.loads(parquet.with_suffix(".manifest.json").read_text())
    assert manifest == receipt
    with pytest.raises(FileExistsError):
        export_phase0(tmp_path, parquet)


def test_fev_task_uses_only_completed_historical_origins(monkeypatch, tmp_path):
    captured = {}
    def fake_task(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "fev", SimpleNamespace(Task=fake_task))
    parquet = tmp_path / "historical.parquet"
    task = build_task(parquet)
    assert task.num_windows == 6
    assert task.initial_cutoff == 92
    assert task.window_step_size == 13
    assert task.horizon == 13
    assert task.min_context_length == 65
    assert task.target == "target"
    assert task.dataset_path == str(parquet.resolve())
