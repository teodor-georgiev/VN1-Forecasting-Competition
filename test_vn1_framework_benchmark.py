"""Synthetic, dependency-light contract tests for VN1 library comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vn1_framework_benchmark import HORIZON, choose_sample, load_origin, vn1_score


def test_official_score_includes_global_bias():
    actual = np.ones((2, HORIZON))
    pred = actual.copy()
    pred[0, 0] += 2.0
    pred[1, 0] -= 1.0
    result = vn1_score(pred, actual)
    assert result["vn1"] == pytest.approx(4.0 / 26.0)
    assert result["signed_relative_bias"] == pytest.approx(1.0 / 26.0)


def test_invalid_shape_and_deterministic_sample():
    with pytest.raises(ValueError):
        vn1_score(np.ones((2, 12)), np.ones((2, 12)))
    sample = choose_sample(100, 20, 11)
    assert len(sample) == len(set(sample)) == 20
    assert (sample == choose_sample(100, 20, 11)).all()
    assert (choose_sample(5, 0, 11) == np.arange(5)).all()


def test_historical_origin_does_not_train_on_future(tmp_path):
    dates = pd.date_range("2020-07-06", periods=170, freq="W-MON")
    keys = pd.DataFrame({"Client": [1, 2], "Warehouse": [1, 1], "Product": [9, 10]})
    matrix = pd.DataFrame(np.tile(np.arange(170, dtype=float), (2, 1))
                          + np.array([[0.], [1.]]), columns=dates.strftime("%Y-%m-%d"))
    pd.concat([keys, matrix], axis=1).to_csv(tmp_path / "Phase 0 - Sales.csv", index=False)
    ids, history, truth, old_dates, forecast_dates = load_origin(tmp_path, 157, False)
    assert len(ids) == 2
    assert history.shape == (2, 157)
    assert truth.shape == (2, 13)
    assert history[0, -1] == 156
    assert truth[0, 0] == 157
    assert old_dates[-1] + pd.Timedelta(weeks=1) == forecast_dates[0]
    with pytest.raises(ValueError):
        load_origin(tmp_path, 158, False)


def test_phase2_freeze_then_score_and_detect_tampering(tmp_path, monkeypatch):
    """No Phase 2 truth file is available when the models generate forecasts."""
    import json
    from types import SimpleNamespace
    import vn1_framework_benchmark as bench

    dates = pd.date_range("2020-07-06", periods=196, freq="W-MON")
    keys = pd.DataFrame({"Client": [1, 2], "Warehouse": [1, 1], "Product": [9, 10]})
    values = np.tile(np.arange(196, dtype=float), (2, 1))
    values[1] += 1
    for name, left, right in (
        ("Phase 0 - Sales.csv", 0, 170),
        ("Phase 1 - Sales.csv", 170, 183),
    ):
        frame = pd.concat([
            keys, pd.DataFrame(values[:, left:right],
                               columns=dates[left:right].strftime("%Y-%m-%d"))
        ], axis=1)
        frame.to_csv(tmp_path / name, index=False)

    def mock_sktime(history, observed_dates):
        assert history.shape == (2, 183)
        assert len(observed_dates) == 183
        return np.repeat(history[:, -13:].mean(axis=1, keepdims=True), HORIZON, axis=1)

    monkeypatch.setattr(bench, "sktime_forecast", mock_sktime)
    output = tmp_path / "frozen"
    bench.run(SimpleNamespace(data_dir=tmp_path, output_dir=output, model="sktime",
                              sample_size=0, seed=42, origin=157, phase2=True))
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "frozen_unscored"
    assert not (tmp_path / "Phase 2 - Sales.csv").exists()
    assert "vn1" not in json.loads((output / "scores.json").read_text())[0]

    phase2 = pd.concat([keys, pd.DataFrame(values[:, 183:196],
                       columns=dates[183:196].strftime("%Y-%m-%d"))], axis=1)
    phase2.to_csv(tmp_path / "Phase 2 - Sales.csv", index=False)
    receipts = bench.score_phase2(tmp_path, output)
    assert len(receipts) == 1
    assert receipts[0]["framework"] == "sktime"
    assert receipts[0]["vn1"] > 0
    assert (output / "phase2_scores.json").is_file()
    with pytest.raises(FileExistsError):
        bench.score_phase2(tmp_path, output)

    with (output / "sktime_forecast.csv").open("a") as fh:
        fh.write("\\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        bench.score_phase2(tmp_path, output)
