"""Independent golden and adversarial tests for frozen VN1 scoring."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import reproduce_vn1_champion_frozen as champion


def test_vn1_global_bias_is_not_mean_series_bias():
    actual = np.array([[10., 20.], [0., 10.]])
    predicted = np.array([[12., 18.], [2., 8.]])
    scored = champion.score_official(actual, predicted)
    assert scored["observations"] == 4
    assert scored["actual_units"] == 40
    assert scored["wape"] == pytest.approx(0.2)
    assert scored["pbias"] == 0
    assert scored["vn1"] == pytest.approx(0.2)


def test_signed_global_bias_is_added_to_absolute_errors():
    actual = np.array([[10., 10.]])
    predicted = np.array([[12., 12.]])
    scored = champion.score_official(actual, predicted)
    assert scored["wape"] == pytest.approx(.2)
    assert scored["pbias"] == pytest.approx(.2)
    assert scored["vn1"] == pytest.approx(.4)


@pytest.mark.parametrize(
    ("actual", "forecast"),
    [
        (np.array([[1.]]), np.ones((1, 2))),
        (np.array([[1.]]), np.array([[-1.]])),
        (np.array([[1.]]), np.array([[np.nan]])),
        (np.array([[1.]]), np.array([[np.inf]])),
        (np.zeros((1, 2)), np.zeros((1, 2))),
    ],
)
def test_invalid_official_inputs_fail_closed(actual, forecast):
    with pytest.raises(ValueError):
        champion.score_official(actual, forecast)


def make_synthetic_phase2(tmp_path, monkeypatch):
    monkeypatch.setattr(champion, "EXPECTED_SERIES", 2)
    monkeypatch.setattr(champion, "EXPECTED_TOTAL_UNITS", 39.)
    monkeypatch.setattr(champion, "EXPECTED_CHAMPION_SCORE", 0.)
    ids = {
        "Client": [1, 2],
        "Warehouse": [3, 4],
        "Product": [5, 6],
    }
    h0 = pd.DataFrame({
        **ids, "2023-12-25": [7., 8.],
    })
    h1 = pd.DataFrame({
        **ids, "2024-01-01": [1., 1.],
    })
    dates = pd.date_range("2024-01-08", periods=13, freq="7D").strftime("%Y-%m-%d")
    predicted = np.stack([np.full(13, 2.), np.full(13, 1.)])
    target = pd.concat(
        [pd.DataFrame(ids), pd.DataFrame(predicted, columns=dates)], axis=1,
    )
    for name, frame in (("phase0", h0), ("phase1", h1), ("truth", target)):
        frame.to_csv(tmp_path / (name + ".csv"), index=False)
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    np.savez_compressed(
        frozen / "frozen-phase2.npz",
        historical_selected=predicted,
        arima30_client_amsi_last10=predicted,
    )
    digest = champion.sha256_file(frozen / "frozen-phase2.npz")
    manifest = {
        "history_only_selection": {"selected": champion.EXPECTED_SELECTED},
        "phase_outputs": {
            "2": {
                "series": 2,
                "dates": list(dates),
                "candidate_names": [
                    "historical_selected", "arima30_client_amsi_last10",
                ],
                "frozen_sha256": digest,
            }
        },
    }
    (frozen / "manifest.json").write_text(json.dumps(manifest))
    return frozen, tmp_path / "truth.csv", tmp_path / "phase0.csv", tmp_path / "phase1.csv", digest


def test_pinned_frozen_replay_succeeds_with_valid_aligned_synthetic_data(tmp_path, monkeypatch):
    frozen, truth, phase0, phase1, digest = make_synthetic_phase2(tmp_path, monkeypatch)
    result = champion.replay(
        frozen, truth, phase0, phase1, expected_frozen_sha256=digest,
    )
    assert result["status"] == "retrospective_reproduction_passed"
    assert result["score"]["vn1"] == 0.
    assert result["no_retraining"] is True
    assert result["is_blind_competition_submission"] is False


def test_modified_archive_is_detected_before_truth_is_read(tmp_path, monkeypatch):
    frozen, truth, phase0, phase1, digest = make_synthetic_phase2(tmp_path, monkeypatch)
    with (frozen / "frozen-phase2.npz").open("ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError, match="hash differs"):
        champion.replay(frozen, truth, phase0, phase1, expected_frozen_sha256=digest)


def test_reordered_targets_fail_closed(tmp_path, monkeypatch):
    frozen, truth, phase0, phase1, digest = make_synthetic_phase2(tmp_path, monkeypatch)
    frame = pd.read_csv(truth)
    frame.iloc[::-1].to_csv(truth, index=False)
    with pytest.raises(ValueError, match="identifiers or order"):
        champion.replay(frozen, truth, phase0, phase1, expected_frozen_sha256=digest)


def test_selected_alias_mismatch_rejected(tmp_path, monkeypatch):
    frozen, truth, phase0, phase1, _ = make_synthetic_phase2(tmp_path, monkeypatch)
    selected = np.zeros((2, 13))
    actual = np.stack([np.full(13, 2.), np.full(13, 1.)])
    np.savez_compressed(
        frozen / "frozen-phase2.npz",
        historical_selected=selected,
        arima30_client_amsi_last10=actual,
    )
    manifest_path = frozen / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    digest = champion.sha256_file(frozen / "frozen-phase2.npz")
    manifest["phase_outputs"]["2"]["frozen_sha256"] = digest
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="differs from named champion"):
        champion.replay(frozen, truth, phase0, phase1, expected_frozen_sha256=digest)


def test_wrong_total_or_future_date_rejected(tmp_path, monkeypatch):
    frozen, truth, phase0, phase1, digest = make_synthetic_phase2(tmp_path, monkeypatch)
    frame = pd.read_csv(truth)
    frame.iloc[0, 3] += 1
    frame.to_csv(truth, index=False)
    with pytest.raises(ValueError, match="checksum differs"):
        champion.replay(frozen, truth, phase0, phase1, expected_frozen_sha256=digest)
    frame.iloc[0, 3] -= 1
    frame = frame.rename(columns={"2024-01-08": "2024-01-09"})
    frame.to_csv(truth, index=False)
    with pytest.raises(ValueError, match="dates do not align"):
        champion.replay(frozen, truth, phase0, phase1, expected_frozen_sha256=digest)
