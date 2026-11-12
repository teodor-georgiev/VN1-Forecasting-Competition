"""Chronology, eligibility, and fallback tests for persistent champion bias."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import vn1_persistent_bias_specialist as v


def history(n: int = 40, weeks: int = 170) -> pd.DataFrame:
    ids = pd.DataFrame({
        "Client": np.arange(n) // 5,
        "Warehouse": np.arange(n) % 3,
        "Product": np.arange(n) + 100,
    })
    dates = pd.date_range("2020-07-06", periods=weeks, freq="7D").strftime("%Y-%m-%d")
    levels = np.ones((n, weeks), dtype=float)
    levels[0] = 40.0
    levels[1] = 20.0
    levels[2] = 10.0
    levels[-1] = 0.0
    return pd.concat([ids, pd.DataFrame(levels, columns=dates)], axis=1)


def records(n: int, origins=(105, 118), *, reverse_second: bool = True) -> list[dict]:
    out = []
    for index, origin in enumerate(origins):
        champion = np.full((n, 13), 10.0)
        actual = champion.copy()
        actual[0, :] += 5.0  # persistent underforecast
        actual[1, :] += 4.0 if index == 0 or not reverse_second else -4.0
        out.append({"origin": origin, "champion": champion, "truth": actual})
    return out


def test_stable_bias_is_positive_but_reversed_bias_is_not_trusted():
    h = history().iloc[:, :3 + 131]
    effects = v.estimate_effects(h, records(len(h)))
    assert effects["series"][0] > 0
    assert effects["series"][1] == 0
    assert effects["top_volume"][0]
    assert not effects["top_volume"][-1]
    assert effects["active"][-1] == 0


def test_candidate_forecasts_preserve_identity_mass_and_nonnegative_output():
    h = history().iloc[:, :3 + 131]
    champion = np.full((len(h), 13), 10.0)
    champion[-1] = 0.0
    choices, details = v.candidate_forecasts(h, champion, records(len(h)))
    assert tuple(choices) == v.POLICIES
    np.testing.assert_array_equal(choices["champion"], champion)
    assert details["oof_origins"] == [105, 118]
    for key, forecast in choices.items():
        assert np.isfinite(forecast).all() and (forecast >= 0).all(), key
        assert forecast.shape == champion.shape
        if key != "group_unscaled25":
            np.testing.assert_allclose(
                forecast.sum(axis=0), champion.sum(axis=0), atol=1e-8)
        np.testing.assert_array_equal(forecast[-1], champion[-1])
        np.testing.assert_array_equal(forecast[3], champion[3])  # inactive fallback
    assert not np.array_equal(choices["sku_stable35"], champion)


def test_records_must_be_completed_ordered_and_full_grid():
    h = history().iloc[:, :3 + 131]
    with pytest.raises(ValueError, match="future"):
        v.estimate_effects(h, records(len(h), origins=(118, 131)))
    with pytest.raises(ValueError, match="order"):
        v.estimate_effects(h, records(len(h), origins=(118, 105)))
    with pytest.raises(ValueError, match="shape"):
        v.estimate_effects(h, records(len(h) - 1))
    with pytest.raises(ValueError, match="nonnegative"):
        bad = records(len(h))
        bad[0]["truth"][0, 0] = -2
        v.estimate_effects(h, bad)


def test_mutations_after_origin_do_not_change_features_or_candidates():
    full = history()
    prior = full.iloc[:, :3 + 131].copy()
    changed = full.copy()
    changed.iloc[:, 3 + 131:] += 100000.0
    before = v.estimate_effects(prior, records(len(prior)))
    after = v.estimate_effects(changed.iloc[:, :3 + 131], records(len(prior)))
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])


def test_historical_gate_rejects_candidate_when_last_holdout_regresses(monkeypatch):
    h = history()
    origins_seen = []

    def fake_records(data, cutoff, cache, **kwargs):
        return records(len(data), origins=(105,))  # completed by first validation

    def fake_champion(past, **kwargs):
        origins_seen.append(past.shape[1] - 3)
        return np.ones((len(past), 13)), {}

    def fake_candidates(past, champion, completed):
        pred = {key: champion.copy() for key in v.POLICIES}
        # Earlier windows: high predictions generally reduce error (observed >= 1);
        # but holdout intentionally uses worse predictions.
        multiplier = 5.0 if past.shape[1] - 3 == 157 else 1.1
        for name in v.POLICIES[1:]:
            pred[name] = champion * multiplier
        return pred, {}

    monkeypatch.setattr(v.base, "_completed_records", fake_records)
    monkeypatch.setattr(v.base, "champion_forecast", fake_champion)
    monkeypatch.setattr(v, "candidate_forecasts", fake_candidates)
    receipt = v.historical_validation(h)
    assert origins_seen == [118, 131, 144, 157]
    assert receipt["holdout_origin"] == 157
    assert receipt["accepted"] is False
    assert receipt["selected"] == "champion"
    assert receipt["target_phase2_seen"] is False


def test_frozen_phase2_is_label_blind_aligned_and_integrity_checked(tmp_path, monkeypatch):
    import json

    phase0 = history()
    phase0.to_csv(tmp_path / "Phase 0 - Sales.csv", index=False)
    dates1 = pd.date_range(pd.Timestamp(phase0.columns[-1]) + pd.Timedelta(weeks=1),
                           periods=13, freq="7D").strftime("%Y-%m-%d")
    phase1 = pd.concat([phase0.iloc[:, :3],
                        pd.DataFrame(np.ones((len(phase0), 13)), columns=dates1)], axis=1)
    phase1.to_csv(tmp_path / "Phase 1 - Sales.csv", index=False)
    receipt = {
        "training_origins": list(v.base.TRAIN_EVAL_ORIGINS),
        "holdout_origin": v.base.HOLDOUT_ORIGIN,
        "candidate_policies": list(v.POLICIES),
        "training_scores": {"champion": .52, "group_unscaled25": .50},
        "holdout_scores": {"champion": .49, "group_unscaled25": .48},
        "train_fold_wins": {"group_unscaled25": 3},
        "winner_training_only": "group_unscaled25",
        "selected": "group_unscaled25", "accepted": True,
    }
    selected = tmp_path / "selection.json"
    selected.write_text(json.dumps(receipt))
    n = len(phase0)
    monkeypatch.setattr(v.base, "_completed_records", lambda *a, **k: records(n))
    monkeypatch.setattr(v.base, "champion_forecast", lambda *a, **k: (
        np.full((n, 13), 10.0), {}))
    frozen = tmp_path / "frozen"
    result = v.freeze(tmp_path, frozen, selected)
    assert result["no_target_labels_loaded_during_freeze"]
    assert result["selected"] == "group_unscaled25"
    assert (frozen / "forecast.npz").is_file()

    dates2 = pd.date_range(pd.Timestamp(dates1[-1]) + pd.Timedelta(weeks=1),
                           periods=13, freq="7D").strftime("%Y-%m-%d")
    truth = pd.concat([phase0.iloc[:, :3],
                       pd.DataFrame(np.full((n, 13), 10.0), columns=dates2)], axis=1)
    target_path = tmp_path / "Phase 2 - Sales.csv"
    with pytest.raises(FileNotFoundError):
        v.score(tmp_path, frozen, target_path)
    truth.to_csv(target_path, index=False)
    scores = v.score(tmp_path, frozen, target_path)
    assert scores["selection"] == "group_unscaled25"
    assert scores["series"] == n and scores["horizon"] == 13
    assert set(scores["scores"]) == {"champion", "historical_selected"}
    assert scores["scores"]["champion"] == 0.0

    with (frozen / "forecast.npz").open("ab") as fp:
        fp.write(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        v.score(tmp_path, frozen, target_path)
    with pytest.raises(FileExistsError):
        v.freeze(tmp_path, frozen, selected)


def test_pin_original_reference_during_freeze_without_phase2_truth(tmp_path, monkeypatch):
    import hashlib
    import json

    prior = history()
    prior.to_csv(tmp_path / "Phase 0 - Sales.csv", index=False)
    dates1 = pd.date_range(pd.Timestamp(prior.columns[-1]) + pd.Timedelta(weeks=1),
                           periods=13, freq="7D").strftime("%Y-%m-%d")
    following = pd.concat([
        prior.iloc[:, :3],
        pd.DataFrame(np.ones((len(prior), 13)), columns=dates1)], axis=1)
    following.to_csv(tmp_path / "Phase 1 - Sales.csv", index=False)
    dates2 = pd.date_range(pd.Timestamp(dates1[-1]) + pd.Timedelta(weeks=1),
                           periods=13, freq="7D").strftime("%Y-%m-%d")
    selection = {
        "training_origins": list(v.base.TRAIN_EVAL_ORIGINS),
        "holdout_origin": v.base.HOLDOUT_ORIGIN,
        "candidate_policies": list(v.POLICIES),
        "training_scores": {"champion": .52, "group_unscaled25": .50},
        "holdout_scores": {"champion": .49, "group_unscaled25": .48},
        "train_fold_wins": {"group_unscaled25": 3},
        "winner_training_only": "group_unscaled25",
        "selected": "group_unscaled25", "accepted": True,
    }
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection))
    n = len(prior)
    monkeypatch.setattr(v.base, "_completed_records", lambda *a, **k: records(n))
    monkeypatch.setattr(v.base, "champion_forecast", lambda *a, **k: (
        np.full((n, 13), 10.0), {}))

    reference = tmp_path / "reference"
    reference.mkdir()
    immutable = reference / "frozen-phase2.npz"
    exact = np.full((n, 13), 7.0)
    np.savez_compressed(immutable, historical_selected=exact)
    digest = hashlib.sha256(immutable.read_bytes()).hexdigest()
    monkeypatch.setattr(v, "PINNED_REFERENCE_SHA256", digest)
    (reference / "manifest.json").write_text(json.dumps({
        "history_only_selection": {"selected": v.base.CHAMPION_NAME},
        "phase_outputs": {"2": {
            "series": n, "dates": list(dates2), "frozen_sha256": digest
        }},
    }))
    frozen = tmp_path / "frozen"
    manifest = v.freeze(tmp_path, frozen, path, reference_dir=reference)
    assert manifest["original_pr17_reference_sha256"] == digest
    assert manifest["no_target_labels_loaded_during_freeze"]
    with np.load(frozen / "forecast.npz") as bundle:
        np.testing.assert_array_equal(bundle["champion"], exact)
    assert not (tmp_path / "Phase 2 - Sales.csv").exists()

    with immutable.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="reference SHA-256"):
        v.freeze(tmp_path, tmp_path / "another-freeze", path,
                 reference_dir=reference)
