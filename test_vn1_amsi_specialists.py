"""Synthetic shape, chronology, selection-gate and optional Nixtla integration tests."""
import numpy as np
import pandas as pd
import pytest

from vn1_amsi_specialists import (
    AMSI_CONFIGS, STATS_MODELS, amsi_forecast, candidate_forecasts,
    choose_challenger, historical_origins, make_stats_model,
    statsforecast_specialist,
)
from vn1_from_scratch import HORIZON, KEYS, forecast as statistical_forecast


def history(n=18, weeks=170, seed=2026):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-07-06", periods=weeks, freq="W-MON").strftime("%Y-%m-%d")
    ids = pd.DataFrame({
        "Client": np.arange(n) // 3,
        "Warehouse": np.arange(n) % 3,
        "Product": np.arange(n) + 100,
    })
    weekly = rng.poisson(np.arange(n)[:, None] % 7 + 7, size=(n, weeks)).astype(float)
    weekly[0] = 0
    weekly[1] = 0
    weekly[1, -30:] = np.array([0, 20, 0] * 10)
    return pd.concat([ids, pd.DataFrame(weekly, columns=dates)], axis=1)


def test_amsi_model_registry_shapes_and_all_zero():
    hist = history()
    output, details = candidate_forecasts(hist)
    assert set(output) == set(AMSI_CONFIGS)
    assert details["series"] == len(hist)
    for forecasts in output.values():
        assert forecasts.shape == (18, HORIZON)
        assert np.isfinite(forecasts).all() and (forecasts >= 0).all()
        assert (forecasts[0] == 0).all()


def test_annual_profile_uses_only_historical_matched_weeks():
    hist = history(n=6)
    first = amsi_forecast(hist, anchor="mean13", seasonality=1, group_shrink=0)
    hist2 = hist.copy()
    # The 52-week-matched future window, not the immediately previous 13 weeks.
    hist2.iloc[2, 3 + 170 - 52] += 10000
    changed = amsi_forecast(hist2, anchor="mean13", seasonality=1, group_shrink=0)
    assert changed[2, 0] > first[2, 0]
    assert changed[2, 1] == pytest.approx(first[2, 1])
    # All other series are unaffected by their neighbor when group_shrink=0.
    np.testing.assert_array_equal(changed[3:], first[3:])


def test_anchor_and_shrink_input_validation():
    hist = history()
    with pytest.raises(ValueError, match="anchor"):
        amsi_forecast(hist, anchor="invalid")
    with pytest.raises(ValueError, match="Seasonality"):
        amsi_forecast(hist, seasonality=1.01)
    with pytest.raises(ValueError, match="historical"):
        amsi_forecast(hist.iloc[:, :65])
    with pytest.raises(ValueError, match="Duplicated"):
        amsi_forecast(pd.concat([hist.iloc[[0]], hist.iloc[[0]]], ignore_index=True))


def test_current_forecasts_do_not_consult_unobserved_labels():
    hist = history()
    early = hist.iloc[:, :3 + 144].copy()
    original = amsi_forecast(early)
    mutated = hist.copy()
    mutated.iloc[:, 3 + 144:] += 100000
    np.testing.assert_array_equal(
        original, amsi_forecast(mutated.iloc[:, :3 + 144]))


def test_statsforecast_registry_has_no_arbitrary_model_execution():
    assert "autoarima52" in STATS_MODELS
    assert "seasonalnaive52" in STATS_MODELS
    with pytest.raises(ValueError, match="Unsupported"):
        make_stats_model("__import__('os').system('echo unsafe')")


def test_bounded_statistical_fit_and_preserved_fallback(monkeypatch):
    hist = history(n=12)
    fallback = np.full((12, HORIZON), 3.0)
    class Stub:
        def forecast(self, *, y, h):
            return {"mean": np.full(h, y[-1] + 2)}
    monkeypatch.setattr("vn1_amsi_specialists.make_stats_model", lambda name: Stub())
    result, mask = statsforecast_specialist(hist, "autoarima52", cap=2, fallback=fallback)
    assert mask.sum() == 2
    assert (result[~mask] == 3.0).all()
    assert np.isfinite(result).all()
    with pytest.raises(ValueError, match="cap"):
        statsforecast_specialist(hist, "autoarima52", cap=513, fallback=fallback)


def test_failing_statistical_fit_falls_back_without_corrupting_others(monkeypatch):
    hist = history(n=12)
    fallback = np.ones((12, HORIZON))
    class Broken:
        def forecast(self, *, y, h):
            raise ValueError("unfittable series")
    monkeypatch.setattr("vn1_amsi_specialists.make_stats_model", lambda name: Broken())
    pred, mask = statsforecast_specialist(hist, "autoarima52", cap=4, fallback=fallback)
    assert mask.sum() == 0
    np.testing.assert_array_equal(pred, fallback)


def test_nixtla_seasonal_naive_one_series():
    pytest.importorskip("statsforecast")
    hist = history(n=9)
    output, mask = statsforecast_specialist(
        hist, "seasonalnaive52", cap=1,
        fallback=np.ones((9, HORIZON)))
    assert mask.sum() == 1
    assert np.isfinite(output).all()


def test_historical_origins_end_before_every_validation_target():
    hist = history()
    origins = historical_origins(hist)
    assert origins == [105, 118, 131, 144, 157]
    assert all(x + HORIZON <= 170 for x in origins)
    with pytest.raises(ValueError):
        historical_origins(history(weeks=130))


def test_champion_fallback_and_historical_split():
    hist = history(n=12, weeks=144)
    def champion(earlier):
        return statistical_forecast(earlier).iloc[:, 3:].to_numpy()
    selected = choose_challenger(hist, champion_fn=champion, min_gain=1)
    assert selected["training_origins"] == [105, 118]
    assert selected["holdout_origin"] == 131
    assert selected["selected_model"] == "champion"
    assert selected["selected_weight"] == 0
    assert selected["accepted"] is False


def test_experiment_can_add_multiple_statistics_without_modifying_champion(monkeypatch):
    hist = history(n=12, weeks=144)
    class Stub:
        def forecast(self, *, y, h):
            return {"mean": np.full(h, y[-13:].mean())}
    monkeypatch.setattr("vn1_amsi_specialists.make_stats_model", lambda name: Stub())
    selected = choose_challenger(
        hist, stats_models=("autoarima52", "dynamictheta"),
        stats_cap=2, min_gain=1,
        champion_fn=lambda earlier: statistical_forecast(earlier).iloc[:, 3:].to_numpy(),
    )
    assert selected["selected_model"] == "champion"
    assert set(selected["stats_models"]) == {"autoarima52", "dynamictheta"}
