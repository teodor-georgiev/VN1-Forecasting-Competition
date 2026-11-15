"""Small *real* TimeCopilot agent pilot with a local, tool-calling Ollama model.

This is intentionally NOT a full-catalogue benchmark and is NOT a TimeRSI run.
All LLM inference stays at the local Ollama endpoint (no API tokens).
Only Phase-0 columns before the chosen historical origin are supplied to
TimeCopilot. Actual future targets are read after predictions are frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vn1_timersi_fev import HORIZON, KEYS, official_fold_components


def load_training_only(source: Path, origin: int) -> tuple[pd.DataFrame, list[str]]:
    columns = list(pd.read_csv(source, nrows=0).columns)
    if tuple(columns[:3]) != KEYS or origin < 104 or origin + HORIZON > len(columns) - 3:
        raise ValueError("Invalid VN1 dataset columns or historical origin")
    names = columns[: 3 + origin]
    frame = pd.read_csv(source, usecols=names)
    dates = pd.DatetimeIndex(pd.to_datetime(names[3:]))
    if not (dates.dayofweek == 0).all() or not (dates[1:] - dates[:-1] == pd.Timedelta(days=7)).all():
        raise ValueError("Expected consecutive Monday weeks")
    values = frame.iloc[:, 3:].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Training data must contain nonnegative finite sales")
    return frame, columns


def choose_series(train: pd.DataFrame, sample_size: int) -> list[int]:
    """Select different volume bands using *training* sales alone."""
    if sample_size < 1 or sample_size > 4:
        raise ValueError("The local agent pilot allows 1-4 series only")
    values = train.iloc[:, 3:].to_numpy(dtype=np.float64)
    volumes = values.sum(axis=1)
    eligible = np.flatnonzero((volumes > 0) & (values[:, -52:].sum(axis=1) > 0))
    if eligible.size < sample_size:
        raise ValueError("Not enough active historical series")
    ranked = eligible[np.argsort(volumes[eligible], kind="stable")]
    fractions = np.linspace(0.25, 0.75, sample_size)
    chosen = [int(ranked[round(float(p) * (len(ranked) - 1))]) for p in fractions]
    if len(set(chosen)) != sample_size:
        raise ValueError("Duplicate deterministic sample")
    return chosen


def to_agent_frame(train: pd.DataFrame, indices: list[int]) -> pd.DataFrame:
    dates = pd.to_datetime(train.columns[3:])
    values = train.iloc[indices, 3:].to_numpy(dtype=np.float64)
    return pd.DataFrame({
        "unique_id": np.repeat([f"{i:08d}" for i in indices], len(dates)),
        "ds": np.tile(dates.to_numpy(), len(indices)),
        "y": values.reshape(-1),
    })


def verify_and_freeze_predictions(result, indices: list[int], source_columns: list[str], origin: int) -> np.ndarray:
    required_tools = {"tsfeatures_tool", "cross_validation_tool", "forecast_tool", "detect_anomalies_tool"}
    actual_tools = {
        getattr(part, "tool_name", None)
        for msg in result.all_messages()
        for part in getattr(msg, "parts", [])
        if getattr(part, "tool_name", None)
    }
    if not required_tools.issubset(actual_tools):
        raise RuntimeError(f"Actual TimeCopilot tool calls incomplete: {sorted(actual_tools)}")
    if any(getattr(result, name, None) is None
           for name in ("features_df", "eval_df", "fcst_df", "anomalies_df")):
        raise RuntimeError("TimeCopilot did not produce all required agent artifacts")
    selected = result.output.selected_model
    if selected not in {"HistoricAverage", "SeasonalNaive", "ZeroModel"}:
        raise ValueError(f"Unexpected selected model: {selected}")
    fcst = result.fcst_df
    if selected not in fcst.columns:
        raise ValueError("Selected model's forecast column was not returned")
    wanted_dates = pd.DatetimeIndex(pd.to_datetime(source_columns[3 + origin:3 + origin + HORIZON]))
    preds = []
    for idx in indices:
        uid = f"{idx:08d}"
        subset = fcst.loc[fcst["unique_id"].astype(str) == uid].copy()
        subset["ds"] = pd.to_datetime(subset["ds"])
        subset = subset.set_index("ds").reindex(wanted_dates)
        vals = subset[selected].to_numpy(dtype=np.float64)
        if vals.shape != (HORIZON,) or not np.isfinite(vals).all():
            raise ValueError(f"Invalid or misaligned predictions for series row {idx}")
        preds.append(vals)
    return np.stack(preds)


def pilot(source: Path, destination: Path, origin: int, sample_size: int, llm: str) -> dict:
    if destination.exists():
        raise FileExistsError("Refusing to overwrite an existing research receipt")
    frame, columns = load_training_only(source, origin)
    indices = choose_series(frame, sample_size)
    agent_data = to_agent_frame(frame, indices)

    from timecopilot import TimeCopilot
    from timecopilot.models.stats import HistoricAverage, SeasonalNaive, ZeroModel

    tc = TimeCopilot(
        llm=llm,
        forecasters=[HistoricAverage(), SeasonalNaive(season_length=52), ZeroModel()],
        retries=2,
    )
    result = tc.forecast(agent_data, h=HORIZON, freq="W-MON", seasonality=52)
    frozen = verify_and_freeze_predictions(result, indices, columns, origin)
    # Separate, immutable-by-convention freeze BEFORE opening future labels.
    destination.parent.mkdir(parents=True, exist_ok=True)
    frozen_path = destination.with_suffix(".frozen.npz")
    if frozen_path.exists():
        raise FileExistsError("Forecast freeze artifact already exists")
    np.savez_compressed(frozen_path, predictions=frozen, series_rows=np.asarray(indices))
    frozen_hash = hashlib.sha256(frozen_path.read_bytes()).hexdigest()

    # Only now open the 13-week historical holdout from Phase 0 for scoring.
    truth_frame = pd.read_csv(source, usecols=columns[3 + origin:3 + origin + HORIZON])
    actual = truth_frame.iloc[indices].to_numpy(dtype=np.float64)
    score = official_fold_components(frozen, actual)
    receipt = {
        "experiment": "genuine TimeCopilot LLM agent via local Ollama",
        "provider": llm,
        "origin": origin,
        "series": len(indices),
        "horizon": HORIZON,
        "selected_model": result.output.selected_model,
        "forecast_sha256": frozen_hash,
        "phase2_labels_loaded": False,
        "phase0_holdout_read_after_freeze": True,
        "sample_only_not_full_catalogue": True,
        "comparable_to_full_catalogue_champion": False,
        "vn1": score["vn1"],
        "relative_mae": score["absolute_error"] / score["actual_units"],
        "signed_relative_bias": score["signed_error"] / score["actual_units"],
    }
    destination.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("VN1_TIMECOPILOT_LOCAL_AGENT_PILOT=PASS")
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source", type=Path, default=Path("Phase 0 - Sales.csv"))
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--origin", type=int, default=157)
    cli.add_argument("--sample-size", type=int, default=2)
    cli.add_argument("--llm", default="ollama:qwen2.5:3b")
    args = cli.parse_args()
    if not args.llm.startswith("ollama:"):
        cli.error("This no-key pilot permits only a local Ollama model")
    pilot(args.source, args.out, args.origin, args.sample_size, args.llm)
