"""Isolated VN1 sktime/Darts/MLForecast comparison, with no champion changes.

Use historical Phase-0 folds for selection, then optional retrospective Phase 2.
Scoring follows functions_for_participants.py (MAE + absolute aggregate bias).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["Client", "Warehouse", "Product"]
HORIZON = 13
LAGS = list(range(1, 14))


def read_sales(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    if list(data.columns[:3]) != KEYS or data.duplicated(KEYS).any():
        raise ValueError(f"Invalid VN1 identifiers: {path}")
    dates = pd.DatetimeIndex(pd.to_datetime(data.columns[3:], errors="raise"))
    if not dates.is_monotonic_increasing or not (dates.dayofweek == 0).all():
        raise ValueError(f"Expected chronological Monday dates: {path}")
    if len(dates) > 1 and not (dates[1:] - dates[:-1] == pd.Timedelta(days=7)).all():
        raise ValueError(f"Non-weekly dates: {path}")
    values = data.iloc[:, 3:].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"Invalid historical sales: {path}")
    return data


def load_origin(data_dir: Path, origin: int, phase2: bool):
    phase0 = read_sales(data_dir / "Phase 0 - Sales.csv")
    if phase2:
        phase1 = read_sales(data_dir / "Phase 1 - Sales.csv")
        if not phase0[KEYS].equals(phase1[KEYS]):
            raise ValueError("Phase 0/1 series alignment differs")
        # Phase 2 *truth is not opened* by any forecasting code.
        phase0_last = pd.Timestamp(phase0.columns[-1])
        phase1_first = pd.Timestamp(phase1.columns[3])
        if phase1_first != phase0_last + pd.Timedelta(weeks=1):
            raise ValueError("Phase 0 and Phase 1 dates are not consecutive")
        history = pd.concat([phase0, phase1.iloc[:, 3:]], axis=1)
        truth = None
    else:
        if origin < 52 or origin + HORIZON > len(phase0.columns) - 3:
            raise ValueError("Historical origin must leave 13 observed target weeks")
        history = pd.concat([phase0.iloc[:, :3], phase0.iloc[:, 3:3 + origin]], axis=1)
        truth = phase0.iloc[:, 3 + origin:3 + origin + HORIZON].to_numpy(dtype=np.float64)
    dates = pd.DatetimeIndex(pd.to_datetime(history.columns[3:]))
    future = pd.date_range(dates[-1] + pd.Timedelta(weeks=1), periods=HORIZON, freq="W-MON")
    return (history[KEYS].copy(), history.iloc[:, 3:].to_numpy(dtype=np.float64),
            truth, dates, future)


def choose_sample(n: int, sample_size: int, seed: int) -> np.ndarray:
    if sample_size < 0:
        raise ValueError("sample-size must be nonnegative")
    if sample_size == 0 or sample_size >= n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=sample_size, replace=False))


def vn1_score(prediction: np.ndarray, actual: np.ndarray) -> dict:
    pred, truth = np.asarray(prediction, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim != 2 or pred.shape[1] != HORIZON:
        raise ValueError("Expected matching (series, 13 weeks) matrices")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all() or truth.sum() <= 0:
        raise ValueError("Invalid scoring values")
    err, volume = pred - truth, truth.sum()
    return {"vn1": float((np.abs(err).sum() + abs(err.sum())) / volume),
            "relative_mae": float(np.abs(err).sum() / volume),
            "signed_relative_bias": float(err.sum() / volume)}


def sktime_forecast(history: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    # Intentionally a local statistical baseline, NOT a global LightGBM comparison.
    from sktime.forecasting.naive import NaiveForecaster
    pred = np.empty((len(history), HORIZON), dtype=np.float64)
    for i, values in enumerate(history):
        forecaster = NaiveForecaster(strategy="mean", window_length=13)
        forecaster.fit(pd.Series(values, index=pd.DatetimeIndex(dates, freq="W-MON")))
        pred[i] = np.asarray(forecaster.predict(fh=list(range(1, HORIZON + 1))), dtype=float)
    return pred


def mlforecast_forecast(history: np.ndarray, dates: pd.DatetimeIndex, future: pd.DatetimeIndex) -> np.ndarray:
    from lightgbm import LGBMRegressor
    from mlforecast import MLForecast
    n, t = history.shape
    long = pd.DataFrame({"unique_id": np.repeat(np.arange(n, dtype=np.int32), t),
                         "ds": np.tile(dates.to_numpy(), n),
                         "y": history.reshape(-1)})
    model = MLForecast(
        models={"LGBM": LGBMRegressor(n_estimators=160, learning_rate=0.05,
                                      num_leaves=31, random_state=42, n_jobs=4, verbosity=-1)},
        freq="W-MON", lags=LAGS)
    model.fit(long)
    table = model.predict(HORIZON).pivot(index="unique_id", columns="ds", values="LGBM")
    pred = table.reindex(index=np.arange(n), columns=future).to_numpy(dtype=np.float64)
    if not np.isfinite(pred).all():
        raise ValueError("MLForecast produced misaligned predictions")
    return pred


def darts_forecast(history: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    from darts import TimeSeries
    from darts.models import LightGBMModel
    series = [TimeSeries.from_times_and_values(dates, row.reshape(-1, 1), freq="W-MON")
              for row in history]
    model = LightGBMModel(lags=13, output_chunk_length=1, n_estimators=160,
                          learning_rate=0.05, num_leaves=31, random_state=42,
                          n_jobs=4, verbosity=-1)
    model.fit(series)
    predictions = model.predict(HORIZON, series=series)
    if len(predictions) != len(history):
        raise ValueError("Darts returned the wrong number of series")
    return np.vstack([np.asarray(ts.values(), dtype=np.float64).reshape(-1)
                      for ts in predictions])


def run(args: argparse.Namespace) -> list[dict]:
    ids, history, truth, dates, future = load_origin(args.data_dir, args.origin, args.phase2)
    selected = choose_sample(len(ids), args.sample_size, args.seed)
    ids, history, truth = ids.iloc[selected].reset_index(drop=True), history[selected], truth[selected]
    if len(dates) != history.shape[1]:
        raise ValueError("History and date index differ")
    adapters = {"sktime": lambda: sktime_forecast(history, dates),
                "darts": lambda: darts_forecast(history, dates),
                "mlforecast": lambda: mlforecast_forecast(history, dates, future)}
    choices = list(adapters) if args.model == "all" else [args.model]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise ValueError("Choose a new empty output directory to preserve frozen results")
    receipts = []
    hashes = {}
    for name in choices:
        start = time.perf_counter()
        forecast = np.asarray(adapters[name](), dtype=np.float64)
        if forecast.shape != (len(ids), HORIZON) or not np.isfinite(forecast).all():
            raise ValueError(f"{name} produced invalid predictions")
        forecast = np.maximum(forecast, 0.0)
        receipt = {"framework": name, "model": "mean13" if name == "sktime" else "LGBM160",
                   "period": "Phase2_retrospective" if args.phase2 else "Phase0_historical",
                   "origin_week": len(dates), "series": len(ids), "seed": args.seed,
                   "seconds": round(time.perf_counter() - start, 3)}
        if truth is not None:
            receipt.update(vn1_score(forecast, truth))
        else:
            receipt["status"] = "frozen_unscored"
        table = ids.copy()
        for j, date in enumerate(future):
            table[date.strftime("%Y-%m-%d")] = forecast[:, j]
        out = args.output_dir / f"{name}_forecast.csv"
        table.to_csv(out, index=False)
        hashes[name] = {"file": out.name, "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
        receipts.append(receipt)
        print(json.dumps(receipt, sort_keys=True), flush=True)
    (args.output_dir / "scores.json").write_text(json.dumps(receipts, indent=2) + "\n", encoding="utf-8")
    if args.phase2:
        manifest = {
            "phase": "Phase 2",
            "status": "frozen_unscored",
            "series_count": len(ids),
            "sample_indices": [int(i) for i in selected],
            "forecast_dates": list(future.strftime("%Y-%m-%d")),
            "source_hashes": {
                name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
                for name in ("Phase 0 - Sales.csv", "Phase 1 - Sales.csv")
            },
            "forecasts": hashes,
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return receipts


def score_phase2(data_dir: Path, output_dir: Path) -> list[dict]:
    """Independently score immutable Phase 2 forecast CSVs after forecasting."""
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("phase") != "Phase 2" or manifest.get("status") != "frozen_unscored":
        raise ValueError("Not a Phase 2 frozen prediction manifest")
    for name, expected in manifest["source_hashes"].items():
        file = data_dir / name
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source data changed after freezing: {name}")
    actual = read_sales(data_dir / "Phase 2 - Sales.csv")
    indices = np.asarray(manifest["sample_indices"], dtype=np.int64)
    if (len(indices) != manifest["series_count"] or len(indices) == 0
            or np.any(np.diff(indices) <= 0) or indices[0] < 0 or indices[-1] >= len(actual)):
        raise ValueError("Invalid sample indices in manifest")
    actual = actual.iloc[indices].reset_index(drop=True)
    expected_dates = manifest["forecast_dates"]
    if list(actual.columns[3:]) != expected_dates or len(expected_dates) != HORIZON:
        raise ValueError("Incorrect Phase 2 target dates")
    results = []
    for name, info in manifest["forecasts"].items():
        if name not in ("sktime", "darts", "mlforecast") or info["file"] != f"{name}_forecast.csv":
            raise ValueError("Unrecognized model or unsafe forecast path")
        file = output_dir / info["file"]
        if hashlib.sha256(file.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Frozen output hash mismatch for {name}")
        predicted = read_sales(file)
        if list(predicted.columns[3:]) != expected_dates or not predicted[KEYS].equals(actual[KEYS]):
            raise ValueError(f"Frozen output rows or dates differ for {name}")
        result = {"framework": name, "period": "Phase2_retrospective",
                  "series": len(actual), **vn1_score(predicted.iloc[:, 3:].to_numpy(),
                                                     actual.iloc[:, 3:].to_numpy())}
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    out = output_dir / "phase2_scores.json"
    if out.exists():
        raise FileExistsError("Phase 2 score receipt already exists; do not overwrite")
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("research/vn1_framework_results"))
    parser.add_argument("--model", choices=("all", "sktime", "darts", "mlforecast"), default="all")
    parser.add_argument("--sample-size", type=int, default=256, help="0 uses all 15,053 series")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--origin", type=int, default=157)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--phase2", action="store_true", help="Freeze Phase 2 predictions WITHOUT opening Phase 2 labels")
    mode.add_argument("--score-phase2", action="store_true", help="Score already-frozen Phase 2 predictions in a separate invocation")
    args = parser.parse_args()
    if args.score_phase2:
        score_phase2(args.data_dir, args.output_dir)
    else:
        run(args)


if __name__ == "__main__":
    main()
