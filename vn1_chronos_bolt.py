"""Optional zero-shot Chronos-Bolt experiment for VN1 13-week forecasts.

The model is downloaded only when this standalone experiment is executed.
The incumbent LightGBM/statistical blend remains the default for all other
VN1 workflows. Models are never selected using scored Phase 1/2 targets.

Pretrained weights may have pretraining overlap with historical public data:
these retrospective scores are not a blind competition submission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from vn1_from_scratch import HORIZON, KEYS, read_sales, vn1_score
from vn1_lgbm_ensemble import choose_weight, ensemble_forecast, read_history

DEFAULT_MODEL = "amazon/chronos-bolt-tiny"


def stratified_indices(history: pd.DataFrame, max_series: int,
                       seed: int = 2026) -> np.ndarray:
    """Preselect test series using historical sparsity only, without outcomes.

    Approximate equal sampling of dense, intermittent, and mostly-zero
    demand avoids testing only the easiest high-volume products. Every
    selection is deterministic for a fixed historical input and seed.
    A nonpositive max_series means all rows.
    """
    n = len(history)
    if max_series <= 0 or max_series >= n:
        return np.arange(n, dtype=int)
    y = history.iloc[:, len(KEYS):].to_numpy(dtype=np.float32)
    if y.ndim != 2 or y.shape[1] < 13 or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Invalid historical sales")
    zeros = (y[:, -52:] == 0).mean(axis=1)
    strata = np.digitize(zeros, bins=[0.25, 0.8])
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(np.flatnonzero(strata == k)) for k in range(3)]
    chosen = []
    # Round-robin strata sampling, filling from available groups when sparse.
    while len(chosen) < max_series:
        any_remaining = False
        for group_index, g in enumerate(groups):
            if len(chosen) >= max_series:
                break
            if len(g):
                chosen.append(int(g[0]))
                groups[group_index] = g[1:]
                any_remaining = True
        if not any_remaining:
            break
    return np.array(sorted(chosen), dtype=int)


def median_quantile(prediction: np.ndarray, *, n_series: int) -> np.ndarray:
    """Extract the centre quantile from Chronos-Bolt [B, Q, H] forecasts."""
    array = np.asarray(prediction, dtype=np.float64)
    if array.ndim != 3 or array.shape[0] != n_series or array.shape[2] != HORIZON:
        raise ValueError("Chronos forecast must be [batch, quantiles, 13]")
    if array.shape[1] % 2 != 1 or array.shape[1] < 1:
        raise ValueError("Expected odd quantile count containing median")
    median = array[:, array.shape[1] // 2, :]
    if not np.isfinite(median).all():
        raise ValueError("Chronos produced nonfinite forecasts")
    return np.maximum(0.0, median)


def predict_batched(history_values: np.ndarray,
                    infer_quantiles: Callable[[np.ndarray], np.ndarray],
                    batch_size: int = 64) -> np.ndarray:
    """Framework-independent adapter; testable without downloading a model."""
    y = np.asarray(history_values, dtype=np.float32)
    if y.ndim != 2 or y.shape[0] == 0 or y.shape[1] < 13:
        raise ValueError("Expect nonempty series-by-history matrix")
    if batch_size <= 0 or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Invalid batch size or sales")
    preds = []
    for start in range(0, len(y), batch_size):
        batch = y[start:start + batch_size]
        preds.append(median_quantile(infer_quantiles(batch), n_series=len(batch)))
    return np.vstack(preds)


def load_chronos_predictor(model_id: str = DEFAULT_MODEL,
                           device: str = "cpu") -> Callable[[np.ndarray], np.ndarray]:
    """Load pretrained weights explicitly; no remote code trust flag or training."""
    if model_id not in ("amazon/chronos-bolt-tiny",
                        "amazon/chronos-bolt-mini",
                        "amazon/chronos-bolt-small"):
        raise ValueError("Only reviewed Chronos-Bolt model IDs are allowed")
    if device not in ("cpu", "cuda", "mps"):
        raise ValueError("Unsupported device")
    try:
        import torch
        from chronos import BaseChronosPipeline
    except ImportError as exc:
        raise RuntimeError("Install torch and chronos-forecasting for real inference") from exc
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    pipeline = BaseChronosPipeline.from_pretrained(
        model_id, device_map=device, torch_dtype=dtype)

    def infer(batch: np.ndarray) -> np.ndarray:
        context = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            result = pipeline.predict(context=context, prediction_length=HORIZON)
        return result.detach().cpu().numpy()
    return infer


def experiment(history: pd.DataFrame, incumbent: pd.DataFrame,
               infer_quantiles: Callable[[np.ndarray], np.ndarray], *,
               max_series: int = 384, blend: float = 0.5,
               batch_size: int = 64, seed: int = 2026
               ) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """Change only preselected series, preserving the champion everywhere else."""
    if not 0 <= blend <= 1 or not np.isfinite(blend):
        raise ValueError("blend must be between 0 and 1")
    if not history[KEYS].equals(incumbent[KEYS]):
        raise ValueError("Forecast and history IDs/order must match")
    chosen = stratified_indices(history, max_series, seed=seed)
    obs = history.iloc[chosen, len(KEYS):].to_numpy(dtype=np.float32)
    transformer = predict_batched(obs, infer_quantiles, batch_size=batch_size)
    out = incumbent.copy(deep=True)
    before = incumbent.iloc[chosen, len(KEYS):].to_numpy(dtype=float)
    out.iloc[chosen, len(KEYS):] = (1 - blend) * before + blend * transformer
    return out, {
        "tested_series": int(len(chosen)),
        "available_series": int(len(history)),
        "historical_selection": "stratified by zero share in last 52 observed weeks",
        "fixed_blend_weight": blend,
        "note": "Unselected series use unchanged incumbent forecasts",
    }, chosen


def run(phase: int, data_dir: Path, output: Path, *,
        model_id: str = DEFAULT_MODEL, max_series: int = 384,
        blend: float = 0.5, batch_size: int = 64, device: str = "cpu") -> dict:
    history = read_history(phase, data_dir)
    selected = choose_weight(history)
    incumbent = ensemble_forecast(history, weight_lgbm=selected["weight_lgbm"])
    infer = load_chronos_predictor(model_id, device=device)
    submission, info, idx = experiment(history, incumbent, infer,
                                       max_series=max_series, blend=blend,
                                       batch_size=batch_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    result = {
        "phase": phase, "model": model_id, "device": device,
        "output": str(output), "incumbent_weight_from_historical_cv": selected,
        **info,
        "limitations": "Retrospective experiment; pretrained model data overlap cannot be ruled out",
    }
    actual_path = data_dir / f"Phase {phase} - Sales.csv"
    if actual_path.exists():
        actual = read_sales(actual_path)
        result["incumbent_official_score"] = vn1_score(incumbent, actual)["vn1_score"]
        result["experiment_official_score"] = vn1_score(submission, actual)["vn1_score"]
        result["incumbent_selected_series_score"] = vn1_score(incumbent.iloc[idx], actual.iloc[idx])["vn1_score"]
        result["experiment_selected_series_score"] = vn1_score(submission.iloc[idx], actual.iloc[idx])["vn1_score"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="VN1 zero-shot Chronos-Bolt ablation")
    parser.add_argument("--phase", type=int, choices=(1, 2), default=1)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("VN1_Chronos_Bolt.csv"))
    parser.add_argument("--model-id", choices=(
        "amazon/chronos-bolt-tiny", "amazon/chronos-bolt-mini",
        "amazon/chronos-bolt-small"), default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--max-series", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--blend", type=float, default=0.5)
    args = parser.parse_args()
    print(json.dumps(run(args.phase, args.data_dir, args.output,
                         model_id=args.model_id, device=args.device,
                         max_series=args.max_series, batch_size=args.batch_size,
                         blend=args.blend), indent=2))


if __name__ == "__main__":
    main()
