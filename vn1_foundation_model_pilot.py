"""Isolated zero-shot Chronos-2 and TimesFM 2.5 VN1 pilots.

Both checkpoints are Apache-2.0 and used only in this standalone experiment.
TimesFM 3.0 weights have a non-commercial license and are intentionally
excluded. No training, no phase-label leakage, and no change to the default
forecast. Pilot selection is made using Phase-0 weeks 1-65 only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from vn1_from_scratch import KEYS, HORIZON, read_sales, vn1_score
from vn1_lgbm_ensemble import choose_weight, ensemble_forecast, read_history

MODEL_IDS = {
    "chronos2-small": "autogluon/chronos-2-small",
    "chronos2": "amazon/chronos-2",
    "timesfm2.5": "google/timesfm-2.5-200m-pytorch",
}


def select_series(history: pd.DataFrame, limit: int = 96, seed: int = 2026) -> np.ndarray:
    """Deterministically sample sparse/medium/dense series from first 65 weeks."""
    if history.shape[1] - len(KEYS) < 65:
        raise ValueError("Need at least 65 weeks of observed sales")
    n = len(history)
    if limit <= 0 or limit >= n:
        return np.arange(n)
    initial = history.iloc[:, len(KEYS):len(KEYS)+65].to_numpy(dtype=np.float32)
    if not np.isfinite(initial).all() or (initial < 0).any():
        raise ValueError("Invalid sales history")
    share = (initial == 0).mean(axis=1)
    strata = np.digitize(share, [0.25, 0.8])
    rng = np.random.default_rng(seed)
    pools = [rng.permutation(np.flatnonzero(strata == i)).tolist() for i in range(3)]
    selected = []
    while len(selected) < limit:
        any_left = False
        for pool in pools:
            if pool and len(selected) < limit:
                selected.append(pool.pop())
                any_left = True
        if not any_left:
            break
    return np.array(sorted(selected), dtype=int)


def as_numpy(result) -> np.ndarray:
    if hasattr(result, "detach"):
        return result.detach().cpu().numpy()
    return np.asarray(result)


def chronos2_median(outputs, expected_batch: int) -> np.ndarray:
    """Chronos-2 returns a list of (variate, quantile, horizon) tensors."""
    if isinstance(outputs, list):
        values = [as_numpy(x) for x in outputs]
        if len(values) != expected_batch:
            raise ValueError("Chronos-2 returned wrong number of series")
        if any(x.ndim != 3 or x.shape[0] != 1 for x in values):
            raise ValueError("Expected one variate per Chronos-2 series")
        array = np.concatenate(values, axis=0)
    else:
        array = as_numpy(outputs)
    if (array.ndim != 3 or array.shape[0] != expected_batch
            or array.shape[1] < 1 or array.shape[1] % 2 != 1
            or array.shape[2] != HORIZON):
        raise ValueError("Chronos-2 predictions must be [batch, odd quantiles, 13]")
    median = array[:, array.shape[1] // 2, :]
    if not np.isfinite(median).all():
        raise ValueError("Nonfinite Chronos-2 median")
    return np.maximum(0, median)


def timesfm_point(values, expected_batch: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (expected_batch, HORIZON) or not np.isfinite(arr).all():
        raise ValueError("TimesFM point forecasts must be [batch, 13] and finite")
    return np.maximum(0, arr)


def build_predictor(model: str, device: str = "cpu") -> Callable[[np.ndarray], np.ndarray]:
    """Download/load a reviewed pretrained checkpoint, with no remote code execution."""
    if model not in MODEL_IDS:
        raise ValueError("Only reviewed pretrained models are supported")
    if device not in ("cpu", "cuda", "mps"):
        raise ValueError("Unsupported device")
    if device == "cuda":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("CUDA requires the PyTorch GPU runtime") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no NVIDIA GPU is available")
    if model.startswith("chronos2"):
        try:
            import torch
            from chronos import BaseChronosPipeline
        except ImportError as exc:
            raise RuntimeError("Install requirements-vn1-foundation-chronos.txt") from exc
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        pipeline = BaseChronosPipeline.from_pretrained(
            MODEL_IDS[model], device_map=device, torch_dtype=dtype)
        def predict(batch: np.ndarray) -> np.ndarray:
            # Chronos-2 requires [n_series, n_variates, history_length], unlike Bolt.
            context = torch.from_numpy(batch.astype(np.float32))[:, None, :]
            with torch.no_grad():
                predictions = pipeline.predict(inputs=context, prediction_length=HORIZON,
                                               batch_size=len(batch))
            return chronos2_median(predictions, len(batch))
        return predict
    if device not in ("cpu", "cuda"):
        raise ValueError("TimesFM 2.5 supports CPU or CUDA")
    try:
        import timesfm
    except ImportError as exc:
        raise RuntimeError("Install requirements-vn1-foundation-timesfm.txt") from exc
    model_instance = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        MODEL_IDS[model], force_download=False)
    model_instance.compile(timesfm.ForecastConfig(
        max_context=512, max_horizon=HORIZON,
        normalize_inputs=True, per_core_batch_size=16,
        use_continuous_quantile_head=True, infer_is_positive=True,
    ))
    def predict(batch: np.ndarray) -> np.ndarray:
        point, _ = model_instance.forecast(
            horizon=HORIZON, inputs=[np.asarray(x, dtype=np.float32) for x in batch])
        return timesfm_point(point, len(batch))
    return predict


def predict_in_batches(history_values: np.ndarray,
                       infer: Callable[[np.ndarray], np.ndarray],
                       batch_size: int = 16) -> np.ndarray:
    sales = np.asarray(history_values, dtype=np.float32)
    if sales.ndim != 2 or len(sales) == 0 or sales.shape[1] < 65:
        raise ValueError("Expect a nonempty series-by-history matrix")
    if batch_size < 1 or not np.isfinite(sales).all() or (sales < 0).any():
        raise ValueError("Invalid historical sales or batch size")
    results = []
    for offset in range(0, len(sales), batch_size):
        block = infer(sales[offset:offset + batch_size])
        results.append(timesfm_point(block, min(batch_size, len(sales)-offset)))
    return np.vstack(results)


def pilot(history: pd.DataFrame, incumbent: pd.DataFrame,
          infer: Callable[[np.ndarray], np.ndarray], *,
          max_series: int = 96, blend: float = 0.5, batch_size: int = 16,
          seed: int = 2026) -> tuple[pd.DataFrame, np.ndarray]:
    if not history[KEYS].equals(incumbent[KEYS]):
        raise ValueError("History and incumbent series order differs")
    if not np.isfinite(blend) or not 0 <= blend <= 1:
        raise ValueError("Blend must be between zero and one")
    indices = select_series(history, max_series, seed)
    context = history.iloc[indices, len(KEYS):].to_numpy(dtype=np.float32)
    transformer = predict_in_batches(context, infer, batch_size=batch_size)
    out = incumbent.copy(deep=True)
    old = incumbent.iloc[indices, len(KEYS):].to_numpy(dtype=np.float64)
    out.iloc[indices, len(KEYS):] = np.maximum(0, (1-blend)*old + blend*transformer)
    return out, indices


def run(phase: int, model: str, data_dir: Path, output: Path, *,
        max_series: int = 96, blend: float = 0.5,
        batch_size: int = 16, device: str = "cpu") -> dict:
    if max_series == 0 and device != "cuda":
        raise ValueError("Full-catalogue runs require explicit CUDA; no CPU fallback")
    history = read_history(phase, data_dir)
    chosen_weight = choose_weight(history)
    incumbent = ensemble_forecast(history, weight_lgbm=chosen_weight["weight_lgbm"])
    infer = build_predictor(model, device=device)
    new, selection = pilot(history, incumbent, infer, max_series=max_series,
                           blend=blend, batch_size=batch_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(output, index=False)
    results = {
        "phase": phase, "model": model, "checkpoint": MODEL_IDS[model],
        "selected_series": len(selection), "total_series": len(history),
        "blend": blend, "device": device,
        "historical_selection": "all series" if max_series == 0 else "only first 65 observed weeks",
        "full_catalogue": bool(len(selection) == len(history)),
        "gpu_requested": bool(device == "cuda"),
        "incumbent_weight_cv": chosen_weight, "output": str(output),
        "warning": "Published Phase targets are retrospective evaluation only, not blind scores; pretrained data overlap cannot be ruled out",
    }
    target = data_dir / f"Phase {phase} - Sales.csv"
    if target.exists():
        actual = read_sales(target)
        results["incumbent_official"] = vn1_score(incumbent, actual)["vn1_score"]
        results["pilot_official"] = vn1_score(new, actual)["vn1_score"]
        results["incumbent_selected_only"] = vn1_score(incumbent.iloc[selection], actual.iloc[selection])["vn1_score"]
        results["pilot_selected_only"] = vn1_score(new.iloc[selection], actual.iloc[selection])["vn1_score"]
    return results


def main():
    parser = argparse.ArgumentParser(description="VN1 zero-shot foundation-model pilot")
    parser.add_argument("--model", choices=tuple(MODEL_IDS), required=True)
    parser.add_argument("--phase", choices=(1, 2), type=int, default=1)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("VN1_Foundation_Model.csv"))
    parser.add_argument("--max-series", type=int, default=96,
                        help="Number of historically chosen series, or 0 for all (requires --device cuda)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--blend", type=float, default=0.5)
    parser.add_argument("--device", choices=("cpu","cuda","mps"), default="cpu")
    args = parser.parse_args()
    print(json.dumps(run(args.phase, args.model, args.data_dir, args.output,
                         max_series=args.max_series, batch_size=args.batch_size,
                         blend=args.blend, device=args.device), indent=2))


if __name__ == "__main__":
    main()
