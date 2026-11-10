"""Zero-shot predictor adapter contract tests without remote model downloads."""
import contextlib
import sys
import types

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from vn1_from_scratch import KEYS, HORIZON, forecast as statistical_forecast
from vn1_foundation_model_pilot import (
    select_series, chronos2_median, timesfm_point, build_predictor,
    predict_in_batches, pilot,
)


def fixture(n=18, weeks=170):
    rng = np.random.default_rng(144)
    ids = pd.DataFrame({"Client": np.arange(n)//3,
                        "Warehouse": np.arange(n)%3,
                        "Product": np.arange(n)+100})
    vals = rng.poisson(1+np.arange(n)[:,None]%6, size=(n,weeks)).astype(float)
    vals[0] = 0
    cols = pd.date_range("2020-07-06", periods=weeks, freq="W-MON").strftime("%Y-%m-%d")
    return pd.concat([ids, pd.DataFrame(vals, columns=cols)], axis=1)


def test_phase2_data_changes_cannot_modify_historical_selection():
    history = fixture()
    selected = select_series(history, 9)
    modified = history.copy()
    modified.iloc[:, len(KEYS) + 65:] += 10_000
    np.testing.assert_array_equal(selected, select_series(modified, 9))
    assert len(selected) == 9


def test_chronos_list_quantiles_and_timesfm_point_shapes():
    predictions = [np.broadcast_to(np.arange(9)[None, :, None],
                                   (1, 9, HORIZON)).copy() for _ in range(3)]
    result = chronos2_median(predictions, 3)
    assert result.shape == (3, HORIZON)
    assert np.all(result == 4)
    with pytest.raises(ValueError):
        chronos2_median(predictions, 2)
    with pytest.raises(ValueError):
        timesfm_point(np.zeros((2, 12)), 2)
    with pytest.raises(ValueError):
        chronos2_median([np.full((1, 9, HORIZON), np.nan)], 1)
    np.testing.assert_array_equal(timesfm_point(np.full((2, HORIZON), -3), 2),
                                  np.zeros((2, HORIZON)))


def test_batched_prediction_and_incumbent_are_unchanged():
    history = fixture()
    incumbent = statistical_forecast(history)
    old = incumbent.copy(deep=True)
    def fake_infer(batch):
        return np.repeat(batch[:, -13:].mean(axis=1)[:, None], HORIZON, axis=1)
    out, selected = pilot(history, incumbent, fake_infer, max_series=8, batch_size=3)
    assert_frame_equal(incumbent, old)
    not_selected = np.setdiff1d(np.arange(len(history)), selected)
    np.testing.assert_allclose(out.iloc[not_selected, len(KEYS):],
                               old.iloc[not_selected, len(KEYS):])
    assert out.shape == old.shape
    assert np.isfinite(out.iloc[:, len(KEYS):].to_numpy()).all()
    with pytest.raises(ValueError):
        predict_in_batches(history.iloc[:, len(KEYS):].to_numpy(), fake_infer, 0)


def test_chronos2_loader_contract(monkeypatch):
    class FakeTensor:
        def __init__(self, value):
            self.value = np.asarray(value)
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.value
    class FakePipeline:
        @classmethod
        def from_pretrained(cls, *a, **kw): return cls()
        def predict(self, inputs, prediction_length, batch_size):
            assert prediction_length == HORIZON and batch_size == 2
            assert inputs.value.shape == (2, 1, 65)
            return [FakeTensor(np.ones((1, 9, HORIZON))) for _ in range(2)]
    torch = types.ModuleType("torch")
    torch.float32, torch.bfloat16 = "float32", "bfloat16"
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.from_numpy = lambda x: FakeTensor(x)
    torch.no_grad = contextlib.nullcontext
    chronos = types.ModuleType("chronos")
    chronos.BaseChronosPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "chronos", chronos)
    predicted = build_predictor("chronos2-small")(np.ones((2, 65), dtype=np.float32))
    assert predicted.shape == (2, HORIZON)



@pytest.mark.parametrize("bf16_supported,expected_dtype", [(False, "float32"), (True, "bfloat16")])
def test_chronos2_gpu_dtype_respects_older_cuda_devices(monkeypatch, bf16_supported, expected_dtype):
    class FakePipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            assert kwargs["device_map"] == "cuda"
            assert kwargs["torch_dtype"] == expected_dtype
            return cls()
    torch = types.ModuleType("torch")
    torch.float32, torch.bfloat16 = "float32", "bfloat16"
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        is_bf16_supported=lambda: bf16_supported,
    )
    chronos = types.ModuleType("chronos")
    chronos.BaseChronosPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "chronos", chronos)
    build_predictor("chronos2-small", device="cuda")


@pytest.mark.parametrize("device,batch_size,core_batch", [
    ("cpu", None, 16),
    ("cuda", None, 4),
    ("cuda", 16, 16),
])
def test_timesfm_25_contract(monkeypatch, device, batch_size, core_batch):
    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_id, force_download):
            assert model_id == "google/timesfm-2.5-200m-pytorch"
            assert force_download is False
            return cls()
        def compile(self, config):
            assert config["max_horizon"] == HORIZON
            assert config["per_core_batch_size"] == core_batch
        def forecast(self, horizon, inputs):
            assert horizon == HORIZON and len(inputs) == 2
            return np.ones((2, HORIZON)), np.zeros((2, HORIZON, 10))
    tfm = types.ModuleType("timesfm")
    tfm.TimesFM_2p5_200M_torch = FakeModel
    tfm.ForecastConfig = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "timesfm", tfm)
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    predict = build_predictor("timesfm2.5", device=device, batch_size=batch_size)
    pred = predict(np.ones((2, 65), dtype=np.float32))
    assert pred.shape == (2, HORIZON)


def test_cuda_is_required_for_full_catalogue(tmp_path):
    from vn1_foundation_model_pilot import run
    with pytest.raises(ValueError, match="Full-catalogue runs require explicit CUDA"):
        run(1, "chronos2-small", tmp_path, tmp_path / "out.csv", max_series=0)


def test_explicit_gpu_request_never_silently_falls_back(monkeypatch):
    import sys, types
    from vn1_foundation_model_pilot import build_predictor
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    with pytest.raises(RuntimeError, match="no NVIDIA GPU"):
        build_predictor("timesfm2.5", device="cuda")
