"""CPU-only contracts for a real, no-key TimeCopilot historical pilot."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from vn1_timecopilot_local_pilot import (
    choose_series, load_training_only, to_agent_frame, verify_and_freeze_predictions
)


def fixture_csv(path: Path) -> pd.DataFrame:
    dates = pd.date_range("2020-07-06", periods=170, freq="W-MON")
    vals = np.stack([
        np.arange(170) * 0 + 1,
        np.arange(170) * 0 + 4,
        np.arange(170) * 0 + 8,
        np.arange(170) * 0 + 14,
    ], axis=0)
    keys = pd.DataFrame({"Client":[1,2,3,4], "Warehouse":[1,1,2,2], "Product":[5,6,7,8]})
    frame = pd.concat([keys, pd.DataFrame(vals, columns=dates.strftime("%Y-%m-%d"))], axis=1)
    frame.to_csv(path,index=False)
    return frame


def test_training_selects_no_future_target_columns(tmp_path, monkeypatch):
    source = tmp_path / "Phase 0 - Sales.csv"
    fixture_csv(source)
    original = pd.read_csv
    observed = []
    def recording_reader(path,*args,**kwargs):
        result = original(path,*args,**kwargs)
        if kwargs.get("nrows") != 0:
            observed.append(list(kwargs["usecols"]))
        return result
    monkeypatch.setattr(pd,"read_csv",recording_reader)
    training, cols = load_training_only(source,157)
    assert len(cols) == 173
    assert training.shape == (4,160)
    assert all(cols.index(name) < 160 for name in observed[0])
    assert choose_series(training,2) == [1,2]
    mapped = to_agent_frame(training,[1,2])
    assert len(mapped) == 314
    assert mapped["unique_id"].iloc[0] == "00000001"
    assert mapped["unique_id"].iloc[157] == "00000002"
    assert mapped["y"].iloc[157] == 8


def test_reject_more_than_4_series(tmp_path):
    source = tmp_path / "Phase 0 - Sales.csv"
    fixture_csv(source)
    training,_ = load_training_only(source,157)
    with pytest.raises(ValueError):
        choose_series(training, 5)


def test_agent_predictions_require_real_tools_and_correct_alignment(tmp_path):
    source = tmp_path / "Phase 0 - Sales.csv"
    frame = fixture_csv(source)
    cols = list(frame.columns)
    events = ["tsfeatures_tool","cross_validation_tool","forecast_tool","detect_anomalies_tool"]
    msgs = [SimpleNamespace(parts=[SimpleNamespace(tool_name=x) for x in events])]
    dates = pd.to_datetime(cols[160:173])
    fcst = pd.concat([
        pd.DataFrame({"unique_id":f"{row:08d}", "ds":dates, "HistoricAverage":float(row+1)})
        for row in (1,2)
    ],ignore_index=True)
    fake = SimpleNamespace(
        all_messages=lambda: msgs,
        output=SimpleNamespace(selected_model="HistoricAverage"),
        fcst_df=fcst,
        eval_df=pd.DataFrame({"HistoricAverage":[1.0]}),
        features_df=pd.DataFrame({"x":[1.0]}),
        anomalies_df=pd.DataFrame({"x":[1.0]}),
    )
    result = verify_and_freeze_predictions(fake,[1,2],cols,157)
    assert result.shape == (2,13)
    assert np.all(result[0] == 2)
    assert np.all(result[1] == 3)
    fake.fcst_df = fcst.iloc[:-1]
    with pytest.raises(ValueError):
        verify_and_freeze_predictions(fake,[1,2],cols,157)
