"""Contracts for combining saved foundation forecasts with the best VN1 champion."""
import json
import numpy as np
import pandas as pd
import pytest

from vn1_foundation_champion_ensemble import (
    load_aligned, optimize, freeze, score_frozen, model_grid, metric,
)


def fixture(tmp_path, n=15):
    keys = pd.DataFrame({
        "Client": np.arange(n)//3, "Warehouse": np.arange(n)%2,
        "Product": 100+np.arange(n)})
    dates1 = pd.date_range("2023-01-02", periods=13, freq="7D").strftime("%Y-%m-%d").tolist()
    dates2 = pd.date_range("2023-04-03", periods=13, freq="7D").strftime("%Y-%m-%d").tolist()
    y1 = pd.concat([keys, pd.DataFrame(np.full((n,13),10.),columns=dates1)],axis=1)
    history = y1.copy()
    y2 = pd.concat([keys, pd.DataFrame(np.full((n,13),12.),columns=dates2)],axis=1)
    def write(name, dates, value):
        x = pd.concat([keys, pd.DataFrame(np.full((n,13),value),columns=dates)],axis=1)
        p = tmp_path / (name + ".csv")
        x.to_csv(p,index=False)
        return p
    target1 = write("phase1-truth",dates1,10)
    target2 = write("phase2-truth",dates2,12)
    h = tmp_path / "history.csv"
    history.to_csv(h,index=False)
    c1 = write("champion-phase1",dates1,12)
    c2 = write("champion-phase2",dates2,13)
    chronos1 = write("chronos-phase1",dates1,9)
    chronos2 = write("chronos-phase2",dates2,11)
    times1 = write("times-phase1",dates1,10)
    times2 = write("times-phase2",dates2,12)
    return {
        "target1":target1,"target2":target2,"history":h,
        "champion1":c1,"champion2":c2,
        "chronos1":chronos1,"chronos2":chronos2,
        "times1":times1,"times2":times2,
    }


def test_full_catalogue_grid_is_simplex_bounded():
    choices = list(model_grid(2, step=0.05, max_foundation=0.3))
    assert choices[0] == (1.,0.,0.)
    assert all(sum(w)==pytest.approx(1.) and w[0]>=0.7-1e-12
               and all(x>=0 for x in w) for w in choices)
    assert len(choices)==28
    assert len(set(choices))==len(choices)


def test_metric_is_absolute_plus_signed_bias():
    prediction=np.array([[12.]*13,[10.]*13])
    truth=np.array([[10.]*13,[10.]*13])
    result=metric(prediction,truth)
    assert result["score"]==pytest.approx(2/20+2/20)
    assert result["absolute"]==pytest.approx(2/20)
    assert result["signed"]==pytest.approx(2/20)


def test_alignment_rejects_permuted_keys_and_wrong_dates(tmp_path):
    files=fixture(tmp_path)
    reference=pd.read_csv(files["target1"])
    bad=pd.read_csv(files["chronos1"])
    bad.iloc[[0,1]]=bad.iloc[[1,0]].to_numpy()
    bad.to_csv(files["chronos1"],index=False)
    with pytest.raises(ValueError,match="key"):
        load_aligned(files["chronos1"],reference)
    with pytest.raises(ValueError,match="date"):
        load_aligned(files["champion2"],reference)


def test_optimize_uses_only_phase1_and_preserves_manifest(tmp_path):
    files=fixture(tmp_path)
    weights_path=tmp_path/"weights.json"
    receipt=optimize(files["target1"],files["champion1"],
        {"chronos2":files["chronos1"],"timesfm2.5":files["times1"]},
        weights_path,min_gain=0.001)
    assert receipt["selected_weights"]["champion"]>=0.7
    assert receipt["selected_weights"]["timesfm2.5"]>0
    assert receipt["phase2_labels_read"] is False
    assert weights_path.exists()
    assert receipt["selection_phase"]==1
    assert receipt["model_order"]==["chronos2","timesfm2.5"]
    assert receipt["selected_score"]<receipt["champion_score"]
    assert len(receipt["source_sha256"])==3


def test_freeze_and_score_require_preselected_hashes(tmp_path):
    files=fixture(tmp_path)
    weightfile=tmp_path/"weights.json"
    optimize(files["target1"],files["champion1"],
        {"chronos2":files["chronos1"],"timesfm2.5":files["times1"]},
        weightfile,min_gain=0.001)
    out=tmp_path/"frozen"
    frozen=freeze(files["history"],files["champion2"],
        {"chronos2":files["chronos2"],"timesfm2.5":files["times2"]},
        weightfile,out)
    assert frozen["phase2_labels_read"] is False
    assert (out/"ensemble-phase2.csv").exists()
    assert frozen["product_count"]==15
    result=score_frozen(out,files["target2"])
    assert result["rows"]==15
    assert result["horizon"]==13
    assert result["ensemble"]["score"]>=0
    # Tampering with any frozen prediction must fail the score gate.
    (out/"ensemble-phase2.csv").write_text("tampered",encoding="utf-8")
    with pytest.raises(ValueError,match="SHA"):
        score_frozen(out,files["target2"])


def test_missing_or_new_foundation_cannot_silently_enter_frozen_ensemble(tmp_path):
    files=fixture(tmp_path)
    weights=tmp_path/"weights.json"
    optimize(files["target1"],files["champion1"],
        {"chronos2":files["chronos1"],"timesfm2.5":files["times1"]},
        weights,min_gain=0.001)
    with pytest.raises(ValueError,match="models"):
        freeze(files["history"],files["champion2"],
               {"chronos2":files["chronos2"]},weights,tmp_path/"invalid")


def test_no_gain_keeps_champion(tmp_path):
    files=fixture(tmp_path)
    p=tmp_path/"weights.json"
    result=optimize(files["target1"],files["champion1"],
      {"chronos2":files["champion1"]},p,min_gain=0.001)
    assert result["selected_weights"]=={"champion":1.0,"chronos2":0.0}
    assert result["accepted"] is False
