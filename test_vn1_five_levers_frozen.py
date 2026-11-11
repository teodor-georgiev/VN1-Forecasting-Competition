"""Freeze and score must be separate and must never expose private prices."""
import json

import numpy as np
import pandas as pd
import pytest

import vn1_five_levers_frozen as f
from vn1_five_levers import FAMILIES
from test_vn1_five_levers import history, prices_for


def fixture_inputs(tmp_path):
    old=history(weeks=170)
    old.to_csv(tmp_path/"Phase 0 - Sales.csv",index=False)
    dates=pd.date_range(pd.Timestamp(old.columns[-1])+pd.Timedelta(weeks=1),
                        periods=13,freq="7D").strftime("%Y-%m-%d")
    p1=pd.concat([old[f.KEYS],
                  pd.DataFrame(np.ones((len(old),13)),columns=dates)],axis=1)
    p1.to_csv(tmp_path/"Phase 1 - Sales.csv",index=False)
    nextdates=pd.date_range(pd.Timestamp(dates[-1])+pd.Timedelta(weeks=1),
                            periods=13,freq="7D").strftime("%Y-%m-%d")
    p2=pd.concat([old[f.KEYS],
                  pd.DataFrame(np.ones((len(old),13))*2,columns=nextdates)],axis=1)
    p2.to_csv(tmp_path/"Phase 2 - Sales.csv",index=False)
    prices_for(old).to_csv(tmp_path/"Phase 0 - Price.csv",index=False)
    prices_for(p1).to_csv(tmp_path/"Phase 1 - Price.csv",index=False)
    sel={
        "origins_train":[105,118,131,144],
        "holdout_origin":157,
        "minimum_absolute_gain":0.001,
        "families":FAMILIES,"prices_loaded":True,
        "training_scores":{"learned_activity":0.51,"target_season_norm":0.50},
        "holdout_scores":{"learned_activity":0.49,"target_season_norm":0.48},
        "winner_train_only":"target_season_norm",
        "selected":"target_season_norm","accepted":True,
    }
    receipt=tmp_path/"historical.json"
    receipt.write_text(json.dumps(sel))
    return receipt


def fake_forecast(monkeypatch):
    monkeypatch.setattr(f,"incumbent",lambda hist,trees:
                        np.full((len(hist),13),2.))
    monkeypatch.setattr(f,"candidates",lambda hist,control,prices,trees,activity_trees:(
        {"learned_activity":control.copy(),
         "target_season_norm":control*.5},
        {"policies":["learned_activity","target_season_norm"],
         "prices_available":prices is not None}))


def test_freeze_phase2_before_private_target_and_score_only_later(monkeypatch,tmp_path):
    receipt=fixture_inputs(tmp_path)
    fake_forecast(monkeypatch)
    phase2=tmp_path/"Phase 2 - Sales.csv"
    phase2.rename(tmp_path/"hidden-Phase2.csv")
    out=tmp_path/"frozen"
    manifest=f.freeze(tmp_path,out,receipt,phase=2,
                      price0=tmp_path/"Phase 0 - Price.csv",
                      price1=tmp_path/"Phase 1 - Price.csv")
    assert manifest["selection"]=="target_season_norm"
    assert manifest["accepted"]
    assert manifest["price_data_embedded_in_archive"] is False
    assert manifest["phase_target_labels_not_loaded_in_freeze"]
    assert manifest["phase1_sales_sha256"]
    assert manifest["price1_sha256"]
    assert not (out/"score.json").exists()
    with pytest.raises(FileNotFoundError):
        f.score(tmp_path,out,phase=2,truth=phase2)
    phase2.rename(phase2)
    scores=f.score(tmp_path,out,phase=2,truth=phase2)
    assert scores["phase0_selected"]=="target_season_norm"
    assert scores["scores"]["historical_selected"]==pytest.approx(0.5)
    assert scores["retrospective_not_blind"]
    with pytest.raises(FileExistsError):
        f.freeze(tmp_path,out,receipt,phase=2)


def test_misaligned_or_tampered_forecasts_rejected(monkeypatch,tmp_path):
    receipt=fixture_inputs(tmp_path)
    fake_forecast(monkeypatch)
    out=tmp_path/"phase1"
    f.freeze(tmp_path,out,receipt,phase=1,
             price0=tmp_path/"Phase 0 - Price.csv")
    with pytest.raises(ValueError):
        f.score(tmp_path,out,phase=2,truth=tmp_path/"Phase 2 - Sales.csv")
    bundle=out/"frozen.npz"
    with bundle.open("ab") as file:file.write(b"tampered")
    with pytest.raises(ValueError,match="modified"):
        f.score(tmp_path,out,phase=1,truth=tmp_path/"Phase 1 - Sales.csv")


def test_changing_phase0_acceptance_or_prices_flag_is_rejected(tmp_path):
    receipt=fixture_inputs(tmp_path)
    selected=json.loads(receipt.read_text())
    selected["accepted"]=False
    receipt.write_text(json.dumps(selected))
    with pytest.raises(ValueError,match="gate"):
        f._selected(receipt,prices=True)
    selected["accepted"]=True
    receipt.write_text(json.dumps(selected))
    with pytest.raises(ValueError,match="mismatched"):
        f._selected(receipt,prices=False)
