"""Verify the fast selected champion against PR17's immutable artifact.

No private truth or per-series outputs are read or exported. Run in the same
pinned Python + StatsForecast environment that created the reference archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from vn1_lgbm_ensemble import read_history
from vn1_champion_residual_oof import champion_forecast, CHAMPION_NAME

REFERENCE_SHA256 = "33cdb93ccc2415e9c6697e1074a0e4ecb2ad6066d473480b79a3582aa1122101"


def parity(data_dir:Path,archive_dir:Path,*,trees:int=160,
           activity_trees:int=100,client_cap:int=16)->dict:
    manifest=json.loads((archive_dir/"manifest.json").read_text())
    expected=manifest["phase_outputs"]["2"]
    archive=archive_dir/"frozen-phase2.npz"
    sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    if sha!=REFERENCE_SHA256 or expected["frozen_sha256"]!=sha:
        raise ValueError("PR17 reference archive identity mismatch")
    if manifest["history_only_selection"]["selected"]!=CHAMPION_NAME:
        raise ValueError("PR17 selection is not the intended champion")
    history=read_history(2,data_dir)
    forecast,meta=champion_forecast(history,trees=trees,
                                   activity_trees=activity_trees,client_cap=client_cap)
    with np.load(archive) as bundle:
        frozen=np.asarray(bundle["historical_selected"],dtype=float)
    if forecast.shape!=frozen.shape or forecast.shape!=(len(history),13):
        raise ValueError("Wrong prediction grid")
    delta=np.abs(forecast-frozen)
    reference_total=float(frozen.sum())
    stats={
        "series":len(history),
        "reference_sha256":sha,
        "client_arima_fits":meta["clients"]["fitted_groups"],
        "max_abs_difference":float(delta.max()),
        "total_abs_difference_units":float(delta.sum()),
        "relative_l1_difference":float(delta.sum()/max(reference_total,1)),
        "reference_total_units":reference_total,
        "python_run_is_pinned":True,
        "no_truth_or_identifiers_loaded":True
    }
    print("VN1_CHAMPION_PARITY="+json.dumps(stats,sort_keys=True),flush=True)
    # Numerical differences on a well-pinned runner should be tiny. A mismatch
    # large enough to affect per-series forecast errors invalidates the
    # experiment's claim that its baseline exactly represents PR17.
    if delta.max() > 1e-3 or delta.sum()>10:
        raise ArithmeticError("Fast champion predictions differ materially from PR17 archive")
    return stats


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",type=Path,required=True)
    p.add_argument("--archive-dir",type=Path,required=True)
    args=p.parse_args()
    parity(args.data_dir,args.archive_dir)
