"""VN1 learned demand-activity probabilities and volume-neutral forecast allocation.

Research only: candidates were developed after inspecting public Phase 2 results.
Phase 0 rolling-origin selection and its final historical holdout cannot turn
retrospective Phase 1/2 scores into independent blind generalization evidence.

Two commands:
  freeze: fit only on fully observed training origins, select only using Phase 0,
          save all candidates and their immutable SHA256 before scoring.
  score:  check frozen artifact hash and series/date alignment, THEN read targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from vn1_from_scratch import HORIZON, KEYS, read_sales
from vn1_lgbm_ensemble import read_history
from vn1_amsi_zero_rule_ablation import incumbent, _metric
from vn1_soft_dormancy import candidate_forecasts as prior_candidates

# All fold targets are inside Phase 0. Last fold (157 -> 170) is held out.
ORIGINS = (105, 118, 131, 144, 157)
TRAIN_START = 65
TRAIN_STRIDE = 13
MIN_GAIN = 0.001
CANDIDATE_NAMES = (
    "incumbent",
    "prior_tsb10_quiet26_weekly",
    "activity_p25",
    "activity_p50",
    "activity_p100",
    "activity_p15_hard",
    "activity_p25_hard",
    "tsb10_activity_p50",
    "tsb10_activity_p15_hard",
)
PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "n_estimators": 100,
    "min_child_samples": 80,
    "reg_lambda": 4.0,
    "colsample_bytree": 0.9,
    "max_bin": 127,
    "random_state": 2026,
    "n_jobs": 2,
    "verbosity": -1,
}


def history_values(history: pd.DataFrame) -> np.ndarray:
    if list(history.columns[:3]) != list(KEYS) or history.duplicated(KEYS).any():
        raise ValueError("Invalid or duplicated series keys")
    y = history.iloc[:, 3:].to_numpy(dtype=np.float64)
    if y.ndim != 2 or y.shape[1] < TRAIN_START or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Invalid origin sales")
    return y


def activity_features(history: pd.DataFrame) -> pd.DataFrame:
    """Only the history before an origin; never read the target horizon.

    Features include per-series recency, rolling activity/level, same-week
    prior year and *historical* leave-one-out client, client×warehouse and
    product-group activity. The forecast calendar is known in advance.
    """
    y = history_values(history)
    n, n_weeks = y.shape
    out: dict[str, np.ndarray] = {}
    for window in (4, 13, 26, 52):
        recent = y[:, -window:]
        out[f"mean{window}"] = recent.mean(axis=1)
        out[f"active{window}"] = (recent > 0).mean(axis=1)
        out[f"sum{window}"] = recent.sum(axis=1)
    for lag in (1, 2, 4, 13, 26, 52):
        out[f"lag{lag}"] = y[:, -lag]
    recent52 = y[:, -52:]
    observed = recent52 > 0
    last_seen = np.where(observed, np.arange(52)[None, :], -1).max(axis=1)
    out["weeks_since_event"] = 51 - last_seen
    out["average_nonzero_size"] = (
        recent52.sum(axis=1) / np.maximum(1, observed.sum(axis=1))
    )
    out["prev13_mean"] = y[:, -26:-13].mean(axis=1)
    out["annual_matched_sum"] = y[:, -52:-39].sum(axis=1)
    out["annual_matched_nonzero"] = (y[:, -52:-39] > 0).sum(axis=1)
    out["annual_matched_mean"] = y[:, -52:-39].mean(axis=1)
    out["all_history_nonzero_fraction"] = (y > 0).mean(axis=1)
    out["all_history_mean"] = y.mean(axis=1)
    out["trend13_vs_prev13"] = (
        (y[:, -13:].sum(axis=1) + 1.0) /
        (y[:, -26:-13].sum(axis=1) + 1.0)
    )

    ids = history[KEYS]
    groupings = {
        "client": ids["Client"],
        "client_wh": pd.MultiIndex.from_frame(ids[["Client","Warehouse"]]),
        "product": ids["Product"],
    }
    # Leave-one-out group activity protects an isolated zero-selling SKU from
    # simply inheriting its own rolling level as a "group signal".
    a = (y[:, -13:] > 0).sum(axis=1).astype(float)
    vol = y[:, -13:].sum(axis=1)
    for prefix, group in groupings.items():
        codes, _ = pd.factorize(group, sort=True)
        size = np.bincount(codes)
        group_active = np.bincount(codes, weights=a)
        group_units = np.bincount(codes, weights=vol)
        out[f"{prefix}_peer_active"] = np.divide(
            group_active[codes] - a, 13 * np.maximum(size[codes] - 1, 1)
        )
        out[f"{prefix}_peer_units"] = np.maximum(
            0.0, (group_units[codes] - vol) / np.maximum(size[codes] - 1, 1)
        )
        out[f"{prefix}_peer_count"] = np.maximum(size[codes] - 1, 0).astype(float)

    last_date = pd.Timestamp(history.columns[-1])
    future = last_date + pd.Timedelta(weeks=HORIZON)
    out["origin_month"] = np.full(n, last_date.month, dtype=float)
    out["forecast_end_month"] = np.full(n, future.month, dtype=float)
    out["origin_year_index"] = np.full(n, last_date.year - 2020, dtype=float)
    result = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k,v in out.items()})
    if result.shape[0] != n or not np.isfinite(result.to_numpy()).all():
        raise ValueError("Invalid learned-feature matrix")
    return result


def training_origins(observed_weeks: int) -> list[int]:
    return [
        origin for origin in range(TRAIN_START, observed_weeks - HORIZON + 1, TRAIN_STRIDE)
        if origin + HORIZON <= observed_weeks
    ]


def train_activity(history: pd.DataFrame, *, estimators: int = 100):
    """Train event probabilities using no future labels beyond the origin."""
    y = history_values(history)
    origins = training_origins(y.shape[1])
    if not origins:
        raise ValueError("Not enough completed training origins")
    features = []
    label = []
    for origin in origins:
        past = history.iloc[:, :len(KEYS) + origin]
        features.append(activity_features(past))
        label.append((y[:, origin:origin+HORIZON].sum(axis=1) > 0).astype(np.int8))
    X = pd.concat(features, ignore_index=True)
    targets = np.concatenate(label)
    if X.shape[0] != len(targets):
        raise AssertionError("Feature and target rows do not align")
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Nonfinite feature")
    class_share = float(targets.mean())
    if not 0 < class_share < 1:
        return None, origins, class_share
    params = dict(PARAMS, n_estimators=estimators)
    classifier = lgb.LGBMClassifier(**params)
    classifier.fit(X, targets)
    return classifier, origins, class_share


def predict_activity(history: pd.DataFrame, *, estimators: int = 100) -> tuple[np.ndarray, dict]:
    model, origins, prior = train_activity(history, estimators=estimators)
    p = (np.full(len(history),prior) if model is None else
         model.predict_proba(activity_features(history))[:, 1])
    p = np.clip(np.asarray(p, dtype=np.float64), 0.001, 0.999)
    if p.shape != (len(history),) or not np.isfinite(p).all():
        raise ValueError("Invalid learned activity probabilities")
    return p, {
        "train_origins": origins,
        "train_event_rate": prior,
        "p05": float(np.quantile(p,0.05)),
        "p50": float(np.quantile(p,0.50)),
        "p95": float(np.quantile(p,0.95)),
        "low15": int((p < 0.15).sum()),
        "low25": int((p < 0.25).sum()),
    }


def allocate_weekly(base: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Scale series, restore the exact forecast total in EACH future week."""
    b = np.asarray(base,dtype=np.float64)
    w = np.asarray(weight,dtype=np.float64)
    if b.ndim != 2 or b.shape[1] != HORIZON or w.shape != (len(b),):
        raise ValueError("Invalid allocation shape")
    if not np.isfinite(b).all() or not np.isfinite(w).all() or (b < 0).any() or (w < 0).any():
        raise ValueError("Invalid forecast or weight")
    scaled = b * w[:, None]
    numerator = b.sum(axis=0)
    denominator = scaled.sum(axis=0)
    if np.any((numerator > 0) & (denominator <= 0)):
        return b.copy()
    scale = np.divide(numerator,denominator,
                      out=np.ones_like(numerator), where=denominator>0)
    out = scaled * scale[None,:]
    if not np.allclose(out.sum(axis=0),numerator,atol=1e-5,rtol=1e-10):
        raise ArithmeticError("Weekly forecast mass is not conserved")
    return out


def learned_candidates(history: pd.DataFrame, base: np.ndarray,
                       *, estimators: int = 100) -> tuple[dict[str,np.ndarray],dict]:
    probabilities, details = predict_activity(history,estimators=estimators)
    baseline, _ = prior_candidates(history,base)
    tsb_base = baseline["tsb_10_volume_preserving"]
    fixed = baseline["tsb10_plus_quiet26_weekly"]
    candidates = {
        "incumbent": np.asarray(base,dtype=np.float64).copy(),
        "prior_tsb10_quiet26_weekly": fixed,
        "activity_p25": allocate_weekly(base,probabilities ** 0.25),
        "activity_p50": allocate_weekly(base,probabilities ** 0.50),
        "activity_p100": allocate_weekly(base,probabilities),
        "activity_p15_hard": allocate_weekly(base,(probabilities >= 0.15).astype(float)),
        "activity_p25_hard": allocate_weekly(base,(probabilities >= 0.25).astype(float)),
        "tsb10_activity_p50": allocate_weekly(tsb_base,probabilities ** 0.50),
        "tsb10_activity_p15_hard": allocate_weekly(
            tsb_base,(probabilities >= 0.15).astype(float)),
    }
    if tuple(candidates) != CANDIDATE_NAMES:
        raise AssertionError("Candidate registry mismatch")
    for name, pred in candidates.items():
        if pred.shape != base.shape or not np.isfinite(pred).all() or (pred < 0).any():
            raise ValueError(f"Invalid candidate {name}")
    return candidates, details


def select_on_phase0(phase0: pd.DataFrame, trees: int, activity_trees: int) -> dict:
    y=history_values(phase0)
    if y.shape[1] < ORIGINS[-1] + HORIZON:
        raise ValueError("Incomplete held-out Phase 0 origin")
    loss={name:{"abs":0.0,"signed":0.0} for name in CANDIDATE_NAMES}
    holdout={}
    fold_details={}
    denominator=0.0
    for origin in ORIGINS:
        past=phase0.iloc[:, :3+origin].copy()
        base=incumbent(past,trees=trees)
        predictions, details=learned_candidates(past,base,estimators=activity_trees)
        target=y[:,origin:origin+HORIZON]
        scores={name:_metric(pred,target) for name,pred in predictions.items()}
        fold_details[str(origin)]={"scores":scores,"activity":details}
        if origin!=ORIGINS[-1]:
            denominator += float(target.sum())
            for name,pred in predictions.items():
                e=pred-target
                loss[name]["abs"] += float(np.abs(e).sum())
                loss[name]["signed"] += float(e.sum())
        else:
            holdout=scores
        print(f"VN1_ACTIVITY_FOLD={origin}",flush=True)
    train_scores={name:(v["abs"]+abs(v["signed"]))/denominator for name,v in loss.items()}
    winner=min(CANDIDATE_NAMES,key=lambda name:(train_scores[name],CANDIDATE_NAMES.index(name)))
    accepted=(winner!="incumbent" and
              train_scores[winner] <= train_scores["incumbent"]-MIN_GAIN and
              holdout[winner] <= holdout["incumbent"]-MIN_GAIN)
    return {
        "origins_train":list(ORIGINS[:-1]),
        "origin_holdout":ORIGINS[-1],
        "training_scores":train_scores,
        "holdout_scores":holdout,
        "winner_train_only":winner,
        "selected":winner if accepted else "incumbent",
        "accepted":bool(accepted),
        "minimum_improvement":MIN_GAIN,
        "folds":fold_details,
    }


def freeze(data_dir:Path, output_dir:Path, *, phase:int, trees:int=160,
           activity_trees:int=100) -> dict:
    if phase not in (1,2):
        raise ValueError("Bad phase")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Freeze directory must be empty")
    phase0=read_sales(data_dir/"Phase 0 - Sales.csv")
    selection=select_on_phase0(phase0,trees,activity_trees)
    history=read_history(phase,data_dir)
    base=incumbent(history,trees=trees)
    candidates,diagnostics=learned_candidates(history,base,estimators=activity_trees)
    candidates["phase0_selected"]=candidates[selection["selected"]].copy()
    output_dir.mkdir(parents=True,exist_ok=True)
    file=output_dir/"frozen.npz"
    np.savez_compressed(file,**candidates)
    dates=list(pd.date_range(pd.to_datetime(history.columns[-1])+
              pd.Timedelta(weeks=1),periods=HORIZON,freq="7D").strftime("%Y-%m-%d"))
    receipt={
        "phase":phase,"series":len(history),"horizon":HORIZON,
        "candidate_names":list(candidates),"history_cutoff":str(history.columns[-1]),
        "forecast_dates":dates,"selection":selection,"activity":diagnostics,
        "models":{"incumbent_trees":trees,"activity_trees":activity_trees},
        "frozen_sha256":hashlib.sha256(file.read_bytes()).hexdigest(),
        "truth_read_after_freeze_only":True,
        "caveat":"Post-hoc research on known periods; not a new blind test.",
    }
    (output_dir/"manifest.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("VN1_ACTIVITY_FROZEN="+json.dumps({
        "phase":phase,"selected":selection["selected"],"sha256":receipt["frozen_sha256"],
        "probability_summary":diagnostics
    }),flush=True)
    return receipt


def score(data_dir:Path, output_dir:Path) -> dict:
    manifest=json.loads((output_dir/"manifest.json").read_text())
    frozen=output_dir/"frozen.npz"
    if hashlib.sha256(frozen.read_bytes()).hexdigest()!=manifest["frozen_sha256"]:
        raise ValueError("Frozen forecast SHA256 mismatch")
    actual=read_sales(data_dir/f"Phase {manifest['phase']} - Sales.csv")
    history=read_history(manifest["phase"],data_dir)
    if (len(actual)!=manifest["series"] or not actual[KEYS].equals(history[KEYS])
            or list(actual.columns[3:])!=manifest["forecast_dates"]):
        raise ValueError("Target identifiers or dates are unaligned")
    truth=actual.iloc[:,3:].to_numpy(dtype=np.float64)
    with np.load(frozen) as forecasts:
        result={name:_metric(np.asarray(forecasts[name],dtype=np.float64),truth)
                for name in manifest["candidate_names"]}
    receipt={
        "phase":manifest["phase"],"series":len(truth),"scores":result,
        "selected":manifest["selection"]["selected"],
        "historical_gate_accepted":manifest["selection"]["accepted"],
        "frozen_sha256":manifest["frozen_sha256"],
        "no_independent_blind_evaluation":True,
    }
    (output_dir/"scores.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("VN1_ACTIVITY_SCORES="+json.dumps(result,sort_keys=True),flush=True)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=("freeze","score"))
    parser.add_argument("--phase",type=int,choices=(1,2),default=2)
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--trees",type=int,default=160)
    parser.add_argument("--activity-trees",type=int,default=100)
    args=parser.parse_args()
    if args.operation=="freeze":
        freeze(args.data_dir,args.output_dir,phase=args.phase,
               trees=args.trees,activity_trees=args.activity_trees)
    else:
        score(args.data_dir,args.output_dir)


if __name__=="__main__":
    main()
