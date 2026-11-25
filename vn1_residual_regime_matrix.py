"""CatBoost, lead-block LightGBM and intermittent-demand research matrix."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
import vn1_champion_residual_oof as base
import vn1_persistent_bias_specialist as persistent
from vn1_from_scratch import HORIZON, KEYS, read_sales

POLICIES = ("incumbent", "catboost25", "lead_lgbm25", "regime_experts", "intermittent25")
MIN_GAIN = 0.001
LANES = POLICIES[1:]


def _policies(lane):
    if lane == "all":
        return POLICIES
    if lane not in LANES:
        raise ValueError("Unknown research lane")
    return ("incumbent", lane)


def _cat_features(frame):
    x = frame.copy()
    for col in ("Client", "Warehouse"):
        x[col] = x[col].astype("int64").astype(str)
    return x


def _residual_models(records, X, policies=POLICIES):
    train = pd.concat([r["X"] for r in records], ignore_index=True)
    labels = np.concatenate([r["residual"] for r in records])
    cat = np.zeros(len(X))
    lead = np.zeros(len(X))
    if "catboost25" in policies:
        model = CatBoostRegressor(iterations=120, depth=5, learning_rate=.05,
            loss_function="MAE", l2_leaf_reg=5, random_seed=2026, thread_count=2,
            verbose=False, allow_writing_files=False)
        model.fit(_cat_features(train), labels, cat_features=["Client", "Warehouse"])
        cat = model.predict(_cat_features(X))
    if "lead_lgbm25" in policies:
        for low, high in ((1, 4), (5, 9), (10, 13)):
            tr = train["horizon"].between(low, high).to_numpy()
            te = X["horizon"].between(low, high).to_numpy()
            if tr.sum() < 100:
                continue
            model = lgb.LGBMRegressor(**base.PARAMS)
            model.fit(train.loc[tr], labels[tr], categorical_feature=["Client", "Warehouse"])
            lead[te] = model.predict(X.loc[te])
    return cat.reshape(-1, HORIZON), lead.reshape(-1, HORIZON)


def _regimes(y):
    active = base._active_mask(y)
    intermittent = ((y[:, -26:] > 0).mean(axis=1) >= .10) & ~active
    return np.where(active, base._regimes(y), np.where(intermittent, 3, 4))


def _experts(y):
    recent = y[:, -13:].mean(axis=1)
    annual = y[:, -52:-39]
    # Smoothed occurrence probability times conditional positive demand.
    window = y[:, -26:]
    count = (window > 0).sum(axis=1)
    probability = (count + 2 * (y[:, -52:] > 0).mean(axis=1)) / 28
    size = np.divide(window.sum(axis=1), count, out=np.zeros(len(y)), where=count > 0)
    return [np.repeat(recent[:, None], HORIZON, axis=1),
            .5 * annual + .5 * recent[:, None],
            np.repeat((probability * size)[:, None], HORIZON, axis=1)]


def _weights(y, records):
    """Select only using completed OOF windows, not current target labels."""
    losses = np.zeros((4, 4))
    support = np.zeros(4)
    for r in records[-3:]:
        prior = y[:, :int(r["origin"])]
        p, a = r["champion"], r["truth"]
        regime = _regimes(prior)
        experts = _experts(prior)
        for k in range(4):
            mask = regime == k
            support[k] += mask.sum()
            losses[k, 0] += np.abs(p[mask] - a[mask]).sum()
            for j, e in enumerate(experts, 1):
                losses[k, j] += np.abs(.75*p[mask] + .25*e[mask] - a[mask]).sum()
    choices = np.argmin(losses, axis=1)
    for k in range(4):
        if support[k] < 100 or losses[k, choices[k]] >= .99 * losses[k, 0]:
            choices[k] = 0
    return choices


def _conserve(proposed, control, mask):
    """Conserve each week's total over eligible pool, leave others untouched."""
    out = control.copy()
    raw = np.maximum(0., proposed[mask])
    needed, got = control[mask].sum(axis=0), raw.sum(axis=0)
    if np.any((needed > 0) & (got <= 0)):
        return out
    out[mask] = raw * np.divide(needed, got, out=np.ones_like(needed), where=got > 0)
    if not np.isfinite(out).all() or (out < 0).any():
        raise ArithmeticError("Invalid proposal")
    if not np.allclose(out.sum(axis=0), control.sum(axis=0), rtol=1e-10, atol=1e-6):
        raise ArithmeticError("Weekly total changed")
    return out


def candidate_forecasts(history, incumbent, records, *, original_champion=None, lane="all"):
    y = base._history_matrix(history)
    n, cutoff = y.shape
    control = np.asarray(incumbent, dtype=float)
    if control.shape != (n,HORIZON) or not np.isfinite(control).all() or (control<0).any():
        raise ValueError("Invalid incumbent")
    origins = [int(r["origin"]) for r in records]
    if not origins or origins != sorted(set(origins)) or any(o<base.OOF_START or o+HORIZON>cutoff for o in origins):
        raise ValueError("Invalid completed OOF chronology")
    for r in records:
        o = int(r["origin"])
        p, a = np.asarray(r["champion"]), np.asarray(r["truth"])
        if p.shape != control.shape or not np.isfinite(p).all() or (p<0).any() or not np.array_equal(a,y[:,o:o+HORIZON]):
            raise ValueError("Invalid OOF truth or forecast")
    policies = _policies(lane)
    active = base._active_mask(y)
    cap = np.maximum(3., 2*y[:, -13:].mean(axis=1))[:, None]
    result = {"incumbent": control.copy()}
    if any(k in policies for k in ("catboost25", "lead_lgbm25")):
        if original_champion is None:
            raise ValueError("Missing current original champion for residual features")
        X = base._regression_features(history, original_champion)
        cat, lead = _residual_models(records, X, policies)
        if "catboost25" in policies:
            result["catboost25"] = _conserve(control + .25*np.clip(cat,-cap,cap), control, active)
        if "lead_lgbm25" in policies:
            result["lead_lgbm25"] = _conserve(control + .25*np.clip(lead,-cap,cap), control, active)
    if any(k in policies for k in ("regime_experts", "intermittent25")):
        regime = _regimes(y)
        experts = _experts(y)
        if "regime_experts" in policies:
            weights = _weights(y, records)
            proposal = control.copy()
            for k in range(4):
                if weights[k]:
                    m = regime == k
                    proposal[m] = .75*control[m] + .25*experts[weights[k]-1][m]
            result["regime_experts"] = _conserve(proposal, control, regime < 4)
        if "intermittent25" in policies:
            result["intermittent25"] = _conserve(.75*control+.25*experts[2], control, regime == 3)
    return result


def historical_validation(phase0, *, trees=160, activity_trees=100, client_cap=16, lane="all"):
    policies = _policies(lane)
    y = base._history_matrix(phase0)
    if y.shape[1] < base.HOLDOUT_ORIGIN + HORIZON:
        raise ValueError("Incomplete historical holdout")
    cache, fold_scores = {}, {}
    totals = {k: np.zeros(2) for k in policies}
    wins = dict.fromkeys(policies, 0)
    demand = 0.0
    for origin in (*base.TRAIN_EVAL_ORIGINS, base.HOLDOUT_ORIGIN):
        past = phase0.iloc[:, :len(KEYS)+origin].copy()
        options = dict(trees=trees, activity_trees=activity_trees, client_cap=client_cap)
        records = base._completed_records(phase0, origin, cache, **options)
        original, _ = base.champion_forecast(past, **options)
        incumbent = persistent.candidate_forecasts(past, original, records)[0]["group_unscaled25"]
        forecasts = candidate_forecasts(past, incumbent, records, original_champion=original, lane=lane)
        truth = y[:, origin:origin+HORIZON]
        volume = float(truth.sum())
        if volume <= 0:
            raise ValueError("Zero validation demand")
        components = {}
        for name, forecast in forecasts.items():
            diff = forecast - truth
            components[name] = np.array([np.abs(diff).sum(), diff.sum()])
        scores = {k: float((v[0]+abs(v[1]))/volume) for k, v in components.items()}
        fold_scores[str(origin)] = scores
        if origin != base.HOLDOUT_ORIGIN:
            demand += volume
            for k, v in components.items():
                totals[k] += v
                wins[k] += int(scores[k] < scores["incumbent"])
        print("VN1_RESIDUAL_REGIME_FOLD=" + str(origin), flush=True)
    train = {k: float((v[0]+abs(v[1]))/demand) for k, v in totals.items()}
    holdout = fold_scores[str(base.HOLDOUT_ORIGIN)]
    winner = min(policies, key=train.__getitem__)
    accepted = bool(winner != "incumbent" and
                    train[winner] <= train["incumbent"] - MIN_GAIN and
                    holdout[winner] <= holdout["incumbent"] - MIN_GAIN and
                    wins[winner] >= 2)
    return {"lane": lane, "incumbent": "PR21 group_unscaled25", "series": len(y), "horizon": HORIZON,
            "training_origins": list(base.TRAIN_EVAL_ORIGINS), "holdout_origin": base.HOLDOUT_ORIGIN,
            "training_scores": train, "holdout_scores": holdout, "fold_scores": fold_scores,
            "training_winner": winner, "train_fold_wins": wins, "accepted": accepted,
            "selected": winner if accepted else "incumbent",
            "used_phase1_labels": False, "used_phase2_labels": False,
            "reused_historical_holdout_exploratory": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("."))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lane", choices=("all", *LANES), default="all")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a research receipt")
    phase0 = read_sales(args.data_dir / "Phase 0 - Sales.csv")
    if len(phase0) != 15053:
        raise ValueError("Full VN1 catalogue required")
    receipt = historical_validation(phase0, lane=args.lane)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
    print("VN1_RESIDUAL_REGIME_RESULT=" + json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
