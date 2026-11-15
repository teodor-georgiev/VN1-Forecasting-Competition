# VN1: OOF counterfactual-expert gate (research only)

## Hypothesis

Our best *verified retrospective Phase 2* score remains **0.4691204554267017**
from PR #21 (persistent group bias on the ARIMA+AMSI champion). The recently
tested global-framework baselines and annual top-SKU seasonal adjustments
failed to beat this. Instead of adding another fixed seasonal adjustment,
check whether **simpler per-SKU forecasts have already demonstrated better
absolute errors on fully completed prior OOF forecasts**.

## Predeclared candidate family

Reference: PR #21 **group_unscaled25** persistent-bias incumbent. Candidate
gates use the original champion forecasts and observed truth from at most
the two latest **completed** 13-week OOF windows; each older origin has no
access to observations after its forecast cutoff.

- `oof_recent15`: where historical errors support it, blend 15% of the
  last-13-week observed mean with the PR #21 incumbent.
- `oof_annual15`: 15% of a bounded 50/50 blend of year-ago matched 13
  weeks and last-13-week mean, where OOF validates it.
- `oof_annual30`: same annual expert, 30% weight.

A product is eligible for consideration only in the **top 10% by recent
13-week volume**, with the existing high-volume/active threshold. A simple
expert must beat original-champion L1 error by at least 8% on **each of the
latest two completed OOF windows** (one if only one is available);
at each OOF origin, at least 26 prior observed units and one champion L1
error unit are required. This reliability screen is not a target-leaking
Phase 2 tuning mechanism.

Predictions stay nonnegative. All candidates conserve the PR #21 forecast
**weekly aggregate demand** and leave historical inactive series unchanged.

## Historical evaluation

Full 15,053-series official VN1 error+bias metric, historical Phase 0
training origins 118/131/144 and later origin 157. Choose on *pooled training*
only, then require >=0.001 absolute improvement versus PR #21 incumbent
in pooled training AND origin 157, with at least two earlier training
fold wins. Otherwise reject and keep incumbent unchanged.

**Warning: origin 157 was already examined in prior research.** This
exercise is exploratory and no longer an independent blind holdout.
Do not tune parameters against this result and claim new independent
validation. Phase 1/2 actuals are never opened by this experiment.

## Run

    python -m pip install -r requirements-vn1-specialists.txt
    python -m pytest -q test_vn1_counterfactual_expert_gate.py test_vn1_persistent_bias_specialist.py
    python -u vn1_counterfactual_expert_gate.py --data-dir . --output results/vn1-counterfactual/phase0.json

CI runs full-catalogue Phase 0 only on an isolated GitHub-hosted runner and
uploads **aggregate scores only**, never product-level forecasts or labels.
No changes to the existing champion, production files, main branch,
deployment or live predictions.
