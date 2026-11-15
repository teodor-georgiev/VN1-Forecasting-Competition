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

## Verified complete Phase 0 result (2026-11-15)

[Successful CI run #43735586625](https://github.com/teodor-georgiev/VN1-Forecasting-Competition/actions/runs/43735586625): isolated contracts passed, full 15,053-series replay completed, and the gate was independently recalculated from aggregate score JSON; artifact `vn1-counterfactual-phase0-scores` was retained.

| Predeclared model | Pooled origin 118/131/144 training ↓ | Origin 157 historical holdout ↓ |
|---|---:|---:|
| PR21 incumbent | 0.5280916828 | 0.4739655397 |
| OOF recent mean 15% | 0.5280968618 | **0.4736093686** |
| OOF year-ago 15% | 0.5268316556 | 0.4738215383 |
| OOF year-ago 30% | **0.5259158367** | 0.4737708283 |

The training-only winner was `oof_annual30`: training gain
**0.0021758461**, but later holdout gain just **0.0001947113**,
which fails the fixed **0.001 absolute holdout improvement** minimum.
CI verified the frozen decision **`accepted=false, selected=incumbent`**.
The OOF recent-mean 15% arm had the best late-fold score, but it
worsened pooled training, and choosing it after seeing that holdout
would be inappropriate. No Phase 2 comparison was run and no new
champion is claimed. Because origin 157 has been used by earlier research,
this exploratory gate cannot establish independent generalization.

**Decision:** retain the existing 0.4691204554 Phase 2 retrospective
research champion; leave the new code in an unmerged draft PR. A genuinely
new later target window is needed for fresh validation.
