# VN1 champion-relative residual experiment

**Date:** 2026-11-11 · **Status:** implemented, 9/9 regression tests passed, full historical
benchmark completed; **NO challenger accepted**. Research only.

## Starting point

The fixed, historically selected VN1 research leader is the
**30% capped client AutoARIMA + 10% last-anchor AMSI + 60% learned activity**
model from [PR #17](https://github.com/teodor-georgiev/VN1-Forecasting-Competition/pull/17).
Its independently verified, *retrospective* Phase 2 error is
**0.4693382738889349**, on 15,053 series × 13 forecast weeks.
We do **not** claim this as an untouched blind competition submission.

Previous research found that the **top 5% of high-volume series** produce
approximately **58.7% of absolute Phase 2 error**, and that 5,192 currently
active higher-volume series account for approximately **91.5%**. These are
descriptive retrospective diagnostics, *not* an outcome-dependent future
eligibility rule.

## Research change

The file vn1_champion_residual_oof.py replaces the earlier *statistical-baseline*
residual target with a true **historical champion residual**:

    residual(series, forecast_week) =
        actual_sales(forecast_week) - frozen_champion_forecast(forecast_week)

The champion is recreated using the PR17 formula, omitting the unused
ARIMA-SKU candidate fits (not changing the selected client ARIMA/AMSI path).

For historical prediction time T:

1. Compute each completed earlier champion forecast at S = 105, 118, ...
   using only sales observed by S. Include residual labels from S+1..S+13
   only when S+13 is no later than T.
2. Train an L1 LightGBM correction on the **historically active higher-volume**
   series (last-13-week average >=2, nonzero fraction >=0.5).
3. Use predeclared 20% or 35% correction weights, or apply a reliability
   gate learned from one *previous completed* OOF forecast origin. A temporary
   calibration model excludes its own calibration-origin labels.
4. The gate distinguishes stable, volatile, and shifting demand using
   pre-origin features. Require >=100 historical series per regime and
   >=1% historical absolute-error improvement before a nonzero correction.
5. Clamp corrections by observed recent demand; re-normalize each forecast
   week to the champion's aggregate demand, retaining the official bias term.
6. Compare using the **official VN1 metric**, not L1 alone:
   (sum(abs(pred-actual)) + abs(sum(pred-actual))) / sum(actual).

## Historical evaluation

Fully observed chronological Phase 0 training forecast origins: **118, 131,
144**; separate holdout origin: **157**. Earliest complete OOF champion
training origin is 105, reflecting PR17's minimum historical length.
All target windows are fully observed by each corresponding model-fit date.

| Candidate | Phase 0 pooled train ↓ | Holdout 157 ↓ |
|---|---:|---:|
| **Existing ARIMA+AMSI champion** | **0.5345784755** | **0.4811455955** |
| Uniform residual 20% | 0.5348908471 | 0.4820336476 |
| Uniform residual 35% | 0.5362478851 | 0.4833881116 |
| Gated residual 20% | 0.5357729762 | 0.4814270081 |
| Gated residual 35% | 0.5375952202 | 0.4819370076 |

**Winner chosen from historical training:** unchanged champion.
**Acceptance gate:** accepted=false; selected=champion.
No new challenger outperforms on either pooled training or heldout origin
157. The calibration-based gating reduces the damage of some corrections
but still does not pass.

We will **not** promote a challenger or chase a Phase 2 retrospective score
on a policy already rejected by the chronology-preserving gate.

These values are the model's full forecast for 15,053 series × 13 weeks
at each specified origin; they are not best-of-holdout hyperparameter fits.
The holdout metric 0.4811456 is close to PR17's original reported
0.4811407; the full historical model path was regenerated rather than
reuse per-origin files from that separate artifact. A full-catalogue identity
audit was initiated independently to verify numerical reproducibility.
The 9 passing tests include an exact formula comparison under capped
client_cap=0, which does not establish identity under a 16-client fit.

## Checks and reproducibility

- 9/9 local synthetic/integration tests passed in the isolated Windows
  development research checkout, exact branch head from
  [PR #20](https://github.com/teodor-georgiev/VN1-Forecasting-Competition/pull/20).
- [Successful GitHub Actions historical run #43107881683](https://github.com/teodor-georgiev/VN1-Forecasting-Competition/actions/runs/43107881683):
  unit tests green, Phase 0 full-catalogue rolling and aggregate-only
  historical score receipt green.
- Frozen forecast creation and score paths are separate, with SHA-256 and
  series/date alignment checks; private Phase 2 truth cannot be opened by
  the freeze operation. No private source data or per-series forecasts uploaded.
- This is a **stacked draft PR above PR #17**, not a default-branch
  replacement. No merge, deployment, or production model promotion.

## Conclusion / next hypothesis

A single L1 regressor predicting raw champion residuals does not
generalize enough here. Future research should test whether **structured
group-level demand dynamics and event/price signals** can predict
systematic rather than per-SKU idiosyncratic errors, with sufficiently
long chronological validation and no Phase 2-driven candidate selection.

The best verified retrospective Phase 2 result remains **0.4693382739**.
