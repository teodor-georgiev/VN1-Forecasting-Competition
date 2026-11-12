# VN1 persistent-bias specialist (candidate, not promoted)

Date: 2026-11-12. This is a **new, unverified historical experiment** layered
on the existing ARIMA+AMSI champion and the first failed OOF residual model
in PR #20. No Phase 2 data is used for fitting, candidate choice, or tuning.

## Motivation

The strongest verified retrospective Phase 2 research score is
**0.4693382738889349**. Phase 2 descriptive audits show that the top 5%
of products account for 58.7% of absolute error; active higher-volume
products account for 91.5%. These figures are not themselves evidence
that a residual predictor can improve accuracy.

The first true champion-relative LightGBM residual approach worsened both
train and holdout, so the next hypothesis is a **much lower-capacity model
of persistent demand bias** rather than another flexible residual regressor.

## Method

* At each historical cutoff, build genuine champion forecasts made at older
  origins using only data available at those origins. A record contributes
  error labels only if its entire 13-week horizon has been observed.
* Estimate smoothed client and warehouse signed errors from the three most
  recent completed OOF windows. Apply group correction to historically
  active series only.
* Estimate per-product correction on the historical top 5% by recent
  volume only when **two completed OOF windows have the same material
  residual sign**. Shrink by forecast volume and two horizon-equivalents
  of recent sales; cap model error rates to +/-25%.
* Compare fixed policy strengths; most policies preserve the champion's
  aggregate forecast by week. One deliberately unscaled group variant
  measures whether a reliable aggregate bias correction helps the *official*
  metric, including the absolute signed-error penalty.
* Select on the official VN1 pooled metric for Phase 0 origins 118, 131,
  144. Origin 157 is a separate acceptance gate. Require at least 0.001
  absolute improvement on both pooled train and holdout plus at least two
  better training folds; otherwise **retain the champion exactly**.

## Run

```bash
python -m pytest -q test_vn1_persistent_bias_specialist.py test_vn1_champion_residual_oof.py
python -u vn1_persistent_bias_specialist.py --data-dir . --output results/vn1-persistent-bias/historical.json
```

The GitHub Actions workflow runs both checks and uploads aggregate-only
historical scores. It never uploads sales labels or individual forecasts.

## Limitations

With only a few completed 13-week origins, historical bias may be
time-varying rather than persistent. All five candidate policies are
predeclared; their thresholds and strengths must **not** be selected
using retrospective Phase 2 labels. This branch is research-only:
no default-branch change, no deployment, and no claim of a better score
until the complete historical validation is green.
