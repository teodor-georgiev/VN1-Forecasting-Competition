# VN1 hierarchical demand correction (research only)

**2026-11-12.** Branch stacked on PR #21. The reference to beat is the
previously verified persistent-bias specialist, not the original ARIMA + AMSI
champion. Original frozen Phase 2 score: **0.4693382738889349**; current
research best with persistent-bias correction: **0.4691204554267017**.

## Fixed, causally defined policies

1. `champion`: unchanged original ARIMA + AMSI policy, as a diagnostic.
2. `persistent_unscaled25`: existing #21 validated group-bias specialist.
3. `global_bias20`: a 20% correction of smoothed aggregate signed bias,
   applied to active series; no scaling to force old global totals.
4. `clientwarehouse25`: direct 25% correction of mean client/warehouse
   signed-bias rates, retaining inactive product forecasts.
5. `hier_clientwarehouse25`: Stage 1 estimates active client-level total
   demand using 25% of a shrinkage-blended client/global bias estimate.
   Stage 2 reallocates that total between active products using their champion
   weekly allocations with 20% warehouse-level adjustments.
6. `hier_sku25`: the same client total as (5), but product allocation also
   uses a 20% adjustment for *historically stable*, high-volume SKU errors.

All rates derive from at most the three most recent **completed** 13-week
out-of-fold champion predictions and their associated observations. The
previous specialist already rejects incomplete windows, duplicate/misordered
origins, nonfinite sales, and future-labelled folds. Historical eligibility,
client/warehouse membership and top-volume SKU stability are computed only
from observations available at the prediction cutoff.

No model receives retrospective Phase 2 labels. The Phase 2 baseline must
use original immutable PR17 frozen predictions for a fair comparison.

## Acceptance protocol

- Fit/compare on Phase 0 origins 118, 131, 144; select the single lowest
  pooled VN1 error candidate on these three windows **before** testing.
- Independent later Phase 0 origin 157 must reduce VN1 error by >=0.001.
- Phase 1 must independently reduce error by >=0.0002, with a forecast
  calculated using Phase 0 only. Phase 1 is a veto, not a hyperparameter grid.
- Historical pooled error must reduce by >=0.001 relative to the **existing
  persistent-bias incumbent**, with >=2 winning Phase 0 training folds.
- Otherwise the selected strategy is the existing persistent-bias specialist;
  no hierarchy is promoted on a regression.

Report each candidate's **absolute error, signed bias, absolute bias penalty,
and combined official VN1 score**. Product-level errors and private targets
must not be uploaded as workflow artifacts.

The experiment is allowed to fail. A strong Phase 0 gain that does not
transfer to Phase 1 is a rejection, not an excuse to tune using Phase 2.

## Reproduction

```bash
python -m pytest -q test_vn1_hierarchical_demand.py test_vn1_persistent_bias_specialist.py
python -u vn1_hierarchical_demand.py --data-dir . --output results/hierarchical/historical-phase1.json
```

Run on the **full 15,053-series catalogue**; no 25-item or synthetic-data
benchmark substitutes for the historical full-catalogue score.

This is research-only: no production code, deployment or merge.
