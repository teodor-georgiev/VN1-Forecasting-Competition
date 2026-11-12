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

## Full-catalogue validation result (2026-11-12)

**Completed CI:** https://github.com/teodor-georgiev/VN1-Forecasting-Competition/actions/runs/43167677043

Both jobs completed successfully: causal/eligibility unit tests and full
historical + Phase 1 replay. **No hierarchical challenger accepted**.

| Policy | Phase 0 pooled train | Phase 0 later holdout | Independent Phase 1 |
|---|---:|---:|---:|
| Original ARIMA+AMSI champion | 0.5345784755 | 0.4811455955 | 0.5704760120 |
| Existing persistent-bias incumbent | **0.5280916828** | **0.4739655397** | 0.5710679477 |
| Global bias 20% | 0.5298894267 | 0.4768075096 | **0.5694310541** |
| Client/warehouse direct 25% | 0.5280916828 | 0.4739655397 | 0.5710679477 |
| Hierarchical client/warehouse | 0.5281006326 | 0.4739564101 | 0.5708531418 |
| Hierarchical + stable high-volume SKU | **0.5280346177** | 0.4740082516 | 0.5709280826 |

The *training-only* winner was `hier_sku25`, but its improvement
relative to the existing incumbent was only 0.0000571 in pooled training.
It **worsened** the later Phase 0 holdout by 0.0000427, and its Phase 1
improvement was below the predefined 0.0002 minimum. The frozen
decision is therefore `selected=persistent_unscaled25`, `accepted=false`.

The global correction had a better Phase 1 score, but cannot be selected
retrospectively based on that holdout because it had already lost the
historical Phase 0 selection. Tuning or selecting it after reading Phase 1
would violate this experiment's protocol.

**No new Phase 2 score was produced.** The best verified retrospective
Phase 2 score remains **0.4691204554267017** from PR #21.

### Diagnostic interpretation

The hierarchy offers small differences from the simpler group-bias
specialist. In the original champion versus incumbent historical training
comparison, the incumbent's normalized absolute error increased
from 0.50377214 to 0.50617189, while the absolute signed-bias penalty
decreased from 0.03080633 to 0.02191979. This shows why optimizing only
individual product L1 error can miss the true official-score driver.
Phase 1 has a different bias pattern and underscores non-stationarity.
Further group-bias work needs more independent chronological folds or
new causal features; this exposed holdout cannot be reused to tune policies.
