# VN1 top-volume SKU × seasonal specialist (research only)

Goal: improve the existing PR #21 persistent-bias incumbent, not the old
LightGBM baseline. The best verified Phase 2 reference is 0.4691204554267017.
This experiment NEVER reads Phase 1 or Phase 2 actuals.

## Predeclared candidates

- **incumbent:** exactly the PR #21 group_unscaled25 specialist.
- **sku_annual10:** capped previous-year 13-week seasonal ratio on the
  top 5% recent high-volume and historically active SKU series.
- **sku_peer10:** leave-one-out Client/Warehouse/Product seasonal ratios
  blended 40/20/40, applied only to those same high-volume series.
- **sku_hybrid15:** annual plus peer plus 15% of PR #21's trusted
  persistently signed out-of-fold SKU residual.

At origin T, the seasonal reference weeks [T-65,T-52) and
[T-52,T-39) are already observed. Future weeks [T,T+13) never enter
features. Correction rates are bounded at ±15%, forecasts remain
nonnegative and total predicted volume is conserved *per forecast week*.
Inactive series retain their untouched forecasts.

## Evaluation

Use all 15,053 client/warehouse/product series and official VN1 metric.
Completed Phase 0 training fold origins: 118, 131, 144.
Separate later holdout origin: 157.

Choose the minimum pooled training-score candidate before checking holdout.
Accept only if improvement over the PR #21 incumbent is at least 0.001
absolute VN1 on pooled training AND on holdout, plus at least two
improved training folds. Otherwise retain incumbent without promotion.
No Phase 2 target can influence fitting or selection. Only one historical
origin is January-like, so this cannot establish January generalization.
Phase 2 labels are public and not a blind evaluation.

## Reproduce in an isolated approved runner

    python -m pip install -r requirements-vn1-specialists.txt
    python -m pytest -q test_vn1_sku_seasonal_specialist.py test_vn1_persistent_bias_specialist.py
    python -u vn1_sku_seasonal_specialist.py --data-dir . --output results/vn1-sku-seasonal/historical.json --trees 160 --activity-trees 100 --client-cap 16

CI uploads aggregate metric receipts only, not forecasts or labels.
No production changes, merge or model promotion.
