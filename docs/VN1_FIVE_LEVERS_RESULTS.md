# VN1: Five forecasting improvement levers — Phase 0 historical audit

**Date:** 2026-11-11  
**Research branch:** \`feat/vn1-five-levers-20261111\`  
**Status:** all five families implemented; Phase 0 historical selection passed; selected model scored retrospectively on Phase 2 using a SHA-256-pinned frozen archive. The Phase 2 result did **not** improve over learned activity.

## Goal and evaluation

Goal: improve on the verified **0.4729805957 retrospective Phase 2** score from the original learned-activity model; **0.450** is a target, not an observed result.

The official VN1 metric is \`(sum(abs(pred - actual)) + abs(sum(pred - actual))) / sum(actual)\`; lower is better. The historical evaluation uses all **15,053 series** and 13 weekly forecast horizons. Training origins **105, 118, 131, 144** have fully observed 13-week targets within Phase 0, and origin **157** is a separate acceptance holdout. Only the pooled training score ranks candidates; the holdout accepts the single training winner if both improve by **≥0.001 absolute**. All candidates use the same historical learned-activity baseline and a fixed 30% blend of direct model forecasts. Every candidate preserves the base forecast total in each week.

## Verified full historical results

| Fixed experimental policy | Pooled training ↓ | Origin-157 holdout ↓ |
|---|---:|---:|
| **Learned-activity baseline** | 0.5257013476 | 0.4859224233 |
| Loss: L1 regression | 0.5291605481 | 0.4788447667 |
| Loss: Huber regression | 0.6456335171 | 0.6080089290 |
| Loss: 0.60 quantile regression | 0.5303278047 | 0.4805358785 |
| Target: log1p transformed, inverted | 0.5287336488 | 0.4828760776 |
| Target: sqrt transformed, inverted | 0.5246320301 | 0.4836172408 |
| **Target: seasonal baseline normalization** | **0.5226790281** | **0.4759555821** |
| Price: historical observations + Tweedie | 0.5289895862 | 0.4852639551 |
| Price: historical observations + L1 | 0.5281675219 | 0.4786797163 |
| Hierarchy: client × warehouse reconciliation | 0.5267945389 | 0.4833964227 |
| Trim: post-launch leading-zero correction | 0.5268096608 | 0.4862153065 |

### Selection decision

The pooled training winner is **\`target_season_norm\`**, with 0.5226790281 versus learned-activity 0.5257013476 (improvement 0.00302232). The heldout historical score is **0.4759555821** versus 0.4859224233 (improvement 0.00996684). Both exceed the predefined **0.001** absolute threshold: **historical acceptance passed**.

This is the only model selected in the recorded Phase 0 recipe; the isolated holdout benefit of other variants cannot override the training-only selection.

### Interpretation by requested improvement lever

1. **LightGBM objective:** L1 reduces the last holdout's error but does not beat the baseline on pooled training. Huber is materially worse; do not use it as a blanket improvement.
2. **Target transformation:** normalized unit demand relative to the historically available statistical forecast is the one convincing model improvement on *both* selection splits. Log and sqrt alone do not pass both gates.
3. **Price:** the private historical Phase 0 price CSV (read on the authorized Windows desktop) improved the L1 candidate on the final holdout, but still underperforms normalized targets on historical training and holdout. **No private price rows or files were committed or uploaded.**
4. **Hierarchical:** a simple client×warehouse temporal aggregate forecast, reallocated to products, is not consistently better on the earlier training folds.
5. **Leading zeros:** trimming launch-era zeros changes recent-product forecasts but did not improve the overall official metric.

## Reproducibility and safety

- The public branch contains only \`vn1_five_levers.py\`, \`vn1_five_levers_frozen.py\`, regression tests and the scoped CI workflow, plus this aggregate result report.
- Prices were consumed **read-only from the private VN1-Forecasting repository** in a desktop isolated research checkout; the results above are aggregate losses only.
- Synthetic tests cover all five families, no future price/sales exposure, positive-value handling, weekly total conservation, source key alignment, selection split, and **freeze-first** Phase 1/2 scoring with SHA-256 integrity.
- **13/13 local tests passed** in the isolated desktop research checkout on the latest branch contents, using a dedicated scratch path on D: because the desktop's C: filesystem was full.
- Public [GitHub Actions run #43066111096](https://github.com/teodor-georgiev/VN1-Forecasting-Competition/actions/runs/43066111096) successfully completed sales-only historical training/holdout. Its winning normalized-target result agrees with the separate **price-enabled desktop benchmark** to floating-point tolerance.
- Public [GitHub Actions run #43067434042](https://github.com/teodor-georgiev/VN1-Forecasting-Competition/actions/runs/43067434042) passed after the independent freeze-first evaluator was added, including the extended regression test suite.
- Private target/price files and per-series forecasts are **never** included in public CI artifacts.

## Retrospective Phase 2 score — verified on 2026-11-11

The historically selected `target_season_norm` forecast was **frozen using Phase 0 and Phase 1 sales and private observed price history before the evaluation labels were opened**. The immutable archive passed the scorer's SHA-256 integrity check and row/date alignment checks. The Windows process exited with code 0.

| Model | Phase 2 VN1 metric ↓ |
|---|---:|
| **Previously accepted learned activity** | **0.4729805956870398** |
| Phase 0-selected seasonal normalization | 0.47373646975740963 |

- **Result:** the seasonal normalization candidate is **worse by 0.0007558740703698 absolute**, approximately **0.160% relative**, on Phase 2; **do not promote it**.
- Evaluated **15,053 series × 13 future weeks**, with metric matching the official VN1 `functions_for_participants.data_competition_evaluation` formula.
- Frozen forecast SHA-256: `e4b168d3e7ad1395f208752b65ff90cba1487d2511b32750f4201ffb7c4efbbf`.
- Frozen Phase 0 selection: `target_season_norm`; historical gate `accepted: true`. Historical selection was made without Phase 2 labels, and the same archive also contains the learned-activity control, allowing matched comparison.
- Verified with `vn1_five_levers_frozen.py score --phase 2` on the authorized desktop, using the clean research checkout `b0bf08f500a70ae0dd01b6065c4c5d1e07f545dd`. Scorer receipt stored in the desktop scratch directory alongside the frozen archive, not uploaded to public GitHub.
- **Important limitation:** Phase 2 is retrospective and has already been analyzed in previous experiments. This result is **not an untouched blind competition estimate**.
- The added public Phase 1 freeze-and-score GitHub job is separately queued, and its outcome is not needed to validate this locally verified Phase 2 receipt.

No default-branch modification, merge, deployment, or champion replacement. The best currently verified Phase 2 retrospective research result remains **0.4729805957** from learned activity.
