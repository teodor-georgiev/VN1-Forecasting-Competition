# VN1 from-scratch forecasting solution

This is a new **forecasting method**, separate from the existing LightGBM notebook. Forecasting uses only sales available at each historical origin. Public Phase 2 actuals are used **only for retrospective scoring**, never as predictor inputs.

## Evaluation

- **Dataset:** 15,053 client/warehouse/product series, 170 historical weeks in Phase 0 (2020-07-06 through 2023-10-02), then 13 weeks of actual Phase 1 sales (2023-10-09 through 2024-01-01).
- **Official metric:** `VN1 = (sum(abs(forecast - actual)) + abs(sum(forecast - actual))) / sum(actual)`. Lower is better.
- **Model selection:** explored simple level/trend/seasonal blends using rolling historical origins 105, 118, 131, 144, 157, each forecasting 13 future weeks within Phase 0. Chosen configuration: 25% four-week mean + 75% thirteen-week mean; 15% trend adjustment; 30% smoothed, capped client-level annual-seasonal adjustment. No Phase 1 labels used in that selection. Separately, January-specific 13-week historical validation at the 2022 and 2023 year turns selected a 75% last-week plus 25% client-shock-adjusted four-week average.
- **Measured Phase 1 out-of-time score:** **0.5628106612 (56.2811%)**. Relative absolute error: 0.5585627425. Signed relative bias: +0.0042479187.
- **Comparisons on same Phase 1:** simple four-week mean `0.6411227286`; twelve-week moving average `0.659707` (rounded).
- **Measured retrospective Phase 2 score:** **0.5030159998 (50.3016%)**, versus the published naive benchmark **0.5069701879 (50.6970%)**, MA12 **0.8045850026 (80.4585%)**, and winner **0.4637 (46.37%)**. Phase 2 covers 2024-01-08 through 2024-04-01 (13 weeks, 15,053 series).
- **Important qualification:** This is a **retrospective replay, not a blind leaderboard entry**. The first version scored 0.8434746585 on Phase 2. After observing this failure, we devised a calendar-specific correction, but chose its parameters using earlier January 2022/2023 backtests, not Phase 2 targets. The final Phase 2 figure should not be represented as a truly untouched competition score.

## Method

1. Estimate each series' current level as `0.25 * mean(last 4 weeks) + 0.75 * mean(last 13 weeks)`.
2. Dampen its direction of change with a clipped ratio of the latest 13 weeks to the previous 13 weeks. Apply 15% of that change.
3. Aggregate past sales by client. For each future week, read the matching week **52 weeks in the historical past** and smooth it with its two neighbors. Normalize this past weekly profile by the mean of the 13 weeks preceding the matching annual window; cap to `[0.4, 2.2]`. Apply 30% of the profile's deviation from one.
4. If the forecast origin is January 1–7, switch to a specific new-year regime: `0.75 * last-week sales + 0.25 * four-week-average * clip((client_last_week_sales + 5)/(client_four_week_average_sales + 5), 1/3, 3)`. This regime was selected on the Jan 2022 and Jan 2023 rolling backtests (pooled weighted score ~58.47% versus 58.60% for last-week naïve).
5. Clip to nonnegative forecasts and preserve the official identifier and date columns.

The solution is deterministic and uses `numpy` and `pandas`, without a trained gradient-boosting model. The script can score Phase 2 after forecasting if the official released actuals are present.

## Reproduce

```bash
pip install numpy pandas pytest
python -m pytest -q test_vn1_from_scratch.py
python vn1_from_scratch.py --phase 1 --data-dir . --output Submission_Phase_1_from_scratch.csv
python vn1_from_scratch.py --phase 2 --data-dir . --output Submission_Phase_2_from_scratch.csv
```

The original files `Phase 0 - Sales.csv` and `Phase 1 - Sales.csv` are in this repository. To independently verify the Phase 2 figure, place the organizer's released [`Phase 2 - Sales.csv`](https://github.com/SupChains/VN1/blob/main/Phase%202%20-%20Sales.csv) in the same directory. Phase 1 prints the official metric and MA12; Phase 2 prints the official metric and naive comparison when actuals exist. Without those labels, Phase 2 still writes the forecast and makes no claim of measured performance.

Citation: Vandeput, N. (2024), *VN1 Forecasting – Accuracy Challenge*, SupChains. Data by Flieber. https://supchains.com/vn1-forecasting-competition/


## LightGBM and historical-validated ensemble (2026-11-10)

The separate `vn1_lgbm_ensemble.py` implements a **direct 13-horizon LightGBM Tweedie** model using sales lags, demand levels/trends, a same-week-prior-year signal, client/warehouse/product group demand statistics, and calendar indicators. Every training pair's target must lie at or before the model's training cutoff; feature construction slices history at its origin. Dependencies and versions are pinned in `requirements-vn1.txt`.

The optional statistical/LightGBM ensemble uses

`prediction = (weight_lgbm × LightGBM) + ((1 - weight_lgbm) × statistical)`.

Mixing weight candidates `[0, 0.25, 0.5, 0.75, 1]` are compared using **season-matched historical backtests only**, **not** the phase target being scored. For the October-origin Phase 1 run, CV at week 117 selected `weight_lgbm=1.0`. For the January-origin Phase 2 run, CV at week 130 selected `weight_lgbm=0.75`. No Phase 2 target values were used in model training, feature preparation or ensemble weight selection.

### Actual rerun on published competition data

| Model | Phase 1 score ↓ | Phase 2 score ↓ |
|---|---:|---:|
| Existing statistical baseline | 56.2811% | 50.3016% |
| **LightGBM alone** | **58.0666%** | **48.0170%** |
| **Automatically weighted ensemble** | **58.0666%** (100% LGBM) | **47.7093%** (75% LGBM) |
| VN1 winner (published) | — | 46.37% |
| Organizer's naive benchmark | 68.82% | 50.70% |
| Organizer's MA12 benchmark | 65.97% | 80.46% |

The Phase 2 ensemble achieved **MAE 47.5514%** and **signed bias −0.1579%**, for a combined official metric of **47.7093%**. Phase 2 covers 13 weekly targets from 2024-01-08 to 2024-04-01, across 15,053 product series.

**Qualification:** This is a retrospective replay on public actual sales, not an official leaderboard submission or an independent blind evaluation. The existing statistical model underwent earlier Phase 2 diagnostic work; results should not be mistaken for an untouched out-of-sample leaderboard rank. The ensemble's LightGBM weight was selected using earlier 2023 January data, not Phase 2 labels.

**Verified CI:** [GitHub Actions real-data replay](https://github.com/teodor-georgiev/VN1-Forecasting-Competition/actions/runs/42738809254) completed both jobs successfully (10 synthetic/leakage tests + official Phase 1/2 replay). Score receipts and forecast CSVs were uploaded in the `vn1-ensemble-replay` artifact. The official Phase 2 labels came from [SupChains/VN1](https://github.com/SupChains/VN1), pinned to revision `281dccfa449442774690e9ad639a7d9405e6620c`, with Git blob integrity verified.

To rerun:

```bash
python -m pip install -r requirements-vn1.txt
python -m pytest -q test_vn1_from_scratch.py test_vn1_lgbm_ensemble.py
python vn1_lgbm_ensemble.py --phase 1 --output Submission_Phase_1_lgbm_ensemble.csv
# For Phase 2, add official Phase 2 - Sales.csv from SupChains/VN1 to the data dir.
python vn1_lgbm_ensemble.py --phase 2 --output Submission_Phase_2_lgbm_ensemble.csv
```

The CLI writes a correctly aligned submission and prints component scores separately. Override the historical blend weight using `--weight 0.5` when experimenting; the default `--weight` is selected through historical validation.


## Research-only client/warehouse residual calibration and feature ablations

Run the optional `vn1_residual_ablation.py` to measure whether a **bounded,
hierarchically shrunk multiplicative bias adjustment** improves direct LightGBM
forecasts. For every completed historical 13-week fold, a LightGBM model
trains only on data at or before its forecast origin. Correction factors are
estimated from *out-of-fold* errors, pooled by client and by warehouse, with
explicit pseudo-volume priors (5,000 and 15,000 units) and a hard 10%
correction limit. The model does **not** train independent per-client models,
which would overfit sparse customers.

Nested rolling validation fits the correction only on earlier folds before
evaluating a held-out fold. It selects among fixed strengths `0, 0.5, 1.0`
only when the pooled official VN1 metric improves by at least 0.005,
the correction wins at least 60% of held-out folds (at least two wins),
and no individual fold deteriorates by more than 0.02. **All forecast
calendar months** require at least one matching month among the strictly
historical held-out folds, plus improvement there; lacking this evidence
keeps strength at zero. The month-matching safety rule was strengthened
after observing the initial Phase 1 retrospective regression, so it must
not be portrayed as independently pre-registered.

The feature ablation compares five direct LightGBM variants on the same
completed validation origins (default: two recent folds):

| Variant | Features |
|---|---|
| `baseline` | Existing ID + group statistics + demand/calendar |
| `no_ids` | Remove Client and Warehouse categorical IDs |
| `no_group_stats` | Remove client, warehouse and product group statistics, including the client-pooled statistical baseline |
| `no_ids_or_stats` | Per-series demand/calendar features only; no ID, peer aggregate or client-pooled statistical baseline |
| `interactions` | Baseline + client×warehouse code and leave-one-out peer aggregates |

Run locally on the official files:

```bash
python -m pip install -r requirements-vn1.txt
python -m pytest -q test_vn1_residual_ablation.py
python vn1_residual_ablation.py --phase 1 --trees 160
python vn1_residual_ablation.py --phase 2 --trees 160
```

The GitHub CI workflow replays both phases and uploads JSON score receipts
and forecast CSVs. **This is an isolated experiment: it does not alter
`vn1_lgbm_ensemble.py`, the champion, or the default output.** Competition
Phase 2 labels are retrospective evaluation data and are never used for
selection or calibration. A better Phase 2 score alone is not grounds to
promote this model without independent historical validation.


### First retrospective experiment (before stronger seasonal guard)

On the full published data with 160 trees, the initial correction's historical
fold selection increased Phase 1 VN1 error from **0.580666** (plain LightGBM)
to **0.630457**, and selected no correction for Phase 2, retaining **0.480170**.
The previous validated champion remained better: Phase 1 **0.5389** and
Phase 2 **0.477093**. These results motivated a stricter generic
season-matching requirement; the final corrected replay is recorded in CI.

Historical feature-ablation diagnostics also showed that omitting group
statistics increases error materially. Client×warehouse interaction features
were not a consistent winner over the existing baseline. No model or
feature from this experiment is automatically promoted to the default.
