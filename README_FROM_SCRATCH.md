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
