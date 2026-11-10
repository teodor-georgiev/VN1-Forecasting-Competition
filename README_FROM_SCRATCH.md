# VN1 from-scratch forecasting solution

This is a new **forecasting method**, separate from the existing LightGBM notebook. No future Phase 1 sales or Phase 2 submission labels are used to train the forecasts.

## Evaluation

- **Dataset:** 15,053 client/warehouse/product series, 170 historical weeks in Phase 0 (2020-07-06 through 2023-10-02), then 13 weeks of actual Phase 1 sales (2023-10-09 through 2024-01-01).
- **Official metric:** \`VN1 = (sum(abs(forecast - actual)) + abs(sum(forecast - actual))) / sum(actual)\`. Lower is better.
- **Model selection:** explored simple level/trend/seasonal blends using rolling historical origins 105, 118, 131, 144, 157, each forecasting 13 future weeks within Phase 0. Chosen configuration: 25% four-week mean + 75% thirteen-week mean; 15% trend adjustment; 30% smoothed, capped client-level annual-seasonal adjustment. No Phase 1 labels used in that selection.
- **Measured Phase 1 out-of-time score:** **0.5628106612 (56.2811%)**. Relative absolute error: 0.5585627425. Signed relative bias: +0.0042479187.
- **Comparisons on same Phase 1:** simple four-week mean \`0.6411227286\`; twelve-week moving average \`0.659707\` (rounded).
- **Scope:** these are **Phase 1** scores, not a competition submission or a **Phase 2** score. The original Phase 2 winning score of 46.4% and MA12 baseline of 80.5% are from a different evaluation period and cannot be directly ranked against Phase 1 results.

## Method

1. Estimate each series' current level as \`0.25 * mean(last 4 weeks) + 0.75 * mean(last 13 weeks)\`.
2. Dampen its direction of change with a clipped ratio of the latest 13 weeks to the previous 13 weeks. Apply 15% of that change.
3. Aggregate past sales by client. For each future week, read the matching week **52 weeks in the historical past** and smooth it with its two neighbors. Normalize this past weekly profile by the mean of the 13 weeks preceding the matching annual window; cap to \`[0.4, 2.2]\`. Apply 30% of the profile's deviation from one.
4. Multiply level, trend and seasonal factors; clip to nonnegative forecasts and preserve the official identifier and date columns.

The solution is deterministic and uses \`numpy\` and \`pandas\`, without a trained gradient-boosting model. The script also emits Phase 2 predictions, but does not invent a Phase 2 score.

## Reproduce

\`\`\`bash
pip install numpy pandas pytest
python -m pytest -q test_vn1_from_scratch.py
python vn1_from_scratch.py --phase 1 --data-dir . --output Submission_Phase_1_from_scratch.csv
python vn1_from_scratch.py --phase 2 --data-dir . --output Submission_Phase_2_from_scratch.csv
\`\`\`

The original files \`Phase 0 - Sales.csv\` and \`Phase 1 - Sales.csv\` are in this repository. Phase 1 prints the official metric and the 12-week MA baseline; Phase 2 writes a forecast without a measured score.

Citation: Vandeput, N. (2024), *VN1 Forecasting – Accuracy Challenge*, SupChains. Data by Flieber. https://supchains.com/vn1-forecasting-competition/
