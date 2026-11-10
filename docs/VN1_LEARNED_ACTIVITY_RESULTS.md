# VN1 Learned Demand Activity — Historical Research and Score Audit

**Date:** 2026-11-10  
**Scope:** 15,053 client × warehouse × product series, 13 future weekly targets.  
**Status:** historically selected research candidate; *not* a new blind champion.
No default-branch changes, production deployment or model promotion.

## Official retrospective results

The official VN1 metric is:

\`(sum(abs(forecast - actual)) + abs(sum(forecast - actual))) / sum(actual)\`

**Lower is better.** Both phases use the same candidate recipe and the same
selection from completed Phase 0 historical origins.

| Model | Phase 1 ↓ (Oct–Jan) | Phase 2 ↓ (Jan–Apr) |
|---|---:|---:|
| Original incumbent: LightGBM Tweedie + statistical | 0.5806656324 | 0.4770926569 |
| Previous TSB10 + weekly inactive-demand redistribution | 0.5779581536 | 0.4748590945 |
| **Learned activity, full probabilistic redistribution (activity_p100)** | **0.5744912366** | **0.4729805957** |

The learned model improves the incumbent by about **1.06% relative on Phase 1**
and **0.86% relative on Phase 2**. It also improves the previous exploratory
candidate by ~0.40% relative on Phase 2. These are **retrospective public
target scores**, not new blind leaderboard scores.

## Historical selection: accepted by fixed rule

The candidate family was frozen before opening either phase's targets in this
experiment. The selection is computed from complete Phase 0 rolling windows.

| Historical selection metric | Incumbent | Learned activity_p100 |
|---|---:|---:|
| Pooled training origins 105, 118, 131, 144 | 0.5291661254 | **0.5257013476** |
| Separate holdout origin 157 | 0.4895997033 | **0.4859224233** |

The prespecified acceptance threshold was at least **0.001 absolute**
improvement on **both** pooled training and the single separate historical
holdout. The best calibration-only candidate was \`activity_p100\`. Its
holdout also cleared the gate. The \`phase0_selected\` frozen output is
therefore **activity_p100**, not the incumbent.

The same Phase 0 historical selection is used for both Phase 1 and Phase 2;
no phase target is read inside \`freeze\` or used to choose the model.
Because this approach was developed after earlier Phase 2 public-data
research, this gate is **not independent evidence of blind generalization**.

## Model recipe

- Train a **LightGBM binary classifier** predicting whether a series sells
  at least one unit during the next 13 weeks. Each training example is
  one series at a historical forecast origin.
- Features read strictly pre-origin observed sales: demand intensity and
  occurrence across 4/13/26/52 weeks, lags, time since last event, event
  size, year-ago matched demand, trends, and leave-one-out
  client/client×warehouse/product aggregate history. Known calendar
  features are allowed.
- For any prediction cutoff T, train at origins S with **S+13 ≤ T**.
  For Phase 2 cutoff at 183 weeks, the last training origin is 169
  with target weeks ending by week 182.
- Predict a probability p of future 13-week activity; the selected policy
  scales the incumbent series forecast by **p**, with per-week mass
  renormalization. The total predicted units for each future week remain
  unchanged, so the official global forecast bias penalty is unchanged.
- No hard zeros on products selected by the final model; very unlikely
  series get smaller forecasts rather than guaranteed zero.
- Fixed alternatives were p raised to 0.25/0.5/1, thresholds p<0.15/0.25,
  10% TSB blends and the earlier TSB+dormancy rule.
- At Phase 2 origin, probability median **0.86889**, 5th percentile
  **0.07723**, and 1,964/2,933 series below p=0.15/0.25 respectively.
  These are model estimates, **not** independently verified calibration.

## Independent Phase 2 arithmetic audit

Calculated separately from the frozen NPZ archive and source CSVs, not by
calling the benchmark scorer:

- **15,053 series** and **3,242,739 units actually sold**.
- Original official score: **0.47709265691428454**.
- Selected learned model: **0.47298059568703976**.
- Absolute error: 1,541,966.77 units → **1,528,632.43**,
  an improvement of **13,334.34**.
- Aggregate signed forecast bias: −5,120.20 units for both methods,
  unchanged up to floating point.
- For the **3,054 series with zero sales over the previous 26 weeks**,
  absolute error improved from **10,686.58** to **3,882.09** units.
- For the other 11,999 series, absolute error improved from
  **1,531,280.19** to **1,524,750.33** units.
- Phase 2 archive hash check **passed**.

The subgroup definitions use only sales before the Phase 2 origin. The
subgroup error calculations use public Phase 2 actuals for retrospective
analysis only, not for policy fitting or selection.

## Reproduction and integrity

Source branch: \`feat/vn1-learned-activity-20261110\`, dependent on
\`feat/vn1-soft-dormancy-seasonal-20261110\` (PR #14), itself dependent on
\`feat/vn1-amsi-zero-rules-20261110\` (PR #13).

- Phase 1 frozen SHA256:
  \`c6e7cd695458f50122f84dd3c3b7c7ffeeca043091367fe348c17106804027d6\`
- Phase 2 frozen SHA256:
  \`0496519bc14fc946ed8f286bc57c0d7c7ed5bf280d677eb9a01b4555958fd8db\`
- \`python -m pytest -q test_vn1_learned_activity.py\`:
  **7 tests passed** on the isolated Windows research checkout.
- [GitHub Actions learned-activity checks](https://github.com/teodor-georgiev/VN1-Forecasting-Competition/actions/runs/42902524569):
  **passed** on PR head prior to this documentation update.
- Test coverage: chronological training origins, history-only features,
  deterministic activity probabilities, per-week forecast total conservation,
  no future-label use in candidate generation, historical selection,
  frozen SHA integrity and scoring without actual files.
- Forecasts and model selection are persisted in a frozen NPZ file and
  manifest before a separate \`score\` invocation reads any Phase 1/2
  target labels. The scorer checks hashes, dates, series counts and keys.

Commands from the isolated development repository checkout with the pinned
\`requirements-vn1.txt\` installed:

    python -m pytest -q test_vn1_learned_activity.py
    python vn1_learned_activity.py freeze --phase 2 --data-dir DATA --output-dir NEW_EMPTY_DIR
    python vn1_learned_activity.py score --phase 2 --data-dir DATA --output-dir NEW_EMPTY_DIR

Repeat using \`--phase 1\` and a separate empty output directory.

## Limits and decision

The first two phases' labels are public, and prior experiments have been
iteratively informed by them; **Phase 2 is no longer a fresh blind holdout**.
This is a stronger *historical* result than the previous manual-rule model,
but cannot be promoted based solely on this report.

The next defensible milestone is to freeze the candidate configuration and
evaluate newly acquired, untouched later sales. Before deployment, verify
segment stability, event probability calibration, score improvement over
different seasonal windows, and no unacceptable loss on active/high-volume
series. Until then preserve the current production champion.
