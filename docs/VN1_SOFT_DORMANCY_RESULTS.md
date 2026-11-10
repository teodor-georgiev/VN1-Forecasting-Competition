# VN1 soft dormancy and TSB reallocation: retrospective research receipt

**Date:** 2026-11-10  
**Status:** Phase 0 historical selection completed; separate Phase 1 and Phase 2
freeze+score replays completed on all **15,053 series × 13 weeks** each.

**Interpretation:** exploratory; no validated champion update, merge or production change.
Lower official VN1 score is better:
\`(sum(abs(forecast - actual)) + abs(sum(forecast - actual))) / sum(actual)\`.

## Results

| Fixed candidate | Phase 1 ↓ (Oct–Jan) | Phase 2 ↓ (Jan–Apr) |
|---|---:|---:|
| **Historical LightGBM + statistical incumbent** | **0.5806656324** | **0.4770926569** |
| Hard zero on 26w dormant+annual-zero, global mass reallocation | 0.5783498563 | 0.4758588216 |
| Hard zero on 26w dormant+annual-zero, weekly mass reallocation | 0.5784282721 | 0.4758418997 |
| Hard zero on 13w dormant+annual-zero, weekly mass reallocation | 0.5778265627 | 0.4752530920 |
| 10% TSB mixed into baseline, total volume preserved | 0.5799996278 | 0.4760244049 |
| 25% TSB mixed into baseline, total volume preserved | 0.5820563918 | 0.4787150641 |
| **10% TSB + 26w dormant weekly reallocation** | **0.5779581536** | **0.4748590945** |

The best **exploratory** Phase 2 candidate scores **0.4748591**, a ~0.47%
relative error reduction versus 0.4770927. The same candidate improves
Phase 1 by ~0.47% relative as well. We did **not** tune a weight or filter on
Phase 2 labels in code, but these rule families were devised after observing
earlier Phase 2 retrospective results. Neither phase can be treated as a new
blind holdout, and performance on future independent periods is unproven.

## Selection and leakage controls

- Forecast creation is a \`freeze\` invocation which opens only Phase 0 sales
  plus historical Phase 1 observations when forecasting Phase 2.
- Historical training/selection uses **Phase 0** origins
  **105, 118, 131, 144** (all labels fully observed within Phase 0). Final
  origin **157** is a distinct retrospective acceptance holdout.
- Candidate registry and weights are fixed in code before the \`score\`
  command opens Phase 1 or Phase 2 truth.
- Frozen outputs are float64 NPZ archives with an SHA256 hash in each
  phase's \`manifest.json\`; independent \`score\` rejects mismatched hash,
  row keys, future dates, or series count. No modifications to original
  champion code, production forecast, or existing defaults.
- Original historical acceptance rule: **at least 0.001 absolute** improvement
  on both pooled calibration origins and a separate validation origin.
  Candidates are ranked by *calibration only*, and **only the top-ranked
  calibration candidate** is tested for acceptance (not an adaptive
  post-hoc sweep of validation results).
- Pooled calibration: incumbent \`0.5291661254\`; best training candidate
  \`quiet13_hard_weekly\` \`0.5266793056\`.
- Holdout 157: incumbent \`0.4895997033\`; selected training candidate
  \`quiet13_hard_weekly\` \`0.4890747194\`, improvement \`0.000525\`,
  **smaller than the required 0.001**.
- The gated selected candidate remains **incumbent**, despite the
  retrospective Phase 2 improvements. Historical selection is not proof of
  forward generalization.

## Cohort and model details

The Phase 2 origin had 2,056 series inactive for the past 26 weeks with
zero sales in the corresponding seasonal window 52 weeks earlier; 1,726
series inactive for 52 weeks; 2,793 series inactive 13 weeks with no matched
annual activity. These are **sales-history-only heuristics**, not verified
stockout/retirement labels.

TSB is Teunter–Syntetos–Babai intermittent-demand forecasting, computed
from previous-period demand event probability and nonzero sizes. A 10% TSB
mix is rescaled to preserve the incumbent's total forecast volume. Inactive
eligible rows can be reduced to zero; their mass is redistributed across the
remaining series **within each forecast week**. This does not require
future targets, inventory visibility or a list of arbitrary product IDs.

No guarantee that a client×warehouse or product with zero past sales will
continue to have zero future sales. The selection gate protects against
changing the active champion on insufficient evidence.

## Verification and reproduction

- [GitHub Actions VN1 Soft Dormancy Research Checks](https://github.com/teodor-georgiev/VN1-Forecasting-Competition/actions/runs/42899113293):
  **completed successfully** at PR-head commit
  \`f5736456ee0f5b5fdda7fbd8d5db8ef2b6a91f9b\`.
- Isolated Windows desktop: \`python -m pytest -q test_vn1_soft_dormancy.py\`:
  **7 passed**, no test failures.
- Full Phase 2 frozen archive SHA256:
  \`5bc3a2d2a80db52492b6d5435eb542d23dc761d97aa4ecd7e48da49181cb2b4f\`.
- Full Phase 1 frozen archive SHA256:
  \`b5e0f36c34652285391cc65fafdd5b63668b0b5fbb62a6275e796ac8d92352a5\`.
- Both full-catalogue score receipts were obtained by separate \`score\`
  invocations after archives existed, using official published labels.

Commands (with the pinned repo requirements, and the published Phase 0/1
sales CSVs; Phase 2 truth present only when scoring):

    python -m pip install -r requirements-vn1.txt
    python -m pytest -q test_vn1_soft_dormancy.py
    python -u vn1_soft_dormancy.py freeze --phase 2 --data-dir DATA --output-dir EMPTY_DIR
    python -u vn1_soft_dormancy.py score --data-dir DATA --output-dir EMPTY_DIR

Repeat with \`--phase 1\` and a different empty output directory. Never
overwrite existing forecasts or promote a retrospective winner based solely
on public Phase 2 labels.

## Next validation milestone

Reserve genuinely later, untouched weekly sales (or a newly obtained
historical period not used when inventing this candidate family). Freeze the
selection method, candidate grid, data hashes and threshold **before**
opening such labels. Require a win in the new forward window with acceptable
seasonal/segment error dispersion before any champion change.
