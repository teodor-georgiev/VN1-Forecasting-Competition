# VN1 AMSI and history-only zero-rule replay (research, November 10, 2026)

**Status:** Retrospective Phase 2 completed on all 15,053 series × 13 weeks.
This is not a blind leaderboard submission. No model was promoted or deployed.

## Historical protocol

- Source history: Phase 0, 170 weeks; frozen Phase 2 forecast additionally uses Phase 1
  (13 weeks). Phase 2 truth is opened **only** by the separate \`score\` step
  after the candidate archive, selection, and SHA256 manifest are frozen.
- All decisions use Phase 0 historical origins **131 and 144** for pooled
  training comparison, followed by independent historical origin **157**
  as an acceptance gate. Every training target exists at its origin cutoff.
- The gate requires **0.001 absolute score improvement** in both historical
  calibration and the final holdout. No candidate passed the fixed gate, so
  \`phase0_selected\` remains \`incumbent\`.
- Models: 30% anchored-seasonality (AMSI) blends, never-sold suppression,
  26/52-week inactive filters, annual matched-week guard, client×warehouse
  closure mask, and explicitly exploratory bias-neutral redistribution.
- AM**SI is not the competition winner's private solution**. The organizer
  reports that the winner's ensemble included ~30% ARIMA; that method is
  not reproduced by these experiments.

## Official metric on retrospective Phase 2

The VN1 metric is (absolute error + absolute *aggregate* bias) divided by
actual units. **Lower is better.** The tiny discrepancy between the previously
verified 0.477092656914 incumbent and the frozen replay score below comes
from deliberate float32 serialization of the candidate archive.

| Candidate | Phase 2 official score | Selected? |
|:--|--:|:--:|
| Existing incumbent | **0.4770926583** | Yes |
| Warehouse closed 26w (hard zero) | 0.4770950291 | No |
| Inactive 52w (hard zero) | 0.4771462975 | No |
| Inactive 26w + same-season-zero (hard zero) | 0.4771543168 | No |
| AMSI last anchor, 30% blend | 0.4824957579 | No |
| AMSI short anchor, 30% blend | 0.5255838712 | No |
| AMSI mean-13 anchor, 30% blend | 0.5563857667 | No |
| Inactive 52w, reallocate globally | **0.4759871010** | No |
| Inactive 26w + annual guard, reallocate globally | **0.4758588214** | No |
| Inactive 52w, reallocate within client | **0.4760291417** | No |
| Warehouse closed 26w, reallocate within client | 0.4770570534 | No |

The new global redistribution candidate is about **0.26% better relative**
than the incumbent on retrospective Phase 2, but is *not an accepted champion*.
It was devised after observing Phase 2 diagnostics; therefore Phase 2 is not
a blind, untouched test for it. Independent forward validation is required.

Historical validation for \`inactive26_reallocate_global\`:
- Calibration origins 131/144: 0.5024208230 vs incumbent 0.5042436971
  (improvement 0.001823).
- Holdout origin 157: 0.4894882506 vs incumbent 0.4895997033
  (improvement 0.000111, **less than 0.001** acceptance threshold).
- Best overall historical calibration candidate was plain
  \`inactive26_annual_zero\`; it failed holdout, so the frozen selection
  remained the incumbent.

## Phase 1 versus Phase 2: season matters

The *same frozen candidate definitions* were independently replayed against
Phase 1 after selecting on the Phase 0 rolling folds. Phase 1 corresponds
to the October-to-January holiday period; Phase 2 is January-to-April.
These two phase targets were scored in their own separate \`score\` processes
after freezing; they were not used to select policies.

| Candidate | Phase 1 score | Phase 2 score |
|:--|--:|--:|
| Incumbent | 0.5806656337 | 0.4770926583 |
| **Inactive 26w + annual guard, redistribute globally** | **0.5783498559** | **0.4758588214** |
| Inactive 52w, redistribute globally | 0.5787410073 | 0.4759871010 |
| Inactive 52w, redistribute within client | 0.5788521855 | 0.4760291417 |
| AMSI last-week anchor, 30% blend | **0.5338301773** | 0.4824957579 |
| AMSI short anchor, 30% blend | 0.5550314113 | 0.5255838712 |
| Plain zero after 26w + season guard | 0.5738361962 | 0.4771543168 |
| Warehouse closed 26w (hard zero) | 0.5805512496 | 0.4770950291 |

This strengthens the business hypothesis that a zero-demand rule must preserve
forecast volume, but it **does not make the candidate a new champion**. The
redistribution idea was motivated by inspection of Phase 2 diagnostics and
the original predeclared Phase 0 holdout gate still fails. A truly independent
forward-period evaluation is required before promotion.

AMSI's strong Phase 1 gain but Phase 2 regression makes a
*seasonally validated policy* a better next hypothesis than a fixed 30% blend.
Such a policy must be selected on multiple historical holiday/nonholiday folds,
not hand-picked from these retrospective phase scores.

## Why plain zero suppression hurt the official score

At the Phase 2 origin, 1,726 series had no sales in the previous 52 weeks.
Of those, 120 sold during Phase 2 (912 total units). The incumbent forecast
4,960.9 units for the group, with 5,699.0 absolute error; forcibly zeroing
those rows yields only 912.0 absolute error for that group, **but also removes
4,960.9 forecast units globally**. The global bias penalty negated that
local benefit. In contrast, bias-neutral reallocation removes the dormant
forecast but proportionally redistributes its volume among active series
while keeping the original global forecast total unchanged.

Other origin-only counts: 2,056 series inactive 26 weeks with zero
matched annual window; 344 series in client+warehouse groups with zero
recent group sales and sufficient earlier group activity; zero series
were inactive for all 183 observed weeks. Never equate zero observed
sales with confirmed lack of stock or permanent discontinuation.

## Verification

- \`python -m pytest -q test_vn1_amsi_zero_rule_ablation.py\`:
  **6 passed** on the authorized isolated desktop checkout at GitHub head
  \`3f2eaf1b5efec38a63fc0d0faf76399e9bba0c3f\`.
- Source changes are isolated on branch
  \`feat/vn1-amsi-zero-rules-20261110\`.
- Real Phase 2 labels were verified against pinned Git blob
  \`714205238d3fcebe8a82b0cec97b28f375914d3b\`.
- Second-pass frozen candidate SHA256:
  \`65f8dbc0246bc83b61156487ec56bbedc5ffd6c5c861b1c89317eb805847b082\`.
- Source repository: \`teodor-georgiev/VN1-Forecasting-Competition\`.
  Original anchored-seasonality module recovered from research
  commit \`2ff1a20b2f7f1d656fd120d691a41e0966724f5d\`.
- The direct desktop benchmark ran without GitHub Actions because the account
  currently has an Actions billing/spending restriction; **no CI gate was bypassed
  for merging**. PR stays draft and unmerged.

## Reproduce without target leakage

Use the pinned Python dependencies and an isolated research directory:

    python -m pytest -q test_vn1_amsi_zero_rule_ablation.py
    python vn1_amsi_zero_rule_ablation.py freeze --phase 2 --data-dir DATA_DIR --output-dir EMPTY_OUTPUT_DIR
    python vn1_amsi_zero_rule_ablation.py score --phase 2 --data-dir DATA_DIR --output-dir EMPTY_OUTPUT_DIR

The \`freeze\` command refuses to overwrite an existing forecast bundle. The
\`score\` command validates the frozen bundle's SHA before reading actuals.
Do not tune policies on the resulting Phase 2 scores or promote a retrospective
winner without new independent chronological evidence.
