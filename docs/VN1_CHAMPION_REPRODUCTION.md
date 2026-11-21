# VN1 ARIMA + AMSI 0.469338 champion — independent frozen-score reproduction

## What this reproduction does

This is the **original, unchanged, historically selected** research policy from [PR #17](https://github.com/teodor-georgiev/VN1-Forecasting-Competition/pull/17), later checked again in [PR #20](https://github.com/teodor-georgiev/VN1-Forecasting-Competition/pull/20):

- 60% learned-activity, 30% top-16 client AutoARIMA (StatsForecast, annually seasonal), and 10% last-anchored AMSI seasonal candidate.
- Each candidate is normalized to the incumbent's weekly aggregate forecast volume by the original code. AutoARIMA is **capped at 16 client aggregates**, not fitted on all 15,053 individual SKUs.
- We **do not** claim to reproduce the private implementation of VN1's actual 2024 winner. This is the user's research champion.
- The retrospective Phase 2 result reported in the independently checked research notes is **0.4693382738889349**. It is **not** an unseen competition leaderboard score and was not chosen by a blind submission.

The new `reproduce_vn1_champion_frozen.py` is an *independent scorer* for the original immutable frozen artifact. It deliberately **does not retrain**, adjust 30/10 weights, select a new model, load Phase 2 targets before freezing, or change original research code. It verifies:

1. The **same frozen Phase 2 forecast archive** from [successful run 43060875239](https://github.com/teodor-georgiev/VN1-Forecasting-Competition/actions/runs/43060875239), artifact `vn1-arima-amsi-capped16-evidence`, original compressed bundle SHA-256 `33cdb93ccc2415e9c6697e1074a0e4ecb2ad6066d473480b79a3582aa1122101`.
2. The frozen manifest's selected candidate is **exactly `arima30_client_amsi_last10`**, and `historical_selected` is byte-for-byte equal to that candidate's float64 forecast.
3. Official publisher [SupChains/VN1](https://github.com/SupChains/VN1), immutable source commit `281dccfa449442774690e9ad639a7d9405e6620c`, Phase 2 data blob SHA `714205238d3fcebe8a82b0cec97b28f375914d3b`. The published target is **15,053 series × 13 Mondays**, **3,242,739 total actual units**, aligned exactly by Client/Warehouse/Product order and forecast dates.
4. Independent official calculation: `(sum(abs(pred−actual)) + abs(sum(pred−actual))) / sum(actual)`, where global bias is applied **after aggregating all series and dates**, and the computation uses float64 arrays.
5. Reproduction must equal **0.4693382738889349** within 1e-10, or the automated job **fails**. Only the aggregate score, input revision and hash receipts may be uploaded; no private data, per-SKU predictions or raw sales.
6. Adversarial tests check wrong score formulas, data corruption, date/order mismatch, changed model selection, numeric errors, zero denominator and archive tampering.

## CI workflow

`.github/workflows/vn1-champion-frozen-replay.yml` is strictly read-only. It uses short-lived GitHub-hosted CI, checkout of public pinned datasets, downloaded short-lived original prediction artifact, and `actions:read`/ `contents:read` only. There are no production writes, external API keys, customer data or tenant connections.

**Artifact retention limit:** the original source run retains the frozen archive only until **2026-11-25**. A long-term reproduction after that date must first regenerate the exact pinned freeze using the original code and historical Phase 0+1 input on an approved runner. Do not silently substitute a regenerated artifact without verifying its hash/score.

### Manual replay of downloaded frozen evidence

```bash
python -m pip install -r requirements-vn1.txt
python -m pytest -q test_reproduce_vn1_champion_frozen.py
python reproduce_vn1_champion_frozen.py \
  --frozen-dir /path/to/downloaded/vn1-arima-amsi-capped16-evidence \
  --truth '/path/to/pinned/SupChains-VN1/Phase 2 - Sales.csv' \
  --phase0 'Phase 0 - Sales.csv' \
  --phase1 'Phase 1 - Sales.csv' \
  --report /path/to/new/directory/score.json
```

No score is claimed for this new run until GitHub Actions has completed with a passing exact-head receipt. A previous retry of the **private** Phase 2 scoring job `43062398066` failed before executing any steps, which is an infrastructure failure, not a contradictory forecast score.

## Separate grocery port — not the same score

DemandAhead's synthetic grocery research has **20 stores × 30 SKUs** and **160 historical weeks plus eight private synthetic holdout weeks** in [draft PR #612](https://github.com/teodor-georgiev/DemandoAI/pull/612). The original model is a **13-week forecast**, while grocery has only **eight actual future weeks**. A correct compatibility adapter must:

- map each fictional store to Client and SKU to Product with a stable Warehouse/site identifier, and maintain stable series keys;
- isolate **kg and piece products** throughout training, global-bias correction and scoring; a global scaler mixing kg and pieces is mathematically wrong;
- construct features only from pre-origin history, using dated stock flags for censored sales instead of treating stockouts as genuine zero demand;
- retain the frozen original 60/30/10 champion weights and predict 13 weeks, **then evaluate only the eight matching held-out weeks**, clearly distinguishing that from official 13-week VN1;
- compare against existing grocery baselines, StatsForecast, MLForecast and LightGBM in PR #610 and later inventory cost/fill-rate replay PR #611 on matching cutoffs;
- never claim the **0.469338** original score as a grocery result, and never use the grocery holdout to tune champion weights.

This scoring PR deliberately does **not** vendor the original research code into DemandAhead, change its forecast engine, train on merchant data, issue supplier orders, deploy, or claim actual grocery numerical performance.
