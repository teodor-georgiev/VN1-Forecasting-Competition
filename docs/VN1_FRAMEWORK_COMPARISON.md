# VN1 forecasting-framework comparison (research only)

Compare three actual library adapters against the **official VN1 metric**, not against
each library's own metric. The script does **not** modify the incumbent,
submission files, or any production pipeline.

## Verified 2026-11-15 results

The full runs completed on GitHub-hosted isolated CPU runners. Both ran
against all **15,053 client/warehouse/product series**, forecasting **13 weeks**
with the official error-plus-absolute-global-bias metric (lower is better).

| Framework recipe | Phase 0 historical fold 157 | Phase 2 retrospective |
|---|---:|---:|
| sktime NaiveForecaster (13-week mean) | 0.5224954971254803 | 0.8131824077399716 |
| Darts LightGBM160 (13 lags) | 0.5152173124693071 | 0.7211724737957992 |
| MLForecast LightGBM160 (13 lags) | 0.5152173124693071 | 0.7211724737957992 |

The *existing, separately verified* Phase 2 persistent-bias specialist is
**0.4691204554267017** ([PR #21](https://github.com/teodor-georgiev/VN1-Forecasting-Competition/pull/21)).
These simple three recipes do not beat it. Comparing this champion to the
Phase 0 fold would be invalid; the Phase 2 column is the matched period.

- [Full historical fold 157 CI run #43727126832](https://github.com/teodor-georgiev/VN1-Forecasting-Competition/actions/runs/43727126832): passed; tests and every full-model step succeeded.
- [Full Phase 2 freeze-then-score CI run #43727609001](https://github.com/teodor-georgiev/VN1-Forecasting-Competition/actions/runs/43727609001): passed; all forecasts and SHA-256 manifests existed before published Phase 2 labels were retrieved, the source Git blob integrity passed, and all three outputs passed alignment and official score checks.
- [Earlier 256-series pilot #43726833973](https://github.com/teodor-georgiev/VN1-Forecasting-Competition/actions/runs/43726833973) was a smoke test only and its results are not comparable to full-catalogue metrics.
- Exact CPU dependency pins: sktime 0.40.1, Darts 0.48.1, MLForecast 1.1.0, scikit-learn 1.7.2.
- No champion replacement, no production model modification, no merge, no deployment.

These are *representative forecasting recipes*, not equivalent tests of a
library's maximum accuracy. The sktime arm is a local statistical baseline,
while Darts and MLForecast implement global LightGBM with matching 13-lag
designs and 160 estimators. Their identical forecasts indicate no distinct
accuracy gain from switching libraries under this configuration; MLForecast
was much faster on the two full-catalogue replays. The positive Phase 2
signed forecast bias (+0.15285 for Darts and MLForecast; +0.20448 for sktime)
is a large contributor to their official Phase 2 scores.

**Research qualification:** Phase 2 actuals were published previously and
have informed earlier work; this is *retrospective*, not an independent blind
competition submission. There is no evidence that any of the three simple
recipes improves the existing champion.

## Methodology

- Original data: `Phase 0 - Sales.csv`, with 15,053 series × 170 weeks.
- Historical cutoffs: weeks 131, 144, and 157 of Phase 0, each forecasting
  the next 13 weeks using *only* the observations before its own origin.
- Pilot: same deterministic 256-series sample for all frameworks, seed 42.
  Full experiment: `--sample-size 0`; do not extrapolate pilot scores to full.
- sktime: one local 13-week-mean `NaiveForecaster` **per series**.
- Darts: global `LightGBMModel`, 13 lags, recursive 13-week forecast,
  160 estimators.
- MLForecast: global LightGBM regressor with the same 13 lags, 160 estimators
  and comparable hyperparameters.
- Score = (sum of absolute per-cell errors + absolute sum of signed errors)
  / observed total sales. Clip negative forecasts to zero for all adapters.
- Note: **these are representative recipes, not identical statistical
  models**. sktime uses a *local statistical baseline*; the other two
  are *global ML models*. Scores cannot demonstrate that one entire
  framework is intrinsically better. A later sktime pooled-regression arm
  could improve model-family parity.
- No future targets enter training, features, hyperparameter selection,
  or model choice. Open released Phase 2 actuals only for retrospective
  reporting after the historical candidates have been fixed.
- Phase 2 retrospective replay is **not blind holdout**: public labels have
  already informed earlier research. New external data would be required
  to establish truly prospective gains.

## Reproduction (approved isolated runner)

Install dependencies into a fresh environment:

```bash
python -m pip install -r requirements-vn1-frameworks.txt
python -m pytest -q test_vn1_framework_benchmark.py
python vn1_framework_benchmark.py --origin 131 --sample-size 256 --output-dir /tmp/vn1_fw_131
python vn1_framework_benchmark.py --origin 144 --sample-size 256 --output-dir /tmp/vn1_fw_144
python vn1_framework_benchmark.py --origin 157 --sample-size 256 --output-dir /tmp/vn1_fw_157
```

After successful pilot checks, run full catalogue on an approved host:

```bash
python vn1_framework_benchmark.py --origin 131 --sample-size 0 --output-dir /tmp/vn1_fw_full_131
python vn1_framework_benchmark.py --origin 144 --sample-size 0 --output-dir /tmp/vn1_fw_full_144
python vn1_framework_benchmark.py --origin 157 --sample-size 0 --output-dir /tmp/vn1_fw_full_157
```

For Phase 2, the training invocation **cannot access Phase 2 targets** and
writes a SHA-256 manifest of the immutable forecasts. Use a new empty output
directory and DO NOT place the actuals in the forecasting process:

```bash
python vn1_framework_benchmark.py --phase2 --sample-size 0 --output-dir /tmp/vn1_fw_phase2
```

Only **after** forecasts and their manifest have been saved, retrieve the
pinned released `Phase 2 - Sales.csv` and run the separate scorer:

```bash
python vn1_framework_benchmark.py --score-phase2 --output-dir /tmp/vn1_fw_phase2
```

The scorer verifies source-file and forecast SHA-256 hashes, sample indices,
series identity, and target dates before writing `phase2_scores.json`.
Repeated scoring cannot overwrite the prior receipt. Historical scores
are written to `scores.json`, along with aligned CSVs. Note: even this
separate Phase 2 scoring is a retrospective replay, not a blind
out-of-sample evaluation, because the public target labels have influenced
earlier research.

The isolated CI workflows now include the 8-series smoke, 256-series pilot,
full historical fold, and a separate Phase 2 forecast-freeze/scoring sequence.
See the verified run links and exact metrics above. No unreviewed code ran on
self-hosted infrastructure. No model was merged or promoted.
