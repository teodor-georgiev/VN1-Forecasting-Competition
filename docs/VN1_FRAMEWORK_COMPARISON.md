# VN1 forecasting-framework comparison (research only)

Compare three actual library adapters against the **official VN1 metric**, not against
each library's own metric. The script does **not** modify the incumbent,
submission files, or any production pipeline.

## Status

- New comparison harness: implemented on an isolated feature branch.
- Contract tests cover official scoring, deterministic sampling, historical cutoff,
  and Phase 2 separate freeze/score with tamper checks. New checks need CI replay.
- sktime, Darts, MLForecast full library execution: **NOT YET VERIFIED**.
  Pull-request CI installs all three and executes an 8-series historical smoke.
  Current approved VPS access returned internal errors. Do not report model
  accuracy before a successful real-data run.
- Champion: earlier 0.477093 LightGBM/statistical ensemble is **not the
  latest**. The owner's reported ~0.467 result is the provisional reference;
  locate the exact frozen receipt before certifying a head-to-head comparison.
  Do not assume an older repository README describes the best result.

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

The pull-request workflow `.github/workflows/vn1-frameworks.yml`
uses a GitHub-hosted runner to run the contract tests and an eight-series
historical real-library smoke test. It is **not** the full-catalogue benchmark,
which must run only after the smoke tests pass. No unreviewed code is run on
self-hosted infrastructure. No model was merged or promoted.
