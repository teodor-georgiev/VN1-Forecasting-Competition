# VN1 forecasting-framework comparison (research only)

Compare three actual library adapters against the **official VN1 metric**, not against
each library's own metric. The script does **not** modify the incumbent,
submission files, or any production pipeline.

## Status

- New comparison harness: implemented on an isolated feature branch.
- Scoring, deterministic sampling, and cutoff synthetic unit tests: 3 passed
  in an isolated local environment with NumPy/pandas/pytest.
- sktime, Darts, MLForecast full library execution: **NOT YET RUN**.
  Current approved VPS access returned internal errors; this isolated
  development environment lacked the three packages and could not download
  them. Do not report benchmark scores before a real run.
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

To score the fixed 13-week Phase 2 forecasts on published truth
(`Phase 2 - Sales.csv` required), run separately and label results retrospective:

```bash
python vn1_framework_benchmark.py --phase2 --sample-size 0 --output-dir /tmp/vn1_fw_phase2
```

Results are stored as `scores.json` and a column-aligned CSV per model.
Verify the expected identifier order, dates, finite predictions, and
full-catalogue coverage before using results. Do not compare a partial-sample
score against a full-catalogue champion score.

No workflow has been dispatched, no code merged, no model promoted.
