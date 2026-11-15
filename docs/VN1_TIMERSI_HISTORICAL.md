# VN1 × upstream TimeRSI (Phase-0-only research)

This branch adds a **genuine upstream TimeRSI FEV adapter**. It does not implement
or imitate the TimeRSI code agent, does not modify the VN1 champion, and does
not access Phase 1/2 labels during conversion or TimeRSI execution.

## Required environment (not supplied by this repository)

- Approved Linux GPU research host; Python 3.12, CUDA, Docker.
- Claude Code 2.1.245 and an authorized Fable 5 / Vertex AI project.
- The TimeRSI repository and a licensed TimesFM-3 checkpoint.
- FEV and TimeRSI dependencies installed as documented by upstream.
- Auth is set up by the host operator; **never** commit credentials or .env.

References:
- https://github.com/haoxin1998/TimeRSI
- https://github.com/haoxin1998/TimeRSI/blob/main/docs/USAGE.md
- https://github.com/autogluon/fev

## Dataset and policy

The source is ONLY `Phase 0 - Sales.csv`. The CSV is converted to an
`id, timestamp, target` long-table Parquet file understood by FEV.
FEV series IDs are deterministic zero-padded row positions; their order matches
the original CSV. Client, Warehouse and Product are separately preserved as
static categorical features (`client`, `warehouse`, `product`) so TimeRSI can
use the competition's original hierarchy without Phase-1 or Phase-2 data.
The exported Parquet contains identifiers from the source file: keep research
outputs in an access-controlled location and do not log individual keys.

The FEV task has 13-week horizons and **six Phase-0 historical origins
92/105/118/131/144/157**, with one additional 13-week test window at each
origin. All targets for these folds end by Phase-0 week 170. The initial
training context is >=65 historical weeks, and the seasonal period is 52.

This adapter intentionally does **not** create a Phase-2 FEV task: TimeRSI
requires ground-truth windows, so using released Phase-2 actuals in research
would risk contaminating candidate selection. The first gate is to benchmark
historically, with no Phase-2 labels.

## Commands on an approved research host

```bash
python -m pytest -q test_vn1_timersi_fev.py
python vn1_timersi_fev.py export \
  --data-dir . \
  --parquet /safe/research/vn1-phase0-timersi.parquet

# On the approved host only, after TimeRSI is installed and authenticated:
python vn1_timersi_fev.py run \
  --parquet /safe/research/vn1-phase0-timersi.parquet \
  --checkpoint /safe/models/timesfm-3.0-pytorch \
  --out /safe/research/timersi-phase0-run
```

The exported manifest records Phase-0 input and dataset SHA-256 hashes.
The runner refuses changed inputs or existing output data. For completed
runs, `vn1_official_historical_scores.json` records per-window and pooled
**official VN1** values, separately from TimeRSI's native FEV optimization
metric. A result is only valid once the upstream runtime completes and
scores are verified; **no empirical TimeRSI score is available yet**.

## Benchmark decision

Compare the **same historical origins**, not the public Phase-2 score directly.
The existing champion-residual research on PR #20 reports these baseline values:

- ARIMA+AMSI champion pooled origins 118/131/144: `0.5345784755`.
- Champion origin-157 holdout: `0.4811455955`.
- Separate retrospective Phase-2 score: `0.4693382739`.

A new model cannot be promoted based on Phase-2 retrospective improvement
alone. It must beat the champion on matched historical windows without
leakage. TimeRSI research may use its own earlier probe windows: this is
research, **not** a blind benchmark or a proven improvement.

This is a draft, additive research integration. No deployment, production
writes, or automated model promotion are in scope.
