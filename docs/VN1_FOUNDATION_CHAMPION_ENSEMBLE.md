# Reusing saved Chronos / TimesFM forecasts in the VN1 champion

This draft does **not** replace the verified ARIMA + AMSI / persistent-bias
champion. The current verified retrospective Phase 2 specialist score is
**0.4691204554**. Future foundation-model weights must be selected on a
historical split and then frozen **before** opening the subsequent labels.

## Desktop saved outputs

Full GPU inference is checkpointed under:
`C:\Users\Toni\VN1-Foundations-20261112\results\`

Each model and phase owns its independent directory, e.g.
`chronos2-phase1/`, `chronos2-phase2/`,
`timesfm2.5-phase1/`, `timesfm2.5-phase2/`. The runner writes 256-series
`.npy` chunks atomically and, on successful completion, produces:

- `forecast_pure.csv`: **reusable model-only forecasts** with `Client`,
  `Warehouse`, `Product` and exactly 13 ISO weekly forecast dates
- `forecast_blend50.csv`: **legacy** 50% model + 50% older LightGBM ensemble;
  **do not** use as the foundation input for a new champion blend
- `metrics.json`: model name/checkpoint, phase, source revision, GPU/device,
  full-series count, prediction hash, pure score and legacy blended score

The `forecast_pure.csv` files alone are sufficient for future ensembles:
no repeat GPU inference is needed. Do not upload private raw forecasts
indiscriminately. Keep durable local copies of each completed model directory.

## Historical weight selection (research only)

A new CLI supports **any 1–4 named, precomputed product-keyed forecast CSVs**
(e.g. Chronos-2, TimesFM 2.5, Chronos-2 Small). It requires an explicitly
supplied best-champion forecast CSV for each phase; it cannot silently substitute
the historical LightGBM incumbent as the current ARIMA + AMSI champion.

At the Phase 1 split, the blend searches a deterministic 5%-increment
non-negative simplex with **champion >=70%** and combined foundation <=30%.
Weights optimize the **official VN1 score** (normalized absolute error plus
absolute signed bias), not MSE. A Phase 1 gain of at least 0.001 is needed to
promote the combination; otherwise weights stay at 100% champion.

```powershell
python vn1_foundation_champion_ensemble.py optimize `
  --phase1-target "Phase 1 - Sales.csv" `
  --champion "best-champion-phase1.csv" `
  --foundation "chronos2=C:\Users\Toni\VN1-Foundations-20261112\results\chronos2-phase1\forecast_pure.csv" `
  --foundation "timesfm2.5=C:\Users\Toni\VN1-Foundations-20261112\results\timesfm2.5-phase1\forecast_pure.csv" `
  --output weights-phase1.json
```

Here `best-champion-phase1.csv` must have been generated from the relevant
champion checkpoint using Phase 0 only. A Phase 2 champion CSV must similarly
come from Phase 0+1 without using Phase 2 targets, and should be verified
against the existing frozen-champion manifest/checkpoint. Do not claim champion
parity from an ad-hoc reconstruction.

```powershell
python vn1_foundation_champion_ensemble.py freeze `
  --history "Phase 1 - Sales.csv" `
  --champion "best-champion-phase2.csv" `
  --foundation "chronos2=C:\Users\Toni\VN1-Foundations-20261112\results\chronos2-phase2\forecast_pure.csv" `
  --foundation "timesfm2.5=C:\Users\Toni\VN1-Foundations-20261112\results\timesfm2.5-phase2\forecast_pure.csv" `
  --weights weights-phase1.json --output-dir frozen-foundation
```

The freeze operation **does not read Phase 2 targets**, writes separate
champion and blend CSVs, and hashes all inputs, selected weights and frozen
predictions. It never overwrites an existing freeze.

Then, for a clearly labelled retrospective diagnostic only:

```powershell
python vn1_foundation_champion_ensemble.py score `
  --output-dir frozen-foundation --phase2-target "Phase 2 - Sales.csv"
```

It verifies frozen hashes before reading the later labels, and reports both
champion and candidate scores, signed bias and absolute-error contributions.
No automatic promotion, merge or deployment is performed.

## What was previously optimized?

- Legacy statistical + LightGBM: an explicit historical grid over
  `[0, .25, .5, .75, 1]`; only an earlier historical fold selected the
  resulting weight.
- ARIMA + AMSI: a finite set of capped mixing policies selected on
  chronological historical folds, not a global multi-foundation simplex search.
- Persistent group-bias specialist: predeclared bias-strength policies fitted
  on completed historical OOF residuals, selected and gated by later Phase 0.
- Chronos/TimesFM: previously fixed 50% blends against that **older** model.
  No valid prior jointly optimized ARIMA+AMSI+Chronos+TimesFM weight vector.

This new procedure tests conservative joint weights but **Phase 1 has already
been inspected in earlier research**. Treat any reported Phase 2 gain as
retrospective/non-blind, and use independent future chronological origins
before promoting a candidate as a robust improvement.
