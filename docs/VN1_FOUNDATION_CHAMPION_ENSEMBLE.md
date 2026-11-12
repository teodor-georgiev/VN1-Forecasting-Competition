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

## First full-catalogue optimization, verified on desktop (2026-11-12)

The immutable original PR17 champion forecast archive was downloaded from
run `43060875239` and its Phase 2 SHA-256
`33cdb93ccc2415e9c6697e1074a0e4ecb2ad6066d473480b79a3582aa1122101`
was verified. Both Phase 1 and Phase 2 original ARIMA + AMSI forecasts were
exported as keyed, 15,053-series CSVs; no refitting was needed. The larger
Chronos-2 (120M) full-catalogue forecasts were saved independently by the
desktop GPU runner.

On 15,053 series and a 13-week horizon, the Phase 1 weight grid selected
**70% original ARIMA+AMSI champion + 30% pure Chronos-2**:

| Model | Phase 1 score | Frozen Phase 2 score |
| --- | ---: | ---: |
| Original ARIMA+AMSI champion | 0.5704981371 | **0.4693382739** |
| Selected 70/30 blend | **0.5039456418** | 0.4769498817 |

The Phase 2 blended forecast has lower **normalized absolute error** than
the original champion (`0.4476793845` versus `0.4677593010`), but it
has **more negative aggregate signed bias** (`-0.0292704972` versus
`-0.0015789729`). Including VN1's mandatory signed-bias penalty, the
Phase 2 score is therefore **worse by 0.0076116078**.

This is diagnostic evidence of a *regime-dependent* global bias risk.
Do not promote the Phase 1-selected 70/30 blend. The best independently
verified retrospective persistent-bias specialist remains `0.4691204554`.
A globally bias-neutralized ensemble may warrant a newly predeclared,
historical validation experiment, not post-hoc selection on Phase 2.

Desktop files (not committed): `champions/original-arima-amsi-phase1.csv`,
`champions/original-arima-amsi-phase2.csv`,
`ensemble_research/weights-original-plus-chronos2.json`,
`ensemble_research/frozen-original-plus-chronos2/{ensemble-phase2.csv,
champion-phase2.csv,frozen-manifest.json,score-phase2.json}`.
A local `saved_models.json` index and compressed ZIP backups
`archives/chronos2-phase1.zip` and `archives/chronos2-phase2.zip`
also verify and preserve both reusable pure forecasts.
