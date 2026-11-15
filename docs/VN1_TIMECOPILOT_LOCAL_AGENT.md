# VN1: genuine TimeCopilot local LLM agent (research pilot)

This is a **research-only agent pilot** of published TimeCopilot v0.0.41, using local
Ollama qwen2.5:3b on an ephemeral GitHub-hosted CPU runner. No API tokens,
cloud LLM credentials or production resources are required. Ollama v0.40.16
is pinned, and its release archive SHA-256 is checked before installation.

## Experiment integrity

The pilot chooses two deterministic Phase-0 series in different sales volume
bands using training-only values before historical origin 157. It never sends
the 13-week holdout to the agent. Instead it requires actual TimeCopilot
feature, cross-validation, forecast and anomaly tool calls, then freezes
and hashes the forecasts before opening Phase-0 holdout sales for VN1 scoring.

Results represent a **2-series pilot**, not an all-15,053-series evaluation.
No Phase-1/Phase-2 actual labels are used. This cannot justify model promotion
or demonstrate an improvement over the full-catalogue champion.

## Running

Unit tests: python -m pytest -q test_vn1_timecopilot_local_pilot.py

The GitHub Actions workflow VN1 TimeCopilot local agent pilot installs the
released Python package, starts a local tool-capable Ollama LLM, and runs
vn1_timecopilot_local_pilot.py. On success it uploads a JSON score receipt
and an immutable-by-convention frozen prediction file. Failed tool calls,
missing model forecasts or invalid date alignment cause nonzero exit.

## Limitations

Qwen2.5:3b is a small, nondeterministic LLM and may fail its tool-use or
structured-output protocol. The workflow will report such failure rather
than manufacture a score. Record the pulled Ollama model digest in job logs.
This is a TimeCopilot test, NOT a substitute for genuine TimeRSI.

TimeRSI still requires an independently approved Linux GPU+Docker runner
and Fable/Vertex authorization. This branch makes no production changes.
