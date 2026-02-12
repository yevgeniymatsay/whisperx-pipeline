# Execution Checklist — gemini pilot v1

## Header
- Date (UTC): 2026-02-12
- Baseline Git SHA: 6e74f78
- Scope (files/symbols):
  - `gemini_pilot/__init__.py`
  - `gemini_pilot/config.py`
  - `gemini_pilot/prompts.py`
  - `gemini_pilot/runner.py`
  - `gemini_pilot/README.md`
  - `scripts/gemini_pilot_extract_call_timestamps.py`
  - `tests/whisperx_pipeline/test_gemini_pilot_runner.py`
  - `.env.example`
  - `requirements.txt`

## Frozen comparability key (must match for comparisons)
- metrics_version: `n/a (pilot timestamp probe)`
- gate_policy_version: `n/a (pilot timestamp probe)`
- split_config_path: `n/a`
- label_prefix: `n/a`
- match_tol_s: `n/a`
- overlap_eps_s: `n/a`
- min_coverage: `n/a`

## Frozen definitions (short)
- strict-valid gate: `n/a`
- FP semantics (must stay spurious-only): `n/a (no GT eval in this cycle)`

## Plan-change log (required if anything above changes)
- 2026-02-12: removed parser module and default parsing behavior
  - What changed: dropped `gemini_pilot/timestamps.py` + parser tests; runner now captures raw output only.
  - Why: v1 goal is to evaluate Gemini raw output quality before introducing parsing logic.
  - Non-comparable note: `n/a`
  - Version bumps: `n/a`
- 2026-02-12: added Rich console UX for pilot runs
  - What changed: runner now prints upload/generate/delete progress + raw model output via `rich` panels.
  - Why: make local pilot runs auditable/readable without changing the raw-output-first contract.
  - Non-comparable note: `n/a`
  - Version bumps: `n/a`

## Checklist
- [x] History-first (`git log -n 20 -- <paths>` + summarize relevant failures)
- [x] Plan locked (goal + success criteria + risks + files + checks)
- [x] Research gate complete (Gemini API docs + python-genai SDK)
- [x] Implement (one atomic change)
- [x] Local checks run (unit tests / static checks only)
- [x] Commit + push (working tree clean)
- [x] Run Review recorded (required before next cycle)

## Cycle discipline (required)
This cycle is exploratory and does not produce GT comparability metrics.

## Run Review (required after metrics-producing cycle)
- `n/a` for this implementation-only cycle (no training/decode/eval metrics produced).

## Manual Smoke (Local)
- 2026-02-12: ran pilot on `/private/tmp/DEt3IRqqUVs.mp3` (local file)
  - Run artifacts: `artifacts/gemini_pilot/run_20260212_054610/`
  - Models: `gemini-3-flash-preview`, `gemini-2.0-flash`
  - Cloud cleanup: uploaded Gemini Files API objects were deleted after each model run (HTTP 200).

## Checks Run
- `python -m py_compile /Users/yevgeniymatsay/whisperx-pipeline/gemini_pilot/*.py /Users/yevgeniymatsay/whisperx-pipeline/scripts/gemini_pilot_extract_call_timestamps.py`
- `pytest /Users/yevgeniymatsay/whisperx-pipeline/tests/whisperx_pipeline/test_gemini_pilot_runner.py -q`
- `python /Users/yevgeniymatsay/whisperx-pipeline/scripts/gemini_pilot_extract_call_timestamps.py --help`
- `./.venv/bin/pip install -r requirements.txt`
- `./.venv/bin/python -m py_compile /Users/yevgeniymatsay/whisperx-pipeline/gemini_pilot/*.py /Users/yevgeniymatsay/whisperx-pipeline/scripts/gemini_pilot_extract_call_timestamps.py`
- `./.venv/bin/pytest /Users/yevgeniymatsay/whisperx-pipeline/tests/whisperx_pipeline/test_gemini_pilot_runner.py -q`
- `./.venv/bin/python /Users/yevgeniymatsay/whisperx-pipeline/scripts/gemini_pilot_extract_call_timestamps.py --help`

## Frozen policy for this cycle
- Models fixed to two-model probe by default:
  - `gemini-3-flash-preview`
  - `gemini-2.0-flash`
- Raw output capture only (no schema-enforced JSON prompt contract, no parsing).
- No ffmpeg clipping in v1.
