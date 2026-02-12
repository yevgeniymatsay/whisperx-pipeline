# Execution Checklist — gemini pilot v1

## Header
- Date (UTC): 2026-02-12
- Baseline Git SHA: 6e74f78
- Scope (files/symbols):
  - `gemini_pilot/__init__.py`
  - `gemini_pilot/config.py`
  - `gemini_pilot/prompts.py`
  - `gemini_pilot/timestamps.py`
  - `gemini_pilot/runner.py`
  - `gemini_pilot/README.md`
  - `scripts/gemini_pilot_extract_call_timestamps.py`
  - `tests/whisperx_pipeline/test_gemini_pilot_timestamps.py`
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
- Date:
- What changed:
- Why:
- Non-comparable note:
- Version bumps:

## Checklist
- [x] History-first (`git log -n 20 -- <paths>` + summarize relevant failures)
- [x] Plan locked (goal + success criteria + risks + files + checks)
- [x] Research gate complete (Gemini API docs + python-genai SDK)
- [x] Implement (one atomic change)
- [x] Local checks run (unit tests / static checks only)
- [ ] Commit + push (working tree clean)
- [x] Run Review recorded (required before next cycle)

## Cycle discipline (required)
This cycle is exploratory and does not produce GT comparability metrics.

## Run Review (required after metrics-producing cycle)
- `n/a` for this implementation-only cycle (no training/decode/eval metrics produced).

## Checks Run
- `python -m py_compile /Users/yevgeniymatsay/whisperx-pipeline/gemini_pilot/*.py /Users/yevgeniymatsay/whisperx-pipeline/scripts/gemini_pilot_extract_call_timestamps.py`
- `pytest /Users/yevgeniymatsay/whisperx-pipeline/tests/whisperx_pipeline/test_gemini_pilot_timestamps.py -q`
- `python /Users/yevgeniymatsay/whisperx-pipeline/scripts/gemini_pilot_extract_call_timestamps.py --help`
- `./.venv/bin/pip install -r requirements.txt`
- `./.venv/bin/python -m py_compile /Users/yevgeniymatsay/whisperx-pipeline/gemini_pilot/*.py /Users/yevgeniymatsay/whisperx-pipeline/scripts/gemini_pilot_extract_call_timestamps.py`
- `./.venv/bin/pytest /Users/yevgeniymatsay/whisperx-pipeline/tests/whisperx_pipeline/test_gemini_pilot_timestamps.py -q`
- `./.venv/bin/python /Users/yevgeniymatsay/whisperx-pipeline/scripts/gemini_pilot_extract_call_timestamps.py --help`

## Frozen policy for this cycle
- Models fixed to two-model probe by default:
  - `gemini-3-flash-preview`
  - `gemini-2.0-flash`
- Timestamp-only extraction output (no schema-enforced JSON prompt contract).
- No ffmpeg clipping in v1.
