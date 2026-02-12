# Execution Checklist — gemini local sweep speed v1

## Header
- Date (UTC): 2026-02-12
- Baseline Git SHA: 5793ed2
- Branch: `codex/gemini-pilot-v1`
- Scope (files/symbols):
  - `scripts/run_gemini_local_dir_sweep_and_eval.py` (new)
  - `scripts/_run_eval_batch.py` (cache safety; no destructive cleanup)
  - `gemini_pilot/runner.py`:
    - optional parallel per-audio model requests (`--model-concurrency`)
    - store full response metadata for later auditing (no prompt/schema changes)
  - `gemini_pilot/eval/` (local run outputs; gitignored)

## Frozen comparability key (must match for comparisons)
- metrics_version: `n/a (Gemini pilot eval; not WavLM gate metrics)`
- gate_policy_version: `n/a`
- split_config_path: `n/a`
- label_prefix: `s3://rezora-whisperx-us-east-1-864981718771/labeling/corrected_boundaries/v1/`
- match_tol_s: `n/a`
- overlap_eps_s: `n/a`
- min_coverage: `n/a`

## Frozen definitions (short)
- Audio source for this cycle: local directory `/Volumes/Yevgeniy's Drive/tmp_gemini_audio` (exactly 10 files).
- Models: 5-model sweep as implemented in `gemini_pilot/config.py` (no changes in this cycle).
- Prompt + schema: frozen; **no changes** in this cycle.
- Pilot concurrency: per-audio model requests may run in parallel (default remains sequential; batch runner enables concurrency=5).

## Plan-change log (required if anything above changes)
- 2026-02-12: add per-audio parallel model calls in the pilot runner
  - What changed: `gemini_pilot/runner.py` runs `generate_content` across models using a bounded thread pool when enabled.
  - Why: speed up 5-model comparisons while keeping the same prompt+schema contract.
  - Non-comparable note: `n/a` (implementation/throughput change only; does not change prompt/schema)
  - Version bumps: `n/a`
- 2026-02-12: remove GT-based eval script/parser from the pilot sweep flow
  - What changed: removed `scripts/eval_gemini_call_detector.py` and associated unit tests; local-dir runner no longer runs eval.
  - Why: for this cycle we only want raw model outputs saved for manual inspection.
  - Non-comparable note: `n/a`
  - Version bumps: `n/a`

## Checklist
- [ ] History-first (`git log -n 20 -- <paths>` + summarize relevant failures)
- [ ] Plan locked (goal + success criteria + risks + files + checks)
- [ ] Research gate complete (if needed; otherwise `n/a`)
- [ ] Implement (one atomic change)
- [ ] Local checks run (unit tests / static checks only)
- [ ] Commit + push (working tree clean)
- [ ] Run Review recorded (required before next cycle)

## Cycle discipline (required)
This is an implementation-only cycle; no production decode/training sweep is run here.

## Run Review (required after metrics-producing cycle)
- `n/a` for this implementation-only cycle (no new WavLM metrics-producing cycle).
