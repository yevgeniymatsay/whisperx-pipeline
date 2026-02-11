# Execution Checklist — prod decode v1 (no training)

## Header
- Date (UTC): 2026-02-11
- Baseline Git SHA: b5d7808
- Scope (files/symbols):
  - `pipeline/call_extractor_wavlm/decode_prod.py` (`decode_production_segments`, `ProdDecodeConfig`)
  - `configs/call_extractor/decode_prod_v1.config.json`
  - `scripts/eval_call_extractor_production_report.py`
  - (planned next) `scripts/decode_call_extractor_from_probs_prod.py`
  - (planned next) `scripts/sweep_call_extractor_prod_decode.py`
  - (planned next) `scripts/run_call_extractor_prod.py`

## Frozen comparability key (must match for comparisons)
- metrics_version: `gate_metrics_v1`
- gate_policy_version: `strict_gates_v1_fp_total`
- split_config_path: `configs/call_extractor/split_v2.config.json`
- label_prefix: `labeling/corrected_boundaries/v2/`
- match_tol_s: `0.25`
- overlap_eps_s: `0.10`
- min_coverage: `0.30`

## Frozen definitions (short)
- strict-valid gate: **n/a (production selection is not strict-valid)**
- FP semantics (must stay spurious-only): **FP_total = predicted segments overlapping zero GT calls under tol+eps**

## Plan-change log (required if anything above changes)
- Date:
- What changed:
- Why:
- Non-comparable note:
- Version bumps:

## Checklist
- [x] History-first (`git log -n 20 -- <paths>` + summarize relevant failures)
- [x] Plan locked (goal + success criteria + risks + files + checks)
- [ ] Research gate complete (if needed; otherwise `n/a`)
- [ ] Implement (one atomic change)
- [ ] Local checks run (unit tests / static checks only)
- [ ] Commit + push (working tree clean)
- [ ] Run Review recorded (required before next cycle)

## Cycle discipline (required)
**A “cycle” = one bounded unit of work that produces new metrics** (decode sweep or eval run).

For every cycle:
- [ ] Freeze comparability key in this checklist (above).
- [ ] Run a capped sweep (<=500 configs unless explicitly approved larger).
- [ ] Materialize outputs to a decode-scoped prefix (never overwrite root `segments/`).
- [ ] Run eval with frozen params; save `eval_report.json` + `eval_report.md`.
- [ ] STOP and complete the Run Review (below). No new cycle may start until Run Review is recorded and user approves next action.

## Run Review (required after any metrics-producing cycle)
- Best prior comparable run:
- New run:
- merges:
- oversplits:
- FP_total:
- keep_rate_iou_0_5 / keep_rate_coverage (reporting-only):
- Production metrics:
  - fp_no_call_seconds_per_hour:
  - fp_no_call_seconds:
  - fp_no_call_segments:
  - purity_segment_p10 / median:
  - non_call_seconds_total:
  - segments_per_hour:
  - predicted_seconds_per_hour:

Decision:
- Proceed / Iterate / Rollback / Method change

Human approval:
- User reply must be explicit: “Proceed”, “Iterate with X”, “Rollback to Y”, or “Method change”.

