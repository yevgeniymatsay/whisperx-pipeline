# Execution Checklist — prod decode v1 (no training)

## Header
- Date (UTC): 2026-02-11
- Baseline Git SHA: b5d7808
- Scope (files/symbols):
  - `pipeline/call_extractor_wavlm/decode_prod.py` (`decode_production_segments`, `ProdDecodeConfig`)
  - `configs/call_extractor/decode_prod_v1.config.json`
  - `scripts/eval_call_extractor_production_report.py`
  - `scripts/decode_call_extractor_from_probs_prod.py`
  - `scripts/sweep_call_extractor_prod_decode.py`
  - `scripts/run_call_extractor_prod.py`

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
- [x] Research gate complete (`n/a` for this cycle; no external assumptions changed)
- [x] Implement (atomic: prod sweep + artifacts)
- [x] Local checks run (unit tests / static checks only)
- [x] Commit + push (working tree clean)
- [x] Run Review recorded (required before next cycle)

## Cycle discipline (required)
**A “cycle” = one bounded unit of work that produces new metrics** (decode sweep or eval run).

For every cycle:
- [ ] Freeze comparability key in this checklist (above).
- [ ] Run a capped sweep (<=500 configs unless explicitly approved larger).
- [ ] Materialize outputs to a decode-scoped prefix (never overwrite root `segments/`).
- [ ] Run eval with frozen params; save `eval_report.json` + `eval_report.md`.
- [ ] STOP and complete the Run Review (below). No new cycle may start until Run Review is recorded and user approves next action.

## Run Review (required after any metrics-producing cycle)
- Best prior comparable run: n/a (first prod-decode sweep under this checklist)
- New run: `reports/prod_decode_sweep_run_20260209_034309_8105f4c_20260211_005720.json`
  - Logs: `logs/run_20260209_034309_8105f4c/20260211_005720_sweep_prod_decode.log`
  - Note: train no-call probs missing for 8 vids; no-call FP budget is based only on the 2 eval no-call videos.
- merges: 17
- oversplits: 49
- FP_total: 37
- kept_calls_iou0.5 / total_gt_calls: 16 / 79
- keep_rate_iou_0_5: 0.203
- keep_rate_coverage: 0.215
- raw_keep (matched_calls / total_gt_calls): 0.215
- Production metrics:
  - fp_no_call_seconds_per_hour: 18.828
  - fp_no_call_seconds: 7.920
  - fp_no_call_segments: 2
  - purity_segment_p10 / median: 0.0 / 1.0
  - non_call_seconds_total: 890.597
  - segments_per_hour: 63.256
  - predicted_seconds_per_hour: 2404.789

Decision:
- Iterate (decoder sweep)
  - Hypothesis: merges remain high because viterbi stays ON across multiple calls and the valley splitter never triggers (split_lo too low / max_segment_s too large / no trim).
  - Next action (pending approval): run a focused <=500 sweep that (1) raises `split_lo` (e.g., 0.45/0.55), (2) reduces `max_segment_s` (e.g., 120/300/600), and (3) adds `trim_s` (0/1/2) and tighter filters, while keeping viterbi params fixed to the current best (smooth=0.5, min_on=0.5, min_off=0.0, cost_mult=0.5).

Human approval:
- User reply must be explicit: “Proceed”, “Iterate with X”, “Rollback to Y”, or “Method change”.
