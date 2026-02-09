# Execution Checklist Template

Use this template for any execution-locked plan: `docs/EXECUTION_CHECKLIST_{YYYYMMDD}_{shortname}.md`.

## Header
- Date (UTC):
- Baseline Git SHA:
- Scope (files/symbols):

## Frozen comparability key (must match for comparisons)
- metrics_version:
- gate_policy_version:
- split_config_path:
- label_prefix:
- match_tol_s:
- overlap_eps_s:
- min_coverage:

## Frozen definitions (short)
- strict-valid gate:
- FP semantics (must stay spurious-only):

## Plan-change log (required if anything above changes)
- Date:
- What changed:
- Why:
- Non-comparable note:
- Version bumps:

## Checklist
- [ ] History-first (`git log -n 20 -- <paths>` + summarize relevant failures)
- [ ] Plan locked (goal + success criteria + risks + files + checks)
- [ ] Research gate complete (if needed; otherwise `n/a`)
- [ ] Implement (one atomic change)
- [ ] Local checks run (unit tests / static checks only)
- [ ] Commit + push (working tree clean)
- [ ] Run Review recorded (required before next cycle)

## Cycle discipline (required)
**A “cycle” = one bounded unit of work that produces new metrics** (training run, decode sweep, or eval run).

For every cycle:
- [ ] Freeze comparability key in this checklist (below).
- [ ] Run the cycle (training OR capped sweep). **Config cap rule:** diagnostic sweep <=500 configs unless user explicitly approves larger.
- [ ] Materialize outputs to a decode-scoped prefix (never overwrite root `segments/`). Example: `call_extractor/wavlm_large_v1/models/{run_id}/decoded_<tag>_<ts>`.
- [ ] Run eval on the decode-scoped prefix with frozen params; save `eval_report.json` + `eval_report.md` (optional: `run_review.json`).
- [ ] STOP and complete the Run Review (below). No new cycle may start until Run Review is recorded and the user approves the next action.

## Run Review (required after any metrics-producing cycle)
Compare new run vs **best prior run with the same comparability key**. If the key differs, record as **non-comparable**.

- Best prior comparable run:
- New run:
- strict_valid? (yes/no):
- kept_calls_iou0.5 / total_gt_calls (e.g., 6/79):
- keep_rate_iou_0.5:
- keep_rate_coverage:
- raw_keep (matched_calls / total_gt_calls):
- merges:
- oversplits:
- FP_total:
- boundary errors (mean_start_abs_err_s, mean_end_abs_err_s):

Required: boundary error distribution (do not guess)
- start_abs_err_s p50/p90/p99:
- end_abs_err_s p50/p90/p99:
- start_abs_err_s counts <=1s / <=3s / <=10s:
- end_abs_err_s counts <=1s / <=3s / <=10s:
- If `eval_report.json` does not contain what is needed to compute these, the only allowed next action is a reporting-only change to emit it (no guessing).

Top failure reason
- Top failure reason (merges vs oversplits vs FP_total vs coverage):
- Videos (ids) driving the top failure:

Per-video table (top 10, worst failures first)
Sort by: merges>0, oversplits>0, FP_total>0, lowest keep_rate_iou0.5, highest boundary error.

| video_id | gt_calls | pred_calls | matched_calls | kept_iou0.5 | keep_rate_iou0.5 | merges | oversplits | FP_total | min_gap_s | tiny_gaps<0.2s | zero_gaps | mean_start_abs_err_s | mean_end_abs_err_s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |

Failure clustering (one paragraph + counts)
- videos failing merges:
- videos failing oversplits:
- videos failing FP_total:
- videos failing coverage/IoU:
- Concentration: high-gt_calls videos? tiny/zero-gap videos?

Kept vs missed summary
- kept_calls_iou0.5 / total_gt_calls:
- matched_calls / total_gt_calls:
- kept_calls_coverage / total_gt_calls:
- One-line: which gate filters most (IoU vs coverage vs no match):

Decision (choose exactly one):
- Proceed / Iterate (1 hypothesis + what changed) / Rollback / Method change

Must end with:
- Single hypothesis:
- Single next action (decoder sweep <=500 configs OR one training tweak):

Notes:
