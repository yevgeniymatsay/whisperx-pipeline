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

## Run Review (required after any metrics-producing cycle)
Compare new run vs **best prior run with the same comparability key**. If the key differs, record as **non-comparable**.

- Best prior comparable run:
- New run:
- strict_valid? (yes/no):
- keep_rate_iou_0.5:
- keep_rate_coverage:
- merges:
- oversplits:
- FP_total:
- boundary errors (mean_start_abs_err_s, mean_end_abs_err_s):

Decision (choose exactly one):
- Proceed / Iterate (1 hypothesis + what changed) / Rollback / Method change

Notes:

