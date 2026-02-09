## WavLM-Large audio-only call extractor (v1)

### Agent operating procedure (required)
This repo expects extremely high traceability and conservatism. For any code/config/docs change:
- **History-first (no edits yet):** identify the exact files/symbols you will touch; review recent related commits (`git log -n 20 -- <paths>`) and diffs; summarize any recent failures in the same area.
- **Plan gate (no edits yet):** write a step-by-step plan with goal + success criteria (explicit metrics/gates), risks/failure modes (especially merge/oversplit risk), exact files to change, and exact checks/tests you will run.
- **Execution checklist (no edits yet):** for any execution-locked plan, create `docs/EXECUTION_CHECKLIST_{YYYYMMDD}_{shortname}.md` with baseline SHA, scope (files/symbols), frozen metric/gate definitions + versions, and an explicit Run Review section. Update the checklist as you work.
- **Research gate (no edits yet):** verify uncertain/unstable assumptions via primary sources (Context7 and/or official docs + web when needed). If verification is inconclusive, choose the conservative path: **drop ambiguous** / output nothing rather than risk merges/oversplits.
- **Implement one atomic change:** make only the smallest cohesive change that can be validated end-to-end.
- **Verify locally (before commit):** run the narrowest relevant checks first (usually a targeted `pytest` subset) and record them in the commit message.
- **Commit + push immediately:** every atomic change = 1 commit pushed to `main`. If push/auth is blocked, stop and resolve it before making further changes.
- **Retrospective:** record what worked/failed and the next hypothesis in the commit message (template below).

### Objective (non-negotiable)
Extract “real call” conversation segments from long MP3s **without ever merging calls** and **without over-splitting calls** for any **kept** outputs. If separation is uncertain, **drop ambiguous** (output nothing) rather than output wrong segments.

### Eval/sweep gates (GT-based)
- **merges == 0**: no predicted segment overlaps >1 ground-truth call.
- **oversplits == 0**: no ground-truth call overlaps >1 predicted segment.
- **FP_total == 0**: no predicted segment overlaps zero ground-truth calls (spurious segments anywhere). FP is overlap-based and must **never** be redefined as containment/overhang.
- **Kept calls (for keep_rate metrics):** IoU_exact >= 0.5 and coverage_exact >= min_coverage (min_coverage is frozen in the comparability key).
- **strict_valid:** merges==0 && oversplits==0 && FP_total==0 under the frozen matching params (match_tol_s, overlap_eps_s).

### Production policy (no GT available)
- **drop ambiguous**: conservative decoder must prefer “no output” over incorrect segmentation.

### Metric/gate definition freeze (required)
Once an execution-locked plan starts, **freeze** all metric definitions and gates. Any change to:
- metric formulas (including FP semantics),
- gate conditions,
- dataset split / label source,
- tolerance/epsilon/min_coverage values,
is a **plan change** and requires:
1. a short justification,
2. an explicit note that before/after results are **not comparable**,
3. a version bump (e.g., `metrics_version`, `gate_policy_version`) and a fresh run.

Additional required rules:
- **Separation of concerns:** FP is only “spurious segment” (overlaps no GT call). Boundary quality belongs in coverage/IoU/error metrics. If you want “outside-GT overhang” safety, implement it as a **new metric** (report-only first), and only later promote it to a gate via a plan change.
- **Single source of truth:** all sweeps/evals must use `pipeline.call_extractor_wavlm.metrics.compute_gate_metrics(...)` (no duplicated formulas).
- **No mid-stream changes:** do not change any scoring/selection code/config while a sweep/eval is running; finish the run, then start a new versioned run.
- **Do not kill running sweeps:** do not terminate/interrupt a running sweep/eval for impatience (or because CPU time is increasing). Let it finish to preserve near-miss diagnostics. Only stop if (a) the user explicitly tells you to stop, or (b) you can prove it is misconfigured (wrong split/prefix/params) and you record why in the execution checklist before stopping.

### Run Review (required after each metrics-producing cycle)
After any training/inference/eval cycle that produces new metrics, **STOP** and record a Run Review **in the execution checklist** (required; optionally mirror into `docs/CALL_EXTRACTOR_WAVLM_RUNS.md`) before taking further action.

Run Review compares the new run vs the **best prior run with the same comparability key**:
- strict_valid? (yes/no)
- keep_rate_iou_0.5, keep_rate_coverage
- merges / oversplits / FP_total
- boundary error summary (mean_start_abs_err_s, mean_end_abs_err_s)

Comparability key (must match exactly):
- metrics_version
- gate_policy_version
- split_config_path
- label_prefix
- match_tol_s, overlap_eps_s, min_coverage

If the comparability key differs, record the run as **non-comparable** and do not treat it as an improvement/regression.

Decision (choose exactly one):
- **Proceed** (new run is better / meets stage target)
- **Iterate (small tweak)**: state exactly one hypothesis and what changed
- **Rollback**: previous run is better; reuse best prior config
- **Method change**: only if two consecutive Iterate cycles fail and strict-almost is “not close”

Strict-almost (reporting-only): merges==0 && FP_total==0 && oversplits<=1 (kept calls still use IoU_exact/coverage_exact under the frozen comparability key).

No new training cycle starts until the Run Review is recorded (checklist required; run ledger optional).

### Eval/sweep invariants (required)
- All sweeps/evals must explicitly pass `--split-config <path>` and record it in the execution checklist (scripts may warn if omitted; do not make the flag mandatory in argparse).
- Every eval report artifact must include: `metrics_version`, `gate_policy_version`, `split_config_path`, `label_prefix`, `match_tol_s`, `overlap_eps_s`, `min_coverage`.

### Canonical data locations (S3)
- Bucket: `rezora-whisperx-us-east-1-864981718771`
- Labels: `labeling/corrected_boundaries/v2/{video_id}.json`
- Audio (expected): `audio/pretraining/*.mp3` (keys end with ` - {video_id}.mp3`)
- Outputs base prefix: `call_extractor/wavlm_large_v1/`
  - Models: `call_extractor/wavlm_large_v1/models/{run_id}/`
  - Segments (run-scoped): `call_extractor/wavlm_large_v1/models/{run_id}/segments/{video_id}.json`
  - Segments (decode-scoped): `call_extractor/wavlm_large_v1/models/{run_id}/decoded_<tag>_<ts>/segments/{video_id}.json`
  - Clips (FLAC 16k mono): `call_extractor/wavlm_large_v1/clips/...`
- **Audio alignment rule:** do not slice training/inference chunks directly from MP3; decode full MP3 → 16k mono FLAC once per video and do sample-accurate slicing from the cached FLAC (`.cache/call_extractor_wavlm/audio_flac/`).

### EC2 is the only runtime for model work
- Host: `ubuntu@ec2-13-217-101-70.compute-1.amazonaws.com`
- Key: `~/.ssh/whisperx-key-east1.pem`
- Instance: `whisperx-worker-1` / `i-08d8c53c1943eac02`
- **Do not run training or inference locally.**

### HF / Transformers requirements (follow `WavLM.md`)
- Backbone: `transformers.WavLMModel`
- Feature extraction: `transformers.AutoFeatureExtractor` (or `Wav2Vec2FeatureExtractor`), operating on raw float waveform arrays.
- Do **not** use tokenizers for this task.
- Training harness (v1): `transformers.Trainer` + `TrainingArguments` (Accelerate installed for distributed setup if needed).

### “Never assume” rule
If a change (training, inference, decoding, metrics, data I/O, etc.) does not improve gates or contradicts prior results, stop and research before making the next change:
- Review git history for the relevant code (`git show`, `git log`, `git blame`) and summarize what prior commits attempted.
- Verify the key assumption(s) via primary sources (Context7 + official docs + web search when needed).
- Write an updated plan/hypothesis, then implement the next **atomic** change.
Do not keep optimizing an approach that is structurally unable to satisfy the gates.

### Versioning discipline (main branch only)
- Work only on `main`.
- **No uncommitted work:** keep the working tree clean between atomic changes.
- **Atomic commits only:** each commit must be a complete, reviewable unit (code + tests/docs needed for that unit).
- **Push every commit immediately** to `main` for full traceability.
- **Do not push knowingly broken commits:** run the relevant local checks first (or document why they are `n/a` in `Checks:`).
- Before any EC2 run: commit+push code/config to `main`.
- After each EC2 run: append results to `docs/CALL_EXTRACTOR_WAVLM_RUNS.md`, commit+push to `main`.
- **Never commit model weights**; upload weights/artifacts to S3 only.
- Run ID format: `run_{YYYYMMDD_HHMMSS}_{gitsha}`

### Batch extraction safety defaults
- `scripts/predict_call_extractor_wavlm.py` uploads **segments JSON** by default; enable probs only for eval/sweeps with `--write-probs --upload-probs` (do **not** upload probs for thousands of videos).
- `scripts/extract_call_clips.py` deletes local clip files after upload by default; use `--keep-local-clips` only for debugging.

### Commit message template (required)
Use:
- `<type>(scope): <summary>`
- `Problem: ...`
- `Hypothesis: ...`
- `Approach: ...`
- `Checks: ...` (exact commands run, or `n/a`)
- `Outcome: ...` (pass/fail + key numbers if applicable)
- `Next: ...` (the next smallest experiment/change if outcome is negative)

Types: `fix`, `refactor`, `test`, `docs`, `chore`.

### EC2 git push auth (required)
Preferred: GitHub PAT stored in AWS Secrets Manager (region `us-east-1`) as secret name `github-token`.
- Use `gh auth login --with-token` (token pulled from Secrets Manager) and `gh auth setup-git`.
- Never print tokens in logs.

### Local testing (safe: unit tests + static checks only)
- Add/maintain unit tests for:
  - label alignment (including 0-gap adjacent calls)
  - decoder drop-on-ambiguity behavior
  - merge/oversplit metrics on toy segments
- Run targeted `pytest` locally before each commit+push (or explicitly document why not in `Checks:`).
- All sweeps/decoding/evals on real data must run on EC2.
