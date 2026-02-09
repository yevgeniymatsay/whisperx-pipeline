# CALL_EXTRACTOR_WAVLM_RUNS

Append one block per training cycle. Keep this file short and factual.

## Template

### {run_id}
- Date (UTC):
- Git SHA:
- Base model:
- Train config:
- Decode config:
- Eval split:
- Eval gates:
  - merges:
  - oversplits:
  - keep_rate:
  - false_positives (no-call vids):
- Artifacts (S3):
  - model:
  - segments:
  - clips:
- Notes:

## Runs

### run_20260207_043541_unknown (SMOKE)
- Date (UTC): 2026-02-07
- Git SHA: d2baf84 (EC2 ran from a tarball snapshot; no `.git` on instance)
- Base model: `microsoft/wavlm-large`
- Train config: `configs/call_extractor/train_smoke.config.json` (1 epoch; train video: `D4uiHjHW4AU`; very small sampling)
- Decode config: `configs/call_extractor/decode_wavlm_large_v1.config.json`
- Eval split: `configs/call_extractor/split_smoke.config.json` (eval videos: `CfMJ01KP_ns`, `15QzruVINDc`)
- Eval gates:
  - merges: 0
  - oversplits: 0
  - keep_rate: 0.0 (0/6 calls kept)
  - false_positives (no-call vids): 0 segments
- Artifacts (S3):
  - model: `call_extractor/wavlm_large_v1/models/run_20260207_043541_unknown/`
  - segments: `call_extractor/wavlm_large_v1/segments/{video_id}.json` (uploaded for smoke eval vids)
  - clips: none
- Notes:
  - End-to-end EC2 training + predict + eval works; decoder is conservatively dropping everything (expected for smoke).
  - Follow-up: enable git clone/push on EC2 (Secrets Manager `github-token`) so run IDs include git SHA and we can commit run logs from the instance.

### run_20260208_005759_fd0ddb6
- Date (UTC): 2026-02-08
- Git SHA: fd0ddb6
- Base model: `microsoft/wavlm-large`
- Train config: `configs/call_extractor/train_wavlm_large_v1.config.json`
- Decode config: `configs/call_extractor/decode_wavlm_large_v1.config.json`
- Eval split: `configs/call_extractor/split_v1.config.json` (10 videos, 79 calls)
- Eval gates:
  - merges: 0
  - oversplits: 0
  - keep_rate: 0.000 (0/79 calls kept)
  - false_positives (no-call vids): 0 segments
- Artifacts (S3):
  - model: `call_extractor/wavlm_large_v1/models/run_20260208_005759_fd0ddb6/`
  - segments: `call_extractor/wavlm_large_v1/segments/{video_id}.json` (all empty in this run)
  - clips: none
- Notes:
  - Training completed and uploaded; inference produced 0 segments on eval with current decoder.
  - Decode sweep (strict gates) picked start_thr=0.6, end_thr=0.6, in_call_mean_min=0.5, but keep_rate remained 0.
  - Re-decode after strict peak-picking change (Git SHA `8fcbb05`): start_thr=0.10, end_thr=0.095, in_call_mean_min=0.56 => keep_rate 0.025 (2/79), merges=0, oversplits=0, fp=0.
  - Infra run (same weights, new code path + run-scoped S3 prefix): `call_extractor/wavlm_large_v1/models/run_20260208_005759_fd0ddb6/infra_20260208_053855_54358d8/`
    - Frame metrics (eval): `in_call` has modest ranking signal but low dynamic range; `start`/`end` AP ~0.006/~0.005 (mostly noise).
    - Viterbi decode (defaults) violated gates badly (merges/oversplits/FPs); do not use Viterbi until boundary heads have real signal.
    - Best strict-gates peaks sweep on the infra probs: keep_rate 0.013 (1/79) with `start_thr=0.11`, `end_thr=0.05`, `in_call_mean_min=0.56` (uploaded as `best_decode_peaks.config.json` + `eval_report_strict_peaks.*` under the infra prefix).
  - Next: fix training signal (especially start/end boundary heads) before spending time on decoder tuning.

### run_20260208_062825_d7d842e (ABORTED; checkpoint eval)
- Date (UTC): 2026-02-08
- Git SHA: d7d842e (weights) / 7fd0080 (eval scripts)
- Base model: `microsoft/wavlm-large`
- Train config: `configs/call_extractor/train_wavlm_large_v2.config.json` (boundary-heavy)
- Decode config: best strict-gates peaks sweep on eval
  - start_thr=0.55, end_thr=0.55, in_call_mean_min=0.60
- Eval split: `configs/call_extractor/split_v1.config.json` (10 videos, 79 calls)
- Eval gates (strict: merges=0, oversplits=0, FP=0):
  - merges: 0
  - oversplits: 0
  - keep_rate: 0.089 (7/79 calls kept)
  - false_positives (including call videos): 0 segments
- Artifacts (S3):
  - model: none (training interrupted before `best_model/` export + upload)
  - probs/segments + reports: `call_extractor/wavlm_large_v1/models/run_20260208_062825_d7d842e/ckpt209_eval_20260208_0715/`
  - clips: none
- Notes:
  - Trainer wrote checkpoints up to `checkpoint-418` (epoch ~2.0) but the run ended early.
  - Exported `checkpoint-209` → `export_checkpoint-209/` and evaluated that snapshot.
  - Boundary heads show real signal (start/end AP ~0.06/0.12 at tolerance 0.4s on eval), enabling non-zero keep under strict gates.
  - `checkpoint-418` was more "peaky" and did not improve strict-gates keep; viterbi sweep found no config satisfying strict gates on eval.

### run_20260208_081007_8ab6a30
- Date (UTC): 2026-02-08
- Git SHA: 8ab6a30 (weights) / d7de478 (eval scripts)
- Base model: `microsoft/wavlm-large`
- Train config: `configs/call_extractor/train_wavlm_large_v3.config.json` (triangle start/end targets; boundary_k=10; tol=0.4s)
- Decode config: best strict-gates peaks sweep on eval (v2)
  - start_thr=0.40, end_thr=0.65, in_call_mean_min=0.64
  - internal_peak_drop_threshold=0.90, nms_min_sep_s=0.35
- Eval split: `configs/call_extractor/split_v1.config.json` (10 videos, 79 calls)
- Eval gates (strict: merges=0, oversplits=0, FP=0):
  - merges: 0
  - oversplits: 0
  - keep_rate: 0.190 (15/79 calls kept)
  - false_positives (including call videos): 0 segments
- Artifacts (S3):
  - model: `call_extractor/wavlm_large_v1/models/run_20260208_081007_8ab6a30/`
  - eval (probs/segments/reports): `call_extractor/wavlm_large_v1/models/run_20260208_081007_8ab6a30/eval_20260208_092231/`
  - clips: none
- Notes:
  - Frame metrics (eval; tolerance=0.4s): start AP≈0.069, end AP≈0.119; in_call AP≈0.847 (pos_frac≈0.75).
  - Peaks decoding is currently the only mode used for strict-gates selection; viterbi remains too risky until in_call separation improves.
  - Re-sweep (peaks) on eval probs found a better strict-gates config: keep_rate 15/79 at start_thr=0.40, end_thr=0.65, in_call_mean_min=0.64.

### run_20260208_114429_59a1f83 (ABORTED; checkpoint sweeps)
- Date (UTC): 2026-02-08
- Git SHA: 59a1f83
- Base model: `microsoft/wavlm-large`
- Train config: `configs/call_extractor/train_wavlm_large_v4.config.json` (boundary_k=12, triangle targets, tol=0.4s, pos_weight start/end=100)
- Decode config: strict-gates peaks sweep per checkpoint (see S3 prefixes)
- Eval split: `configs/call_extractor/split_v1.config.json` (10 videos, 79 calls)
- Eval gates (strict: merges=0, oversplits=0, FP=0):
  - checkpoint-283: keep_rate 0.025 (2/79)
  - checkpoint-566: keep_rate 0.101 (8/79)  <-- best in this run
  - checkpoint-849: keep_rate 0.076 (6/79)
- Artifacts (S3):
  - exported checkpoints:
    - `call_extractor/wavlm_large_v1/models/run_20260208_114429_59a1f83/export_checkpoint-283/`
    - `call_extractor/wavlm_large_v1/models/run_20260208_114429_59a1f83/export_checkpoint-566/`
    - `call_extractor/wavlm_large_v1/models/run_20260208_114429_59a1f83/export_checkpoint-849/`
  - eval prefixes:
    - `call_extractor/wavlm_large_v1/models/run_20260208_114429_59a1f83/ckpt283_eval_20260208_122702/`
    - `call_extractor/wavlm_large_v1/models/run_20260208_114429_59a1f83/ckpt566_eval_20260208_124154/`
    - `call_extractor/wavlm_large_v1/models/run_20260208_114429_59a1f83/ckpt849_eval_20260208_125044/`
- Notes:
  - Trainer eval_loss worsened sharply after epoch 3; run stopped early.
  - This run did **not** beat v3 (keep_rate 0.190). Next iteration should adjust training (in_call dynamic range) rather than longer training.

### run_20260209_010446_df80d5a
- Date (UTC): 2026-02-09
- Git SHA: df80d5a
- Base model: `microsoft/wavlm-large`
- Train config: `configs/call_extractor/train_wavlm_large_v5.config.json` (epochs=5; triangle targets; tol=0.4s; boundary_k=12; pos_weight start/end=50; freeze_feature_encoder=true)
- Decode config: `configs/call_extractor/decode_wavlm_large_v1.config.json`
  - peaks: start_thr=0.40, end_thr=0.65, in_call_mean_min=0.64
- Eval split: `configs/call_extractor/split_v2.config.json` (10 videos, 79 calls; labels `labeling/corrected_boundaries/v2/`)
- Eval gates (strict: merges=0, oversplits=0, FP=0):
  - default decode: keep_rate 0.038 (3/79), merges=0, oversplits=0, fp=0
  - best strict-gates peaks sweep (from eval probs): keep_rate 0.139 (11/79), merges=0, oversplits=0, fp=0
    - best cfg: start_thr=0.45, end_thr=0.50, in_call_mean_min=0.62
- Artifacts (S3):
  - model: `call_extractor/wavlm_large_v1/models/run_20260209_010446_df80d5a/`
  - eval probs+segments+reports: `call_extractor/wavlm_large_v1/models/run_20260209_010446_df80d5a/eval_20260209_021242/`
  - re-decoded (best peaks cfg) + report: `call_extractor/wavlm_large_v1/models/run_20260209_010446_df80d5a/eval_20260209_021242_decoded_peaks_best/`
  - clips: none
- Notes:
  - Frame metrics (eval; tolerance=0.4s): in_call AP≈0.873 (pos_frac≈0.75), start AP≈0.138, end AP≈0.143.
  - Keep-rate regressed vs v3 (0.190) but boundary heads show improved ranking signal vs earlier low-AP runs.
  - Matched segments still often have very large boundary errors (segments overlap the correct call but start/end are far off); decoder/training needs refinement before any large batch extraction.
  - Viterbi sweeps over full-length eval probs are too slow with the current Python-loop implementation; avoid sweeping Viterbi until optimized.

### run_20260209_052329_a83e45a
- Date (UTC): 2026-02-09
- Git SHA: a83e45a
- Base model: `microsoft/wavlm-large`
- Train config: `configs/call_extractor/train_wavlm_large_v9.config.json` (tol=0.6s; boundary_k=8; heavy out-call sampling)
- Decode config: sweep-only (no single default)
  - strict-gates transitions best: `best_decode_transitions.json` (keep_rate_iou_0_5=0.025; 2/79)
  - strict-gates peaks best (internal=0.6): `best_decode_peaks_internal0.6.json` (keep_rate_iou_0_5=0.038; 3/79)
- Eval split: `configs/call_extractor/split_v2.config.json` (10 videos, 79 calls; labels `labeling/corrected_boundaries/v2/`)
- Eval gates (strict: merges=0, oversplits=0, FP=0):
  - transitions: merges=0, oversplits=0, fp=0, keep_rate_iou_0_5=0.025 (2/79)
  - peaks (internal=0.6): merges=0, oversplits=0, fp=0, keep_rate_iou_0_5=0.038 (3/79)
  - peaks (internal=0.9): no strict-gates config found
  - in_call: no strict-gates config found (for the tested grid)
  - viterbi: sweep too slow with current implementation (avoid until optimized)
- Artifacts (S3):
  - model + probs + reports: `call_extractor/wavlm_large_v1/models/run_20260209_052329_a83e45a/`
- Notes:
  - Frame metrics (eval; tolerance=0.6s): in_call AP≈0.908 (pos_frac≈0.75), start AP≈0.103, end AP≈0.164.
  - Despite good in_call ranking signal, strict-gates keep-rate regressed; likely driven by poor boundary localization / decoder mismatch rather than separation alone.
