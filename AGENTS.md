## WavLM-Large audio-only call extractor (v1)

### Agent operating procedure (required)
This repo expects extremely high traceability and conservatism. For any code/config/docs change:
- **History-first (no edits yet):** identify the exact files/symbols you will touch; review recent related commits (`git log -n 20 -- <paths>`) and diffs; summarize any recent failures in the same area.
- **Plan gate (no edits yet):** write a step-by-step plan with goal + success criteria (explicit metrics/gates), risks/failure modes (especially merge/oversplit risk), exact files to change, and exact checks/tests you will run.
- **Research gate (no edits yet):** verify uncertain/unstable assumptions via primary sources (Context7 and/or official docs + web when needed). If verification is inconclusive, choose the conservative path: **drop ambiguous** / output nothing rather than risk merges/oversplits.
- **Implement one atomic change:** make only the smallest cohesive change that can be validated end-to-end.
- **Verify locally (before commit):** run the narrowest relevant checks first (usually a targeted `pytest` subset) and record them in the commit message.
- **Commit + push immediately:** every atomic change = 1 commit pushed to `main`. If push/auth is blocked, stop and resolve it before making further changes.
- **Retrospective:** record what worked/failed and the next hypothesis in the commit message (template below).

### Objective (non-negotiable)
Extract “real call” conversation segments from long MP3s **without ever merging calls** and **without over-splitting calls** for any **kept** outputs. If separation is uncertain, **drop ambiguous** (output nothing) rather than output wrong segments.

### Hard gates (eval + production policy)
- **merges == 0**: no predicted segment overlaps >1 ground-truth call.
- **oversplits == 0**: no ground-truth call overlaps >1 predicted segment.
- **drop ambiguous**: conservative decoder must prefer “no output” over incorrect segmentation.

### Canonical data locations (S3)
- Bucket: `rezora-whisperx-us-east-1-864981718771`
- Labels: `labeling/corrected_boundaries/v2/{video_id}.json`
- Audio (expected): `audio/pretraining/*.mp3` (keys end with ` - {video_id}.mp3`)
- Outputs base prefix: `call_extractor/wavlm_large_v1/`
  - Models: `call_extractor/wavlm_large_v1/models/{run_id}/`
  - Predictions: `call_extractor/wavlm_large_v1/segments/{video_id}.json`
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

### Local testing (CPU only; no model runs)
- Add/maintain unit tests for:
  - label alignment (including 0-gap adjacent calls)
  - decoder drop-on-ambiguity behavior
  - merge/oversplit metrics on toy segments
- Run targeted `pytest` locally before each commit+push (or explicitly document why not in `Checks:`).
