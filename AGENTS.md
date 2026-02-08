## WavLM-Large audio-only call extractor (v1)

### Objective (non-negotiable)
Extract “real call” conversation segments from long MP3s **without ever merging calls** and **without over-splitting calls** for any **kept** outputs. If separation is uncertain, **drop ambiguous** (output nothing) rather than output wrong segments.

### Hard gates (eval + production policy)
- **merges == 0**: no predicted segment overlaps >1 ground-truth call.
- **oversplits == 0**: no ground-truth call overlaps >1 predicted segment.
- **drop ambiguous**: conservative decoder must prefer “no output” over incorrect segmentation.

### Canonical data locations (S3)
- Bucket: `rezora-whisperx-us-east-1-864981718771`
- Labels: `labeling/corrected_boundaries/v1/{video_id}.json`
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
If a training cycle does not improve gates, stop and research (Context7 + web search) before making the next change. Do not keep optimizing an approach that is structurally unable to satisfy the gates.

### Versioning discipline (main branch only)
- Work only on `main`.
- Before any EC2 run: commit+push code/config to `main`.
- After each EC2 run: append results to `docs/CALL_EXTRACTOR_WAVLM_RUNS.md`, commit+push to `main`.
- **Never commit model weights**; upload weights/artifacts to S3 only.
- Run ID format: `run_{YYYYMMDD_HHMMSS}_{gitsha}`

### EC2 git push auth (required)
Preferred: GitHub PAT stored in AWS Secrets Manager (region `us-east-1`) as secret name `github-token`.
- Use `gh auth login --with-token` (token pulled from Secrets Manager) and `gh auth setup-git`.
- Never print tokens in logs.

### Local testing (CPU only; no model runs)
- Add/maintain unit tests for:
  - label alignment (including 0-gap adjacent calls)
  - decoder drop-on-ambiguity behavior
  - merge/oversplit metrics on toy segments
- Run `pytest` locally before pushing changes when feasible.
