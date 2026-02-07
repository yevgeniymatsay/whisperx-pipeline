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
