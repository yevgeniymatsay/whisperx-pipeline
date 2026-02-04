# Call Segmenter Snapshot (2026-02-04)

This is a context-preserving snapshot for the call segmenter work on branch:

- `codex/call-segmenter-v6-features-2026-02-04`

## Goal

Improve **call split correctness** (segment IoU-F1) while keeping:

- **No-call false positives:** 0 on the 10 no-call videos
- **Time micro-F1:** high (do not regress coverage)
- **Pred/Truth segment ratio:** near 1.0 (avoid "winning" by spamming or merging)

## What Changed (v6)

### Cheap acoustic features (train/inference parity)

New shared module:

- `pipeline/call_segmenter/audio_features.py`

Adds per-window features computed from decoded audio (16k mono float32):

- Energy:
  - `rms_energy`, `log_rms`, `log_rms_z` (per-video z-score of `log_rms`)
- Time-domain:
  - `zcr` (zero-crossing rate)
- Band-energy fractions:
  - `low_band_frac` (<300 Hz)
  - `phone_band_frac` (300-3400 Hz)
  - `mid_hf_band_frac` (3400-7000 Hz)
  - `hf_energy_frac` (>4000 Hz)
  - `ultra_hf_frac` (>7000 Hz)
  - `hi_ratio_3p5_7k` = E(3500-7000) / max(E(0-3500), eps)

Used by both:

- `scripts/build_call_segmenter_dataset.py`
- `scripts/predict_call_segmenter.py`

### Streaming audio decode (avoid temp MP3/WAV on disk)

Updated:

- `pipeline/audio_preprocess.py`

Adds `decode_audio_stream_to_float32(...)` which decodes audio via ffmpeg **streaming from S3 bytes**.

Rationale: the machine disk was extremely full; prior approaches that downloaded MP3/WAV intermediates hit "No space left on device".

### Wider text context (optional)

Updated dataset + predictor to support extracting hashed-text over a wider span:

- dataset flags:
  - `--text-context-s`
  - `--text-max-chars`
- meta carries:
  - `text_hashing.context_s`
  - `text_hashing.max_chars`

Note: v6b (context +/- 5s) showed overfitting/instability on the holdout eval set; v6a (context 0) is the stable improvement.

### Sweep / reporting / orchestration scripts

Added:

- `scripts/update_call_segmenter_split_meta.py`
- `scripts/report_call_segmenter_side_by_side.py`
- `scripts/run_call_segmenter_experiment.py`

Updated:

- `scripts/sweep_call_segmenter.py` (cache validation now includes audio/text config hashes; selection gates include no-call FP==0 and segment ratio constraints)
- `scripts/train_call_segmenter.py`

## Fixed Split Meta (stable eval)

Use this split meta for comparable iterations:

- `data/call_segmenter/split_meta_v1_plus3_plus10nocall.json`

Eval set is fixed at 6 videos:

- `D4uiHjHW4AU`, `RtOncKiOT48`, `S8yFUyD_JXU`, `bPv8JzD_bMI`, `ci-FdcWiJiA`, `pCcAlwOYYx8`

Train includes the original labeled set + 3 newly labeled + 10 no-call videos.

## Results (45 videos evaluated; 3 truth-only labels missing preds)

Baseline (pre-v6):

- Model: `data/call_segmenter/models/exp_v5c_neg5`
- S3 preds: `s3://rezora-whisperx-us-east-1-864981718771/call_segmenter/predictions/exp_v5c_neg5_20260204_thr065_off055_gap30_p020_min10/`
- Aggregate:
  - Time micro-F1: 0.8978
  - Segment IoU-F1 (micro): 0.3987
  - Pred/Truth segments: 159 / 157
  - No-call FP: 0/10

v6a (new acoustics + normalization, text context=0):

- Model: `data/call_segmenter/models/exp_v6a_neg5`
- S3 preds: `s3://rezora-whisperx-us-east-1-864981718771/call_segmenter/predictions/exp_v6a_neg5_20260204_202830_thr70_off50_gap30_p20_min10/`
- Aggregate:
  - Time micro-F1: 0.9227
  - Segment IoU-F1 (micro): 0.4684
  - Pred/Truth segments: 159 / 157
  - No-call FP: 0/10

## Side-by-side Reports (local)

Reports are generated to:

- `artifacts/s3_audit/call_segmenter/<report_name>_<timestamp>/side_by_side.md`
- `artifacts/s3_audit/call_segmenter/<report_name>_<timestamp>/side_by_side.json`

Example directories created during this run:

- `artifacts/s3_audit/call_segmenter/exp_v6a_neg5_20260204_203417/`
- `artifacts/s3_audit/call_segmenter/exp_v5c_neg5_20260204_thr065_off055_gap30_p020_min10_20260204_212057/`

## Repro Commands (typical loop)

1) Train model:

```bash
python scripts/train_call_segmenter.py \
  --data data/call_segmenter/v6a \
  --split-meta data/call_segmenter/split_meta_v1_plus3_plus10nocall.json \
  --neg-weight 5.0 \
  --output-dir data/call_segmenter/models/exp_v6a_neg5
```

2) Sweep post-processing:

```bash
python scripts/sweep_call_segmenter.py \
  --model-dir data/call_segmenter/models/exp_v6a_neg5 \
  --cache-dir data/call_segmenter/prob_cache_exp_v6a_neg5
```

3) Predict + evaluate:

```bash
python scripts/predict_call_segmenter.py \
  --video-list data/call_segmenter/labeled_videos_plus3_plus10nocall.txt \
  --model-dir data/call_segmenter/models/exp_v6a_neg5 \
  --threshold 0.70 --threshold-off 0.50 \
  --gap-merge-s 30 --gap-merge-min-p 0.20 \
  --min-seg-s 10 \
  --output-prefix s3://rezora-whisperx-us-east-1-864981718771/call_segmenter/predictions/<EXPERIMENT_PREFIX>/

python scripts/eval_call_segmenter_predictions.py \
  --predictions s3://rezora-whisperx-us-east-1-864981718771/call_segmenter/predictions/<EXPERIMENT_PREFIX>/ \
  --ground-truth s3
```

4) Generate truth-vs-pred report:

```bash
python scripts/report_call_segmenter_side_by_side.py \
  --pred-prefix s3://rezora-whisperx-us-east-1-864981718771/call_segmenter/predictions/<EXPERIMENT_PREFIX>/ \
  --labels-prefix s3://rezora-whisperx-us-east-1-864981718771/labeling/corrected_boundaries/v1/ \
  --out-root artifacts/s3_audit/call_segmenter \
  --report-name <EXPERIMENT_PREFIX>
```

## Known Gaps / Next Changes

- Segment IoU-F1 is still not "bulletproof" (many unmatched segments). The next cheap wins are:
  - make gap-merge less permissive (use mean/p90 over the gap instead of max)
  - dynamic min-seg rule (keep short segments if mean_p is high)
  - add more labeled call videos and/or more hard no-call negatives
