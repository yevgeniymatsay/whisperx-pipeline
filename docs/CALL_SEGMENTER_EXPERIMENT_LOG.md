# Call Segmenter Experiment Log

This log tracks model-training + post-processing sweep experiments with consistent evaluation.

## Metrics

- TimeF1: time-overlap F1 (micro) between predicted call time and truth call time.
- SegF1: segment IoU-F1 (micro) using greedy matching with IoU >= 0.5.
- SegRatio: total_pred_segments / total_truth_segments.

Holdout eval set is whatever is recorded in the model's `meta.json` (`eval_video_ids`).

---

## 2026-02-04: Baselines

### Baseline model v1

- Model dir: `data/call_segmenter/models/v1`
- Model git SHA: `64792daee01e`
- Sweep cmd (reduced grid):
  - `python scripts/sweep_call_segmenter.py --model-dir data/call_segmenter/models/v1 --cache-dir data/call_segmenter/prob_cache_v1_small \
      --threshold-range 0.50,0.90,0.05 --threshold-off-delta-values 0.0,0.10 \
      --gap-merge-values 5,10,20 --gap-merge-min-p-values 0,0.4 --min-seg-values 5,10 --top-n 10`

Best (constrained):
- thr_on=0.85, thr_off=0.75, gap=20, gap_min_p=0.00, min_seg=10

Eval (holdout 6 videos):
- TimeF1: 0.7953
- SegF1: 0.3133
- SegRatio: 1.44
- Pred: 49, Truth: 34

Notes:
- High thresholds + larger gap merge improved train score; eval still over-segments.

---

### exp_v4_base (audio features + +3 labeled videos)

- Dataset: `data/call_segmenter/v4` (35 included labeled videos; 3 excluded by QC)
- Model dir: `data/call_segmenter/models/exp_v4_base`
- Model git SHA: `e50a2bd867f1`
- Training params (from meta.json): neg_weight=2.7146, boundary_weight=0.0, boundary_tau=10.0
- Sweep cmd (full default grid):
  - `python scripts/sweep_call_segmenter.py --model-dir data/call_segmenter/models/exp_v4_base --cache-dir data/call_segmenter/prob_cache_exp_v4_base --output-csv data/call_segmenter/exp_v4_base_sweep.csv --top-n 20`

Best (constrained):
- thr_on=0.60, thr_off=0.60, gap=10, gap_min_p=0.00, min_seg=10

Eval (holdout 6 videos):
- TimeF1: 0.7778
- SegF1: 0.1980
- SegRatio: 1.97
- Pred: 67, Truth: 34

Notes:
- With current selection strategy (tune params on train only), holdout degraded vs v1.
- Next step is to add strong negatives (no-call videos) + retrain with updated weighting.

---

## 2026-02-04: Add no-call negatives (partial)

### exp_v5a_base (v5a dataset = call videos + 5 no-call videos)

- Dataset: `data/call_segmenter/v5a` (40 videos total; includes 5 no-call negatives)
- Split meta: `data/call_segmenter/split_meta_v1_plus3_plus5nocall.json`
- Model dir: `data/call_segmenter/models/exp_v5a_base`
- Model git SHA: `48b7582bddb8`
- Training cmd:
  - `python scripts/train_call_segmenter.py --data data/call_segmenter/v5a --split-meta data/call_segmenter/split_meta_v1_plus3_plus5nocall.json --output-dir data/call_segmenter/models/exp_v5a_base`
- Sweep cmd (reduced grid):
  - `python scripts/sweep_call_segmenter.py --model-dir data/call_segmenter/models/exp_v5a_base --cache-dir data/call_segmenter/prob_cache_exp_v5a_base \\\n+      --threshold-range 0.55,0.95,0.05 --threshold-off-delta-values 0.0,0.10 \\\n+      --gap-merge-values 10,20,30 --gap-merge-min-p-values 0,0.4 --min-seg-values 5,10 --top-n 10`

Best (constrained):
- thr_on=0.80, thr_off=0.70, gap=20, gap_min_p=0.00, min_seg=10

Eval (holdout 6 videos):
- TimeF1: 0.8227
- SegF1: 0.3421
- SegRatio: 1.24
- Pred: 42, Truth: 34

No-call sanity (train negatives, with best params):
- 0 predicted segments on: nNA0XQO_7jQ, 4OweikRF7bg, jvFLW5EClgk, hs44BGJtOwg, pjf5uhOMcTc

---

## 2026-02-04: Add no-call negatives (full 10) + retrain (v5c)

### exp_v5c_base (v5c dataset = call videos + 10 no-call videos; clean rerun for lioFo9pz9x8)

- Dataset: `data/call_segmenter/v5c` (call videos + 10 no-call videos)
- Split meta: `data/call_segmenter/split_meta_v1_plus3_plus10nocall.json`
- Model dir: `data/call_segmenter/models/exp_v5c_base`
- Model git SHA: `5ccbb49b083a`
- Sweep cmd (fuller grid):
  - `python scripts/sweep_call_segmenter.py --model-dir data/call_segmenter/models/exp_v5c_base --cache-dir data/call_segmenter/prob_cache_exp_v5c_base \
      --threshold-range 0.65,0.95,0.05 --threshold-off-delta-values 0.0,0.10,0.20 \
      --gap-merge-values 0,1,2,5,10,20,30 --gap-merge-min-p-values 0,0.2,0.4,0.6,0.8 --min-seg-values 1,3,5,10 --top-n 20`

Best (constrained):
- thr_on=0.85, thr_off=0.65, gap=30, gap_min_p=0.40, min_seg=5

Eval (holdout 6 videos):
- TimeF1: 0.8050
- SegF1: 0.3500
- SegRatio: 1.35
- Pred: 46, Truth: 34

Full labeled-set eval (45 videos = 35 call + 10 no-call; no-call FP==0 required):
- Predictions prefix: `call_segmenter/predictions/exp_v5c_base_20260204_thr085_off065_gap30_p040_min5/`
- TimeF1 (micro): 0.8897
- SegF1 (IoU micro): 0.3683
- Pred: 158, Truth: 157

Notes:
- Stable + zero no-call false positives, but segment IoU-F1 still limited; try stronger negative weighting.

---

### exp_v5c_neg3 (same dataset, higher negative weight)

- Model dir: `data/call_segmenter/models/exp_v5c_neg3`
- Training: `--neg-weight 3.0`
- Sweep best (constrained):
  - thr_on=0.75, thr_off=0.55, gap=20, gap_min_p=0.00, min_seg=10
- Eval (holdout 6 videos):
  - TimeF1: 0.7975
  - SegF1: 0.3678
  - SegRatio: 1.56

Full labeled-set eval (45 videos; no-call FP==0):
- Predictions prefix: `call_segmenter/predictions/exp_v5c_neg3_20260204_thr075_off055_gap20_min10/`
- TimeF1 (micro): 0.8846
- SegF1 (IoU micro): 0.3801
- Pred: 185, Truth: 157

Notes:
- Segment IoU-F1 improves, but over-segments (pred/truth ~1.18).

---

### exp_v5c_neg5 (best so far; higher negative weight)

- Model dir: `data/call_segmenter/models/exp_v5c_neg5`
- Training: `--neg-weight 5.0`
- Sweep best (constrained):
  - thr_on=0.65, thr_off=0.55, gap=30, gap_min_p=0.20, min_seg=10
- Eval (holdout 6 videos):
  - TimeF1: 0.8008
  - SegF1: 0.3797
  - SegRatio: 1.32

Full labeled-set eval (45 videos; no-call FP==0):
- Predictions prefix: `call_segmenter/predictions/exp_v5c_neg5_20260204_thr065_off055_gap30_p020_min10/`
- TimeF1 (micro): 0.8978
- SegF1 (IoU micro): 0.3987
- Pred: 159, Truth: 157

Notes:
- Best overall balance so far: higher time-F1, higher seg IoU-F1, segment ratio ~1.0, and 0/10 no-call false positives.

---

## 2026-02-04: v6 acoustic feature expansion + per-video normalization

### exp_v6a_neg5 (new cheap acoustic features + per-video normalization; text context = 0s)

- Dataset: `data/call_segmenter/v6a`
- Split meta: `data/call_segmenter/split_meta_v1_plus3_plus10nocall.json`
- Model dir: `data/call_segmenter/models/exp_v6a_neg5`
- Model git SHA: `b190a86eb512`
- Training: `--neg-weight 5.0` (no-call-video-weight=1.0)
- Sweep best (constrained):
  - thr_on=0.70, thr_off=0.50, gap=30, gap_min_p=0.20, min_seg=10

Full labeled-set eval (45 videos; no-call FP==0):
- Predictions prefix: `call_segmenter/predictions/exp_v6a_neg5_20260204_202830_thr70_off50_gap30_p20_min10/`
- TimeF1 (micro): 0.9227
- SegF1 (IoU micro): 0.4684
- Pred: 159, Truth: 157

Notes:
- Big jump in segment IoU-F1 vs exp_v5c_neg5 (~0.40 -> ~0.47) while keeping 0/10 no-call false positives.
- Remaining failure mode is still imperfect call splitting on a handful of long/multi-call videos; next step is to widen text context (v6b).

---

## 2026-02-05: exp_v8a_spectral

- Dataset: `data/call_segmenter/exp_v8a_spectral`
- Model dir: `data/call_segmenter/models/exp_v8a_spectral`
- Split meta: `data/call_segmenter/split_meta_v1_plus3_plus10nocall.json`

Best sweep params (train, constrained):
- decode=threshold thr_on=0.65, thr_off=0.55, gap=30, gap_min_p=0.20, min_seg=10
- Train: TimeF1=0.9427, SegF1=0.5316, SegRatio=0.93, NoCallFPVideos=0

Best sweep params (train, overall score):
- decode=threshold thr_on=0.65, thr_off=0.55, gap=30, gap_min_p=0.20, min_seg=10
- Train: TimeF1=0.9427, SegF1=0.5316, SegRatio=0.93, NoCallFPVideos=0

Holdout eval (6 videos) using best constrained params:
- TimeF1=0.8421, SegF1=0.3733, SegRatio=1.21, Pred=41, Truth=34

Full labeled-set eval (train+eval ids; no-call FP==0 required):
- Predictions prefix: `s3://rezora-whisperx-us-east-1-864981718771/call_segmenter/predictions/exp_v8a_spectral_20260205_232428_thr65_off55_gap30_gp90_p20_min10/`
- TimeF1(micro)=0.9251, SegF1(IoU micro)=0.4936, Pred=155, Truth=157, NoCallFPVideos=0

Side-by-side report:
- `artifacts/s3_audit/call_segmenter/exp_v8a_spectral_20260205_232859/side_by_side.md`
- `artifacts/s3_audit/call_segmenter/exp_v8a_spectral_20260205_232859/side_by_side.json`
---

## 2026-02-05: exp_v8b_mfcc

- Dataset: `data/call_segmenter/exp_v8b_mfcc`
- Model dir: `data/call_segmenter/models/exp_v8b_mfcc`
- Split meta: `data/call_segmenter/split_meta_v1_plus3_plus10nocall.json`

Best sweep params (train, constrained):
- decode=threshold thr_on=0.55, thr_off=0.45, gap=10, gap_min_p=0.00, min_seg=5
- Train: TimeF1=0.9649, SegF1=0.5812, SegRatio=0.90, NoCallFPVideos=0

Best sweep params (train, overall score):
- decode=threshold thr_on=0.55, thr_off=0.35, gap=10, gap_min_p=0.00, min_seg=10
- Train: TimeF1=0.9658, SegF1=0.6182, SegRatio=0.79, NoCallFPVideos=0

Holdout eval (6 videos) using best constrained params:
- TimeF1=0.8263, SegF1=0.2697, SegRatio=1.62, Pred=55, Truth=34

Full labeled-set eval (train+eval ids; no-call FP==0 required):
- Predictions prefix: `s3://rezora-whisperx-us-east-1-864981718771/call_segmenter/predictions/exp_v8b_mfcc_20260205_234409_thr55_off45_gap10_gmax_p00_min05/`
- TimeF1(micro)=0.9401, SegF1(IoU micro)=0.4954, Pred=166, Truth=157, NoCallFPVideos=0

Side-by-side report:
- `artifacts/s3_audit/call_segmenter/exp_v8b_mfcc_20260205_234908/side_by_side.md`
- `artifacts/s3_audit/call_segmenter/exp_v8b_mfcc_20260205_234908/side_by_side.json`
---

## 2026-02-06: exp_v8a_spectral_viterbi

- Dataset: `data/call_segmenter/exp_v8a_spectral`
- Model dir: `data/call_segmenter/models/exp_v8a_spectral`
- Split meta: `data/call_segmenter/split_meta_v1_plus3_plus10nocall.json`

Best sweep params (train, constrained):
- decode=viterbi enter_cost=4.00, exit_cost=2.00, call_bias=0.50, min_seg=10
- Train: TimeF1=0.9388, SegF1=0.3818, SegRatio=1.85, NoCallFPVideos=0

Best sweep params (train, overall score):
- decode=viterbi enter_cost=4.00, exit_cost=2.00, call_bias=0.50, min_seg=10
- Train: TimeF1=0.9388, SegF1=0.3818, SegRatio=1.85, NoCallFPVideos=0

Holdout eval (6 videos) using best constrained params:
- TimeF1=0.8236, SegF1=0.2883, SegRatio=2.26, Pred=77, Truth=34

Full labeled-set eval (train+eval ids; no-call FP==0 required):
- Predictions prefix: `s3://rezora-whisperx-us-east-1-864981718771/call_segmenter/predictions/exp_v8a_spectral_viterbi_20260205_235628_viterbi_enter400_exit200_bias050_min10/`
- TimeF1(micro)=0.9200, SegF1(IoU micro)=0.3593, Pred=305, Truth=157, NoCallFPVideos=0

Side-by-side report:
- `artifacts/s3_audit/call_segmenter/exp_v8a_spectral_viterbi_20260206_000143/side_by_side.md`
- `artifacts/s3_audit/call_segmenter/exp_v8a_spectral_viterbi_20260206_000143/side_by_side.json`
---

## 2026-02-06: exp_v8c_text5

- Dataset: `data/call_segmenter/exp_v8c_text5`
- Model dir: `data/call_segmenter/models/exp_v8c_text5`
- Split meta: `data/call_segmenter/split_meta_v1_plus3_plus10nocall.json`

Best sweep params (train, constrained):
- decode=threshold thr_on=0.60, thr_off=0.40, gap=1, gap_min_p=0.00, min_seg=3
- Train: TimeF1=0.9812, SegF1=0.8426, SegRatio=0.91, NoCallFPVideos=0

Best sweep params (train, overall score):
- decode=threshold thr_on=0.60, thr_off=0.40, gap=1, gap_min_p=0.00, min_seg=3
- Train: TimeF1=0.9812, SegF1=0.8426, SegRatio=0.91, NoCallFPVideos=0

Holdout eval (6 videos) using best constrained params:
- TimeF1=0.7974, SegF1=0.1837, SegRatio=4.76, Pred=162, Truth=34

Full labeled-set eval (train+eval ids; no-call FP==0 required):
- Predictions prefix: `s3://rezora-whisperx-us-east-1-864981718771/call_segmenter/predictions/exp_v8c_text5_20260206_002647_thr60_off40_gap01_gmax_p00_min03/`
- TimeF1(micro)=0.9526, SegF1(IoU micro)=0.5429, Pred=274, Truth=157, NoCallFPVideos=0

Side-by-side report:
- `artifacts/s3_audit/call_segmenter/exp_v8c_text5_20260206_003216/side_by_side.md`
- `artifacts/s3_audit/call_segmenter/exp_v8c_text5_20260206_003216/side_by_side.json`
---

## 2026-02-06: exp_v8d_boundary1

- Dataset: `data/call_segmenter/exp_v8a_spectral`
- Model dir: `data/call_segmenter/models/exp_v8d_boundary1`
- Split meta: `data/call_segmenter/split_meta_v1_plus3_plus10nocall.json`

Best sweep params (train, constrained):
- decode=threshold thr_on=0.70, thr_off=0.60, gap=30, gap_min_p=0.20, min_seg=10
- Train: TimeF1=0.9478, SegF1=0.5407, SegRatio=1.20, NoCallFPVideos=0

Best sweep params (train, overall score):
- decode=threshold thr_on=0.70, thr_off=0.60, gap=30, gap_min_p=0.20, min_seg=10
- Train: TimeF1=0.9478, SegF1=0.5407, SegRatio=1.20, NoCallFPVideos=0

Holdout eval (6 videos) using best constrained params:
- TimeF1=0.8377, SegF1=0.3448, SegRatio=1.56, Pred=53, Truth=34

Full labeled-set eval (train+eval ids; no-call FP==0 required):
- Predictions prefix: `s3://rezora-whisperx-us-east-1-864981718771/call_segmenter/predictions/exp_v8d_boundary1_20260206_004443_thr70_off60_gap30_gmean_p20_min10/`
- TimeF1(micro)=0.9290, SegF1(IoU micro)=0.4930, Pred=200, Truth=157, NoCallFPVideos=0

Side-by-side report:
- `artifacts/s3_audit/call_segmenter/exp_v8d_boundary1_20260206_004922/side_by_side.md`
- `artifacts/s3_audit/call_segmenter/exp_v8d_boundary1_20260206_004922/side_by_side.json`
---

## 2026-02-06: ab_pilot9_wx_notext_spectral

- Dataset: `data/call_segmenter/ab_pilot9_wx_notext_spectral`
- Model dir: `data/call_segmenter/models/ab_pilot9_wx_notext_spectral`
- Split meta: `data/call_segmenter/split_meta_azure_pilot9.json`

Best sweep params (train, constrained):
- decode=threshold thr_on=0.95, thr_off=0.95, gap=5, gap_min_p=0.60, min_seg=1
- Train: TimeF1=0.9926, SegF1=0.9600, SegRatio=1.08, NoCallFPVideos=0

Best sweep params (train, overall score):
- decode=threshold thr_on=0.95, thr_off=0.95, gap=5, gap_min_p=0.60, min_seg=1
- Train: TimeF1=0.9926, SegF1=0.9600, SegRatio=1.08, NoCallFPVideos=0

Holdout eval (6 videos) using best constrained params:
- TimeF1=0.6414, SegF1=0.0000, SegRatio=19.50, Pred=156, Truth=8

Full labeled-set eval (train+eval ids; no-call FP==0 required):
- Predictions prefix: `s3://rezora-whisperx-us-east-1-864981718771/call_segmenter/predictions/ab_pilot9_wx_notext_spectral_20260206_042020_thr95_off95_gap05_gmean_p60_min01/`
- TimeF1(micro)=0.9018, SegF1(IoU micro)=0.1270, Pred=169, Truth=20, NoCallFPVideos=1

Side-by-side report:
- `artifacts/s3_audit/call_segmenter/ab_pilot9_wx_notext_spectral_20260206_042102/side_by_side.md`
- `artifacts/s3_audit/call_segmenter/ab_pilot9_wx_notext_spectral_20260206_042102/side_by_side.json`
---

## 2026-02-06: ab_pilot9_az_notext_spectral

- Dataset: `data/call_segmenter/ab_pilot9_az_notext_spectral`
- Model dir: `data/call_segmenter/models/ab_pilot9_az_notext_spectral`
- Split meta: `data/call_segmenter/split_meta_azure_pilot9.json`

Best sweep params (train, constrained):
- decode=threshold thr_on=0.55, thr_off=0.45, gap=2, gap_min_p=0.00, min_seg=3
- Train: TimeF1=0.9954, SegF1=0.8800, SegRatio=1.08, NoCallFPVideos=0

Best sweep params (train, overall score):
- decode=threshold thr_on=0.80, thr_off=0.80, gap=2, gap_min_p=0.00, min_seg=1
- Train: TimeF1=0.9949, SegF1=0.8889, SegRatio=1.25, NoCallFPVideos=0

Holdout eval (6 videos) using best constrained params:
- TimeF1=0.6630, SegF1=0.1587, SegRatio=6.88, Pred=55, Truth=8

Full labeled-set eval (train+eval ids; no-call FP==0 required):
- Predictions prefix: `s3://rezora-whisperx-us-east-1-864981718771/call_segmenter/predictions/ab_pilot9_az_notext_spectral_20260206_042222_thr55_off45_gap02_gmax_p00_min03/`
- TimeF1(micro)=0.8926, SegF1(IoU micro)=0.3636, Pred=68, Truth=20, NoCallFPVideos=1

Side-by-side report:
- `artifacts/s3_audit/call_segmenter/ab_pilot9_az_notext_spectral_20260206_042246/side_by_side.md`
- `artifacts/s3_audit/call_segmenter/ab_pilot9_az_notext_spectral_20260206_042246/side_by_side.json`