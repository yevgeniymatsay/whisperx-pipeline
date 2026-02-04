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
