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
