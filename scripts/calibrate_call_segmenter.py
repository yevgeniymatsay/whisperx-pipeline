#!/usr/bin/env python3
"""Fit a simple probability calibration layer for a trained call-segmenter model.

We fit Platt scaling on the model's per-window probabilities:
  p_cal = sigmoid(a * logit(p_raw) + b)

This is useful for:
  - more stable thresholding/hysteresis
  - better-behaved emissions for Viterbi/HMM decoding
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

# Ensure sibling script imports work when invoked as `python scripts/...`.
import sys

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.call_segmenter.calibration import PlattCalibration
from predict_call_segmenter import load_model_and_meta  # type: ignore[import-not-found]


def load_dataset(data_dir: Path) -> Tuple[pd.DataFrame, Dict]:
    meta_path = data_dir / "dataset_meta.json"
    parquet_path = data_dir / "windows_all.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing {parquet_path}")
    meta = json.loads(meta_path.read_text())
    df = pd.read_parquet(parquet_path)
    return df, meta


def load_text_features(data_dir: Path) -> Tuple[sparse.csr_matrix, np.ndarray]:
    X_path = data_dir / "text_features.npz"
    ids_path = data_dir / "text_row_ids.npy"
    if not X_path.exists() or not ids_path.exists():
        raise FileNotFoundError("Missing text_features.npz or text_row_ids.npy")
    X = sparse.load_npz(X_path).tocsr()
    row_ids = np.load(ids_path).astype(np.int64)
    return X, row_ids


def align_text_rows(X_text: sparse.csr_matrix, text_row_ids: np.ndarray, target_row_ids: np.ndarray) -> sparse.csr_matrix:
    pos = np.searchsorted(text_row_ids, target_row_ids)
    if np.any(pos < 0) or np.any(pos >= len(text_row_ids)):
        raise ValueError("Some target row_ids are outside text_row_ids range")
    if not np.array_equal(text_row_ids[pos], target_row_ids):
        bad = target_row_ids[text_row_ids[pos] != target_row_ids][:10]
        raise ValueError(f"Row_id mismatch between parquet and text features (sample: {bad})")
    return X_text[pos]


def log_loss(y: np.ndarray, p: np.ndarray, eps: float) -> float:
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1.0 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate call segmenter probabilities (Platt scaling)")
    parser.add_argument("--model-dir", type=Path, required=True, help="Model dir containing model_b.ubj and meta.json")
    parser.add_argument("--data-dir", type=Path, required=True, help="Dataset dir containing windows_all.parquet")
    parser.add_argument("--split-meta", type=Path, required=True, help="Fixed split meta JSON with train_video_ids")
    parser.add_argument("--output", type=Path, default=None, help="Output calibration JSON (default: model_dir/calibration.json)")
    parser.add_argument("--n-splits", type=int, default=5, help="GroupKFold splits (by video_id) for diagnostics")
    parser.add_argument("--eps", type=float, default=1e-6, help="Clamp epsilon for logits/logloss")
    args = parser.parse_args()

    out_path = args.output or (args.model_dir / "calibration.json")

    model, model_meta = load_model_and_meta(args.model_dir)
    feature_columns = model_meta["feature_columns"]

    df, _ds_meta = load_dataset(args.data_dir)
    if "y" not in df.columns:
        raise ValueError("Dataset parquet missing column: y")
    if "video_id" not in df.columns:
        raise ValueError("Dataset parquet missing column: video_id")
    if "row_id" not in df.columns:
        raise ValueError("Dataset parquet missing column: row_id")

    split_meta = json.loads(args.split_meta.read_text())
    train_vids = [str(v) for v in split_meta.get("train_video_ids", [])]
    if not train_vids:
        raise ValueError("--split-meta missing train_video_ids")

    df = df[df["video_id"].astype(str).isin(train_vids)].copy().reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No rows after filtering to train_video_ids")

    # Numeric features
    missing_cols = [c for c in feature_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required feature columns (sample): {missing_cols[:10]}")
    X_num = df[feature_columns].to_numpy(dtype=np.float32)
    X_num_sp = sparse.csr_matrix(X_num)

    # Text features (optional)
    X = X_num_sp
    try:
        X_text, text_row_ids = load_text_features(args.data_dir)
        row_ids = df["row_id"].to_numpy(dtype=np.int64)
        X_text_aligned = align_text_rows(X_text, text_row_ids, row_ids)
        X = sparse.hstack([X_num_sp, X_text_aligned], format="csr")
        print(f"Using text features: {X_text_aligned.shape[1]} dims")
    except FileNotFoundError:
        print("No text_features.npz found; calibrating numeric-only probabilities")

    y = df["y"].to_numpy(dtype=np.int32)
    groups = df["video_id"].astype(str).to_numpy()

    # Raw model probabilities -> logits.
    p_raw = model.predict_proba(X)[:, 1].astype(np.float64, copy=False)
    p_raw = np.clip(p_raw, float(args.eps), 1.0 - float(args.eps))
    x_logit = np.log(p_raw / (1.0 - p_raw)).reshape(-1, 1)

    # Diagnostics: group-k-fold evaluation of calibration mapping.
    n_splits = int(args.n_splits)
    n_splits = max(2, min(n_splits, len(np.unique(groups))))
    gkf = GroupKFold(n_splits=n_splits)

    brier_raw: List[float] = []
    brier_cal: List[float] = []
    ll_raw: List[float] = []
    ll_cal: List[float] = []

    for tr_idx, va_idx in gkf.split(x_logit, y, groups=groups):
        lr = LogisticRegression(solver="lbfgs", max_iter=2000)
        lr.fit(x_logit[tr_idx], y[tr_idx])
        a = float(lr.coef_.reshape(-1)[0])
        b0 = float(lr.intercept_.reshape(-1)[0])

        z = a * x_logit[va_idx].reshape(-1) + b0
        p_cal = 1.0 / (1.0 + np.exp(-z))

        brier_raw.append(brier(y[va_idx], p_raw[va_idx]))
        brier_cal.append(brier(y[va_idx], p_cal))
        ll_raw.append(log_loss(y[va_idx], p_raw[va_idx], eps=float(args.eps)))
        ll_cal.append(log_loss(y[va_idx], p_cal, eps=float(args.eps)))

    # Fit final calibrator on all train windows.
    lr = LogisticRegression(solver="lbfgs", max_iter=2000)
    lr.fit(x_logit, y)
    a = float(lr.coef_.reshape(-1)[0])
    b0 = float(lr.intercept_.reshape(-1)[0])

    calib = PlattCalibration(a=a, b=b0, eps=float(args.eps))

    out = calib.to_dict()
    out.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_git_sha": model_meta.get("git_sha", "unknown"),
            "data_dir": str(args.data_dir),
            "split_meta": str(args.split_meta),
            "train_videos": len(train_vids),
            "train_rows": int(len(df)),
            "groupkfold": {
                "n_splits": n_splits,
                "brier_raw_mean": float(np.mean(brier_raw)) if brier_raw else None,
                "brier_cal_mean": float(np.mean(brier_cal)) if brier_cal else None,
                "logloss_raw_mean": float(np.mean(ll_raw)) if ll_raw else None,
                "logloss_cal_mean": float(np.mean(ll_cal)) if ll_cal else None,
            },
        }
    )

    out_path.write_text(json.dumps(out, indent=2) + "\n")

    print(f"Wrote calibration: {out_path}")
    g = out["groupkfold"]
    if g.get("brier_raw_mean") is not None:
        print(
            "GroupKFold diagnostics:\n"
            f"  Brier:   raw={g['brier_raw_mean']:.6f}  cal={g['brier_cal_mean']:.6f}\n"
            f"  LogLoss: raw={g['logloss_raw_mean']:.6f}  cal={g['logloss_cal_mean']:.6f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

