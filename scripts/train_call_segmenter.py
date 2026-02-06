#!/usr/bin/env python3
"""Train call segmenter models (XGBoost) on a v2 dataset.

Trains and evaluates two models with a strict split by video_id:
  - Model A: diarization-only numeric features
  - Model B: diarization + hashed char n-grams (if text_features.npz exists)
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


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


def split_by_video(df: pd.DataFrame, eval_frac: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    vids = df["video_id"].dropna().unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(vids)
    n_eval = max(1, int(round(len(vids) * eval_frac)))
    eval_vids = set(vids[:n_eval])
    train_df = df[~df["video_id"].isin(eval_vids)].copy().reset_index(drop=True)
    eval_df = df[df["video_id"].isin(eval_vids)].copy().reset_index(drop=True)
    return train_df, eval_df, sorted(eval_vids)


def split_by_fixed_meta(df: pd.DataFrame, split_meta_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """Split rows by video_id using an existing meta.json containing train/eval ids."""
    meta = json.loads(split_meta_path.read_text())
    train_vids = meta.get("train_video_ids")
    eval_vids = meta.get("eval_video_ids")
    if not isinstance(train_vids, list) or not isinstance(eval_vids, list):
        raise ValueError(f"{split_meta_path} missing train_video_ids/eval_video_ids")

    train_vids = [str(v) for v in train_vids]
    eval_vids = [str(v) for v in eval_vids]

    overlap = set(train_vids) & set(eval_vids)
    if overlap:
        raise ValueError(f"Split meta has overlapping train/eval video_ids: {sorted(overlap)[:10]}")

    present = set(df["video_id"].dropna().astype(str).unique().tolist())
    missing_train = [v for v in train_vids if v not in present]
    missing_eval = [v for v in eval_vids if v not in present]
    if missing_train or missing_eval:
        raise ValueError(
            f"Split meta video_ids missing from dataset (train_missing={missing_train[:5]} eval_missing={missing_eval[:5]})"
        )

    train_df = df[df["video_id"].isin(train_vids)].copy().reset_index(drop=True)
    eval_df = df[df["video_id"].isin(eval_vids)].copy().reset_index(drop=True)

    return train_df, eval_df, sorted(eval_vids), sorted(train_vids)


def classification_report(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1), "roc_auc": float(auc)}


def per_video_report(df_eval: pd.DataFrame, y_prob: np.ndarray, threshold: float) -> List[Dict]:
    out: List[Dict] = []
    for vid, grp in df_eval.groupby("video_id"):
        y_true = grp["y"].to_numpy(dtype=int)
        # df_eval is reset_index'd; group indices are safe to use as positions into y_prob.
        idx = grp.index.to_numpy(dtype=int)
        rep = classification_report(y_true, y_prob[idx], threshold)
        rep.update({"video_id": vid, "rows": int(len(grp)), "pos_frac": float(y_true.mean()) if len(y_true) else 0.0})
        out.append(rep)
    out.sort(key=lambda r: r["f1"])
    return out


def load_text_features(data_dir: Path) -> Tuple[sparse.csr_matrix, np.ndarray]:
    X_path = data_dir / "text_features.npz"
    ids_path = data_dir / "text_row_ids.npy"
    if not X_path.exists() or not ids_path.exists():
        raise FileNotFoundError("Missing text_features.npz or text_row_ids.npy")
    X = sparse.load_npz(X_path).tocsr()
    row_ids = np.load(ids_path).astype(np.int64)
    return X, row_ids


def align_text_rows(X_text: sparse.csr_matrix, text_row_ids: np.ndarray, target_row_ids: np.ndarray) -> sparse.csr_matrix:
    # text_row_ids are saved sorted ascending by the dataset builder.
    pos = np.searchsorted(text_row_ids, target_row_ids)
    if np.any(pos < 0) or np.any(pos >= len(text_row_ids)):
        raise ValueError("Some target row_ids are outside text_row_ids range")
    if not np.array_equal(text_row_ids[pos], target_row_ids):
        missing = target_row_ids[text_row_ids[pos] != target_row_ids][:10]
        raise ValueError(f"Row_id mismatch between parquet and text features (sample missing: {missing})")
    return X_text[pos]


def get_git_sha() -> str:
    """Get short git SHA for reproducibility tracking."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            check=True,
        )
        return result.stdout.strip()[:12]
    except Exception:
        return "unknown"


def save_model_b(
    model,
    output_dir: Path,
    feature_columns: List[str],
    text_features_meta: Dict,
    audio_features_meta: Optional[Dict],
    window_config: Dict,
    seed: int,
    threshold: float,
    neg_weight: float,
    boundary_weight: float,
    boundary_tau: float,
    train_video_ids: List[str],
    eval_video_ids: List[str],
    eval_metrics: Dict[str, float],
) -> None:
    """Save Model B artifacts for inference."""
    import xgboost as xgb

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model in Universal Binary JSON format
    model_path = output_dir / "model_b.ubj"
    model.save_model(str(model_path))

    # Save metadata for inference reproducibility
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_sha(),
        "seed": seed,
        "default_threshold": threshold,
        "feature_columns": feature_columns,
        "text_hashing": {
            "n_features": text_features_meta.get("n_features", 4096),
            "ngram_range": text_features_meta.get("ngram_range", [2, 5]),
            "analyzer": text_features_meta.get("analyzer", "char_wb"),
            "context_s": float(text_features_meta.get("context_s", 0.0)),
            "max_chars": int(text_features_meta.get("max_chars", 300)),
        },
        "audio_features": audio_features_meta,
        "window_config": window_config,
        "training": {
            "neg_weight": float(neg_weight),
            "boundary_weight": float(boundary_weight),
            "boundary_tau": float(boundary_tau),
        },
        "versions": {
            "xgboost": xgb.__version__,
            "sklearn": sklearn.__version__,
            "scipy": scipy.__version__,
        },
        "train_video_ids": sorted(train_video_ids),
        "eval_video_ids": sorted(eval_video_ids),
        "eval_metrics": eval_metrics,
    }

    meta_path = output_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved Model B artifacts to {output_dir}/")
    print(f"  model_b.ubj: {model_path.stat().st_size / 1024:.1f} KB")
    print(f"  meta.json: {meta_path}")


def save_model_a(
    model,
    output_dir: Path,
    feature_columns: List[str],
    text_features_meta: Dict,
    audio_features_meta: Optional[Dict],
    window_config: Dict,
    seed: int,
    threshold: float,
    neg_weight: float,
    boundary_weight: float,
    boundary_tau: float,
    train_video_ids: List[str],
    eval_video_ids: List[str],
    eval_metrics: Dict[str, float],
) -> None:
    """Save Model A (numeric-only) artifacts for inference."""
    import xgboost as xgb

    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model_a.ubj"
    model.save_model(str(model_path))

    # Keep schema compatible with predict/sweep code: include text_hashing, but mark disabled.
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_sha(),
        "seed": seed,
        "default_threshold": threshold,
        "model_variant": "a",
        "feature_columns": feature_columns,
        "text_hashing": {
            "enabled": False,
            "n_features": int(text_features_meta.get("n_features", 4096)),
            "ngram_range": text_features_meta.get("ngram_range", [2, 5]),
            "analyzer": text_features_meta.get("analyzer", "char_wb"),
            "context_s": float(text_features_meta.get("context_s", 0.0)),
            "max_chars": int(text_features_meta.get("max_chars", 300)),
        },
        "audio_features": audio_features_meta,
        "window_config": window_config,
        "training": {
            "neg_weight": float(neg_weight),
            "boundary_weight": float(boundary_weight),
            "boundary_tau": float(boundary_tau),
        },
        "versions": {
            "xgboost": xgb.__version__,
            "sklearn": sklearn.__version__,
            "scipy": scipy.__version__,
        },
        "train_video_ids": sorted(train_video_ids),
        "eval_video_ids": sorted(eval_video_ids),
        "eval_metrics": eval_metrics,
    }

    meta_path = output_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"\nSaved Model A artifacts to {output_dir}/")
    print(f"  model_a.ubj: {model_path.stat().st_size / 1024:.1f} KB")
    print(f"  meta.json: {meta_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train call segmenter (XGBoost) on v2 dataset")
    parser.add_argument("--data", type=Path, default=Path("data/call_segmenter/v2"))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eval-frac", type=float, default=0.2)
    parser.add_argument("--split-meta", type=Path, default=None,
                        help="Use fixed train/eval split from an existing model meta.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--neg-weight", type=float, default=None,
                        help="Weight multiplier for y==0 windows (default: pos/neg in train split)")
    parser.add_argument("--no-call-video-weight", type=float, default=1.0,
                        help="Extra multiplier applied to all windows from videos that contain 0 positive windows "
                             "(i.e., 'no-call' negatives). Use <1.0 to keep hard negatives from dominating.")
    parser.add_argument("--boundary-weight", type=float, default=0.0,
                        help="Extra weight for windows near boundaries: 1 + w*exp(-|dist|/tau)")
    parser.add_argument("--boundary-tau", type=float, default=10.0,
                        help="Decay timescale (seconds) for boundary weighting")
    parser.add_argument("--no-text", action="store_true", help="Skip training Model B (text)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/call_segmenter/models/v2"),
                        help="Directory to save trained model artifacts")
    parser.add_argument("--no-save", action="store_true", help="Skip saving model artifacts")
    args = parser.parse_args()

    try:
        import xgboost as xgb
    except Exception as e:
        print("ERROR: xgboost is not installed.")
        print("Install it with: pip install xgboost")
        print(f"Details: {e}")
        return 2

    df, meta = load_dataset(args.data)

    # Drop ignored windows (near boundaries) for training/eval.
    if "ignore" in df.columns:
        df = df[df["ignore"] == 0].copy()
    df = df.dropna(subset=["y", "video_id", "row_id"])

    feature_cols = meta.get("feature_columns", [])
    if not feature_cols:
        raise ValueError("dataset_meta.json missing feature_columns")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in parquet: {missing}")

    if args.split_meta:
        train_df, eval_df, eval_vids, train_vids = split_by_fixed_meta(df, args.split_meta)
        print(f"Split (fixed): train videos={len(train_vids)} eval videos={len(eval_vids)}")
    else:
        train_df, eval_df, eval_vids = split_by_video(df, args.eval_frac, args.seed)
        train_vids = sorted(train_df['video_id'].dropna().unique().tolist())
        print(f"Split (random): train videos={train_df['video_id'].nunique()} eval videos={len(eval_vids)}")
        print(f"Eval videos: {', '.join(eval_vids)}")

    print(f"Rows: train={len(train_df)} eval={len(eval_df)}")

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["y"].to_numpy(dtype=int)
    X_eval = eval_df[feature_cols].to_numpy(dtype=np.float32)
    y_eval = eval_df["y"].to_numpy(dtype=int)

    # Sample weights: emphasize negatives and boundary-adjacent windows to improve split quality.
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    neg_weight = float(args.neg_weight) if args.neg_weight is not None else (pos / max(neg, 1))

    w_train = np.ones(len(train_df), dtype=np.float32)
    w_train[y_train == 0] *= neg_weight

    # Optionally down-weight (or up-weight) entire "no-call" videos (videos with 0 positive windows).
    if float(args.no_call_video_weight) != 1.0:
        per_vid_pos = train_df.groupby("video_id")["y"].sum()
        no_call_vids = set(per_vid_pos[per_vid_pos == 0].index.astype(str).tolist())
        if no_call_vids:
            mask = train_df["video_id"].astype(str).isin(no_call_vids).to_numpy()
            w_train[mask] *= float(args.no_call_video_weight)
            print(f"No-call videos in train split: {len(no_call_vids)} (weight={float(args.no_call_video_weight)})")

    if args.boundary_weight > 0.0 and "dist_to_boundary_s" in train_df.columns:
        dist = train_df["dist_to_boundary_s"].to_numpy(dtype=np.float32)
        dist = np.abs(dist)
        w_train *= (1.0 + float(args.boundary_weight) * np.exp(-dist / float(args.boundary_tau)))

    # Model A (numeric)
    model_a = xgb.XGBClassifier(
        booster="gbtree",
        objective="binary:logistic",
        n_estimators=600,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=0.0,
        tree_method="hist",
        n_jobs=8,
        random_state=args.seed,
    )
    model_a.fit(X_train, y_train, sample_weight=w_train)
    prob_a = model_a.predict_proba(X_eval)[:, 1]

    rep_a = classification_report(y_eval, prob_a, args.threshold)
    print("\nModel A (diarization only)")
    print(rep_a)

    per_vid_a = per_video_report(eval_df, prob_a, args.threshold)
    print("\nModel A per-video (worst first)")
    for r in per_vid_a[:10]:
        print({k: r[k] for k in ["video_id", "rows", "pos_frac", "precision", "recall", "f1"]})

    # Model B (numeric + text)
    if args.no_text:
        if not args.no_save:
            text_features_meta = meta.get("text_features", {}) or {}
            audio_features_meta = meta.get("config", {}).get("audio_features")
            window_config = {
                "win_s": meta["config"]["win_s"],
                "hop_s": meta["config"]["hop_s"],
                "ignore_s": meta["config"]["ignore_s"],
            }
            save_model_a(
                model=model_a,
                output_dir=args.output_dir,
                feature_columns=feature_cols,
                text_features_meta=text_features_meta,
                audio_features_meta=audio_features_meta,
                window_config=window_config,
                seed=args.seed,
                threshold=args.threshold,
                neg_weight=neg_weight,
                boundary_weight=args.boundary_weight,
                boundary_tau=args.boundary_tau,
                train_video_ids=train_vids,
                eval_video_ids=eval_vids,
                eval_metrics=rep_a,
            )
        return 0

    try:
        X_text_all, text_row_ids = load_text_features(args.data)
    except Exception as e:
        print(f"\nSkipping Model B: {e}")
        if not args.no_save:
            text_features_meta = meta.get("text_features", {}) or {}
            audio_features_meta = meta.get("config", {}).get("audio_features")
            window_config = {
                "win_s": meta["config"]["win_s"],
                "hop_s": meta["config"]["hop_s"],
                "ignore_s": meta["config"]["ignore_s"],
            }
            save_model_a(
                model=model_a,
                output_dir=args.output_dir,
                feature_columns=feature_cols,
                text_features_meta=text_features_meta,
                audio_features_meta=audio_features_meta,
                window_config=window_config,
                seed=args.seed,
                threshold=args.threshold,
                neg_weight=neg_weight,
                boundary_weight=args.boundary_weight,
                boundary_tau=args.boundary_tau,
                train_video_ids=train_vids,
                eval_video_ids=eval_vids,
                eval_metrics=rep_a,
            )
        return 0

    # Align text features with train/eval rows using row_id join.
    train_row_ids = train_df["row_id"].to_numpy(dtype=np.int64)
    eval_row_ids = eval_df["row_id"].to_numpy(dtype=np.int64)

    X_text_train = align_text_rows(X_text_all, text_row_ids, train_row_ids)
    X_text_eval = align_text_rows(X_text_all, text_row_ids, eval_row_ids)

    X_train_comb = sparse.hstack([sparse.csr_matrix(X_train), X_text_train], format="csr")
    X_eval_comb = sparse.hstack([sparse.csr_matrix(X_eval), X_text_eval], format="csr")

    model_b = xgb.XGBClassifier(
        booster="gbtree",
        objective="binary:logistic",
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        reg_alpha=0.0,
        tree_method="hist",
        n_jobs=8,
        random_state=args.seed,
    )
    model_b.fit(X_train_comb, y_train, sample_weight=w_train)
    prob_b = model_b.predict_proba(X_eval_comb)[:, 1]

    rep_b = classification_report(y_eval, prob_b, args.threshold)
    print("\nModel B (diarization + hashed char n-grams)")
    print(rep_b)

    per_vid_b = per_video_report(eval_df, prob_b, args.threshold)
    print("\nModel B per-video (worst first)")
    for r in per_vid_b[:10]:
        print({k: r[k] for k in ["video_id", "rows", "pos_frac", "precision", "recall", "f1"]})

    # Save Model B artifacts
    if not args.no_save:
        text_features_meta = meta.get("text_features", {})
        audio_features_meta = meta.get("config", {}).get("audio_features")
        window_config = {
            "win_s": meta["config"]["win_s"],
            "hop_s": meta["config"]["hop_s"],
            "ignore_s": meta["config"]["ignore_s"],
        }
        save_model_b(
            model=model_b,
            output_dir=args.output_dir,
            feature_columns=feature_cols,
            text_features_meta=text_features_meta,
            audio_features_meta=audio_features_meta,
            window_config=window_config,
            seed=args.seed,
            threshold=args.threshold,
            neg_weight=neg_weight,
            boundary_weight=args.boundary_weight,
            boundary_tau=args.boundary_tau,
            train_video_ids=train_vids,
            eval_video_ids=eval_vids,
            eval_metrics=rep_b,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
