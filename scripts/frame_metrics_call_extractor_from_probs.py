#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import AWS_REGION, S3_BUCKET
from pipeline.call_extractor_wavlm.io import s3_download_if_missing, s3_read_json
from pipeline.call_extractor_wavlm.labels import TargetConfig, make_targets_for_frames, parse_video_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if y_true.size == 0:
        return None
    pos = float(y_true.sum())
    if pos == 0.0 or pos == float(y_true.size):
        return None
    try:
        from sklearn.metrics import average_precision_score
    except Exception as e:  # pragma: no cover
        raise RuntimeError("scikit-learn is required: pip install scikit-learn") from e
    return float(average_precision_score(y_true, y_score))


def _metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    y_true = y_true.astype(np.float32, copy=False)
    y_score = y_score.astype(np.float32, copy=False)
    pos_mask = y_true >= 0.5
    neg_mask = ~pos_mask
    pos_frac = float(pos_mask.mean()) if y_true.size else 0.0
    mean_pos = float(y_score[pos_mask].mean()) if pos_mask.any() else None
    mean_neg = float(y_score[neg_mask].mean()) if neg_mask.any() else None
    ap = _average_precision(y_true, y_score)

    q = np.quantile(y_score, [0.5, 0.9, 0.99]).astype(float) if y_score.size else np.array([0.0, 0.0, 0.0])
    return {
        "pos_frac": pos_frac,
        "mean_p_pos": mean_pos,
        "mean_p_neg": mean_neg,
        "avg_precision": ap,
        "p50": float(q[0]),
        "p90": float(q[1]),
        "p99": float(q[2]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Frame-level metrics from saved per-frame probs (*.npz) + S3 labels.")
    parser.add_argument("--split-config", type=Path, default=Path("configs/call_extractor/split_v1.config.json"))
    parser.add_argument("--output-config", type=Path, default=Path("configs/call_extractor/output_wavlm_large_v1.config.json"))
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default=None,
        help="Override S3 prefix for downloading probs (default: output-config's s3_output_prefix)",
    )
    parser.add_argument(
        "--probs-dir",
        type=Path,
        default=None,
        help="Local dir containing {video_id}.npz (default: <local_artifacts_dir>/predictions)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=["eval", "train", "all"],
        default="eval",
        help="Which split videos to process (default: eval)",
    )
    parser.add_argument("--start-tolerance-s", type=float, default=0.20)
    parser.add_argument("--end-tolerance-s", type=float, default=0.20)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("artifacts/call_extractor/wavlm_large_v1/frame_metrics.json"),
    )
    args = parser.parse_args()

    split_cfg = _load_json(args.split_config)
    out_cfg = _load_json(args.output_config)
    s3_prefix = str(args.s3_prefix or out_cfg.get("s3_output_prefix", "call_extractor/wavlm_large_v1/")).rstrip("/")

    probs_dir = args.probs_dir
    if probs_dir is None:
        probs_dir = Path(out_cfg.get("local_artifacts_dir", "artifacts/call_extractor/wavlm_large_v1")) / "predictions"
    probs_dir.mkdir(parents=True, exist_ok=True)

    if args.subset == "eval":
        video_ids = list(split_cfg["eval_video_ids"])
    elif args.subset == "train":
        video_ids = list(split_cfg["train_video_ids"])
    else:
        video_ids = list(split_cfg["eval_video_ids"]) + list(split_cfg["train_video_ids"])

    label_prefix = str(split_cfg["label_prefix"])
    target_cfg = TargetConfig(start_tolerance_s=float(args.start_tolerance_s), end_tolerance_s=float(args.end_tolerance_s))

    all_true: dict[str, list[np.ndarray]] = {"in_call": [], "start": [], "end": []}
    all_score: dict[str, list[np.ndarray]] = {"in_call": [], "start": [], "end": []}
    per_video: list[dict[str, Any]] = []

    for vid in video_ids:
        vid = str(vid)
        local = probs_dir / f"{vid}.npz"
        if not local.exists():
            s3_key = f"{s3_prefix}/probs/{vid}.npz"
            s3_download_if_missing(S3_BUCKET, s3_key, local, region=AWS_REGION)
        arr = np.load(local)
        times_s = arr["times_s"].astype(np.float32, copy=False)
        in_call = arr["in_call"].astype(np.float32, copy=False)
        start = arr["start"].astype(np.float32, copy=False)
        end = arr["end"].astype(np.float32, copy=False)

        label_key = f"{label_prefix}{vid}.json"
        labels = parse_video_labels(s3_read_json(S3_BUCKET, label_key, region=AWS_REGION))
        y_in_call, y_start, y_end = make_targets_for_frames(times_s, boundaries=labels.boundaries, cfg=target_cfg)

        all_true["in_call"].append(y_in_call)
        all_true["start"].append(y_start)
        all_true["end"].append(y_end)
        all_score["in_call"].append(in_call)
        all_score["start"].append(start)
        all_score["end"].append(end)

        per_video.append(
            {
                "video_id": vid,
                "metrics": {
                    "in_call": _metrics(y_in_call, in_call),
                    "start": _metrics(y_start, start),
                    "end": _metrics(y_end, end),
                },
            }
        )

    report = {
        "subset": args.subset,
        "videos": len(video_ids),
        "tolerances_s": {"start": float(args.start_tolerance_s), "end": float(args.end_tolerance_s)},
        "aggregate": {},
        "per_video": per_video,
    }

    for head in ["in_call", "start", "end"]:
        y_true = np.concatenate(all_true[head]) if all_true[head] else np.zeros((0,), dtype=np.float32)
        y_score = np.concatenate(all_score[head]) if all_score[head] else np.zeros((0,), dtype=np.float32)
        report["aggregate"][head] = _metrics(y_true, y_score)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    logger.info(f"Wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
