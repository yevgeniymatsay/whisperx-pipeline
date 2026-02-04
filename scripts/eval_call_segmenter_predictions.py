#!/usr/bin/env python3
"""Evaluate call segmenter predictions against ground truth.

Computes segment-level metrics:
  - Time-level precision/recall/F1 (interval intersection)
  - Boundary MAE using IoU-based segment matching
  - Segment count delta

Usage:
    python scripts/eval_call_segmenter_predictions.py \
        --predictions s3://bucket/call_segmenter/predictions/v1/ \
        --ground-truth s3

Note: For parameter tuning, use sweep_call_segmenter.py instead.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import boto3

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import S3_BUCKET, AWS_REGION

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Segment:
    """A segment with start and end times."""
    start_s: float
    end_s: float


@dataclass
class TimeMetrics:
    """Time-level precision/recall/F1 metrics."""
    precision: float
    recall: float
    f1: float
    intersection_s: float
    pred_duration_s: float
    truth_duration_s: float


@dataclass
class BoundaryMetrics:
    """Boundary-level metrics using IoU matching."""
    matched_pairs: int
    unmatched_pred: int
    unmatched_truth: int
    mae_start_s: Optional[float]
    mae_end_s: Optional[float]
    mean_iou: Optional[float]


@dataclass
class VideoEvalResult:
    """Evaluation result for a single video."""
    video_id: str
    time_metrics: TimeMetrics
    boundary_metrics: BoundaryMetrics
    num_pred_segments: int
    num_truth_segments: int
    segment_count_delta: int


@dataclass
class AggregateResult:
    """Aggregate evaluation results."""
    num_videos: int
    micro_precision: float
    micro_recall: float
    micro_f1: float
    seg_precision: float
    seg_recall: float
    seg_f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    mean_mae_start_s: Optional[float]
    mean_mae_end_s: Optional[float]
    mean_iou: Optional[float]
    total_matched_pairs: int
    total_unmatched_pred: int
    total_unmatched_truth: int
    total_pred_segments: int
    total_truth_segments: int


# =============================================================================
# Data Loading
# =============================================================================


def get_s3_client():
    return boto3.client("s3", region_name=AWS_REGION)


def load_predictions_s3(s3_client, s3_prefix: str) -> Dict[str, List[Segment]]:
    """Load predictions from S3 prefix."""
    predictions: Dict[str, List[Segment]] = {}
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue

            # Extract video_id from key
            video_id = key.split("/")[-1].replace(".json", "")
            if not video_id:
                continue

            try:
                resp = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
                data = json.loads(resp["Body"].read())
                segments = [
                    Segment(start_s=float(s["start_s"]), end_s=float(s["end_s"]))
                    for s in data.get("segments", [])
                ]
                predictions[video_id] = segments
            except Exception as e:
                logger.warning(f"Error loading prediction {key}: {e}")

    return predictions


def load_predictions_local(local_dir: Path) -> Dict[str, List[Segment]]:
    """Load predictions from local directory."""
    predictions: Dict[str, List[Segment]] = {}

    for json_file in local_dir.glob("*.json"):
        video_id = json_file.stem
        try:
            data = json.loads(json_file.read_text())
            segments = [
                Segment(start_s=float(s["start_s"]), end_s=float(s["end_s"]))
                for s in data.get("segments", [])
            ]
            predictions[video_id] = segments
        except Exception as e:
            logger.warning(f"Error loading prediction {json_file}: {e}")

    return predictions


def load_ground_truth_s3(s3_client, prefix: str = "labeling/corrected_boundaries/v1/") -> Dict[str, List[Segment]]:
    """Load ground truth labels from S3."""
    labels: Dict[str, List[Segment]] = {}
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue

            video_id = key[len(prefix):-5]
            if not video_id:
                continue

            try:
                resp = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
                doc = json.loads(resp["Body"].read())
                boundaries = [
                    Segment(start_s=float(b["start_s"]), end_s=float(b["end_s"]))
                    for b in doc.get("boundaries", [])
                ]
                if boundaries:
                    labels[video_id] = boundaries
            except Exception as e:
                logger.warning(f"Error loading labels for {video_id}: {e}")

    return labels


def load_ground_truth_local(path: Path) -> Dict[str, List[Segment]]:
    """Load ground truth from local JSON file."""
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        labels: Dict[str, List[Segment]] = {}

        # Handle both formats: list of {video_id, calls: [...]} or direct {video_id: [...]}
        if isinstance(data, list):
            for item in data:
                video_id = item.get("video_id")
                calls = item.get("calls", [])
                labels[video_id] = [
                    Segment(start_s=float(c["start"]), end_s=float(c["end"]))
                    for c in calls
                ]
        else:
            for video_id, calls in data.items():
                labels[video_id] = [
                    Segment(start_s=float(c["start"]), end_s=float(c["end"]))
                    for c in calls
                ]
        return labels
    else:
        raise ValueError(f"Unsupported format: {path}")


# =============================================================================
# Metrics Computation
# =============================================================================


def compute_interval_intersection(
    segments_a: List[Segment],
    segments_b: List[Segment],
) -> float:
    """Compute total intersection duration between two sets of segments."""
    if not segments_a or not segments_b:
        return 0.0

    total_intersection = 0.0

    for a in segments_a:
        for b in segments_b:
            # Compute overlap
            overlap_start = max(a.start_s, b.start_s)
            overlap_end = min(a.end_s, b.end_s)
            if overlap_end > overlap_start:
                total_intersection += overlap_end - overlap_start

    return total_intersection


def compute_total_duration(segments: List[Segment]) -> float:
    """Compute total duration of segments (handles overlaps by merging)."""
    if not segments:
        return 0.0

    # Sort by start time
    sorted_segs = sorted(segments, key=lambda s: s.start_s)

    # Merge overlapping segments
    merged: List[Tuple[float, float]] = []
    cur_start, cur_end = sorted_segs[0].start_s, sorted_segs[0].end_s

    for seg in sorted_segs[1:]:
        if seg.start_s <= cur_end:
            cur_end = max(cur_end, seg.end_s)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = seg.start_s, seg.end_s

    merged.append((cur_start, cur_end))

    return sum(e - s for s, e in merged)


def compute_time_metrics(
    pred_segments: List[Segment],
    truth_segments: List[Segment],
) -> TimeMetrics:
    """Compute time-level precision/recall/F1."""
    intersection = compute_interval_intersection(pred_segments, truth_segments)
    pred_duration = compute_total_duration(pred_segments)
    truth_duration = compute_total_duration(truth_segments)

    if pred_duration > 0:
        precision = intersection / pred_duration
    else:
        precision = 1.0 if truth_duration == 0 else 0.0

    if truth_duration > 0:
        recall = intersection / truth_duration
    else:
        recall = 1.0 if pred_duration == 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return TimeMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        intersection_s=intersection,
        pred_duration_s=pred_duration,
        truth_duration_s=truth_duration,
    )


def compute_iou(seg_a: Segment, seg_b: Segment) -> float:
    """Compute IoU between two segments."""
    intersection_start = max(seg_a.start_s, seg_b.start_s)
    intersection_end = min(seg_a.end_s, seg_b.end_s)

    if intersection_end <= intersection_start:
        return 0.0

    intersection = intersection_end - intersection_start
    union = (seg_a.end_s - seg_a.start_s) + (seg_b.end_s - seg_b.start_s) - intersection

    return intersection / union if union > 0 else 0.0


def match_segments_by_iou(
    pred_segments: List[Segment],
    truth_segments: List[Segment],
    iou_threshold: float = 0.5,
) -> List[Tuple[int, int, float]]:
    """Match predicted segments to truth segments using IoU.

    Returns:
        List of (pred_idx, truth_idx, iou) for matched pairs
    """
    if not pred_segments or not truth_segments:
        return []

    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_segments), len(truth_segments)))
    for i, pred in enumerate(pred_segments):
        for j, truth in enumerate(truth_segments):
            iou_matrix[i, j] = compute_iou(pred, truth)

    # Greedy matching (assign best IoU pairs first)
    matched: List[Tuple[int, int, float]] = []
    used_pred = set()
    used_truth = set()

    while True:
        # Find max IoU among unmatched
        max_iou = 0.0
        max_i, max_j = -1, -1

        for i in range(len(pred_segments)):
            if i in used_pred:
                continue
            for j in range(len(truth_segments)):
                if j in used_truth:
                    continue
                if iou_matrix[i, j] > max_iou:
                    max_iou = iou_matrix[i, j]
                    max_i, max_j = i, j

        if max_iou < iou_threshold:
            break

        matched.append((max_i, max_j, max_iou))
        used_pred.add(max_i)
        used_truth.add(max_j)

    return matched


def compute_boundary_metrics(
    pred_segments: List[Segment],
    truth_segments: List[Segment],
    iou_threshold: float = 0.5,
) -> BoundaryMetrics:
    """Compute boundary-level metrics using IoU matching."""
    matches = match_segments_by_iou(pred_segments, truth_segments, iou_threshold)

    matched_pairs = len(matches)
    unmatched_pred = len(pred_segments) - matched_pairs
    unmatched_truth = len(truth_segments) - matched_pairs

    if matched_pairs == 0:
        return BoundaryMetrics(
            matched_pairs=0,
            unmatched_pred=unmatched_pred,
            unmatched_truth=unmatched_truth,
            mae_start_s=None,
            mae_end_s=None,
            mean_iou=None,
        )

    # Compute MAE on matched pairs
    start_errors: List[float] = []
    end_errors: List[float] = []
    ious: List[float] = []

    for pred_idx, truth_idx, iou in matches:
        pred = pred_segments[pred_idx]
        truth = truth_segments[truth_idx]
        start_errors.append(abs(pred.start_s - truth.start_s))
        end_errors.append(abs(pred.end_s - truth.end_s))
        ious.append(iou)

    return BoundaryMetrics(
        matched_pairs=matched_pairs,
        unmatched_pred=unmatched_pred,
        unmatched_truth=unmatched_truth,
        mae_start_s=float(np.mean(start_errors)),
        mae_end_s=float(np.mean(end_errors)),
        mean_iou=float(np.mean(ious)),
    )


def evaluate_video(
    video_id: str,
    pred_segments: List[Segment],
    truth_segments: List[Segment],
    iou_threshold: float = 0.5,
) -> VideoEvalResult:
    """Evaluate predictions for a single video."""
    time_metrics = compute_time_metrics(pred_segments, truth_segments)
    boundary_metrics = compute_boundary_metrics(pred_segments, truth_segments, iou_threshold)

    return VideoEvalResult(
        video_id=video_id,
        time_metrics=time_metrics,
        boundary_metrics=boundary_metrics,
        num_pred_segments=len(pred_segments),
        num_truth_segments=len(truth_segments),
        segment_count_delta=len(pred_segments) - len(truth_segments),
    )


def aggregate_results(per_video: List[VideoEvalResult]) -> AggregateResult:
    """Aggregate per-video results into summary metrics."""
    if not per_video:
        return AggregateResult(
            num_videos=0,
            micro_precision=0.0, micro_recall=0.0, micro_f1=0.0,
            seg_precision=0.0, seg_recall=0.0, seg_f1=0.0,
            macro_precision=0.0, macro_recall=0.0, macro_f1=0.0,
            mean_mae_start_s=None, mean_mae_end_s=None, mean_iou=None,
            total_matched_pairs=0, total_unmatched_pred=0, total_unmatched_truth=0,
            total_pred_segments=0, total_truth_segments=0,
        )

    # Micro-average (sum intersections / sum durations)
    total_intersection = sum(r.time_metrics.intersection_s for r in per_video)
    total_pred_dur = sum(r.time_metrics.pred_duration_s for r in per_video)
    total_truth_dur = sum(r.time_metrics.truth_duration_s for r in per_video)

    micro_precision = total_intersection / total_pred_dur if total_pred_dur > 0 else 0.0
    micro_recall = total_intersection / total_truth_dur if total_truth_dur > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    # Macro-average (average of per-video metrics)
    macro_precision = float(np.mean([r.time_metrics.precision for r in per_video]))
    macro_recall = float(np.mean([r.time_metrics.recall for r in per_video]))
    macro_f1 = float(np.mean([r.time_metrics.f1 for r in per_video]))

    # Boundary metrics
    total_matched = sum(r.boundary_metrics.matched_pairs for r in per_video)
    total_unmatched_pred = sum(r.boundary_metrics.unmatched_pred for r in per_video)
    total_unmatched_truth = sum(r.boundary_metrics.unmatched_truth for r in per_video)

    mae_starts = [r.boundary_metrics.mae_start_s for r in per_video if r.boundary_metrics.mae_start_s is not None]
    mae_ends = [r.boundary_metrics.mae_end_s for r in per_video if r.boundary_metrics.mae_end_s is not None]
    ious = [r.boundary_metrics.mean_iou for r in per_video if r.boundary_metrics.mean_iou is not None]

    # Segment-level metrics from IoU matching (micro)
    total_pred_segments = sum(r.num_pred_segments for r in per_video)
    total_truth_segments = sum(r.num_truth_segments for r in per_video)
    seg_precision = total_matched / total_pred_segments if total_pred_segments > 0 else 0.0
    seg_recall = total_matched / total_truth_segments if total_truth_segments > 0 else 0.0
    seg_f1 = 2 * seg_precision * seg_recall / (seg_precision + seg_recall) if (seg_precision + seg_recall) > 0 else 0.0

    return AggregateResult(
        num_videos=len(per_video),
        micro_precision=micro_precision,
        micro_recall=micro_recall,
        micro_f1=micro_f1,
        seg_precision=seg_precision,
        seg_recall=seg_recall,
        seg_f1=seg_f1,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        mean_mae_start_s=float(np.mean(mae_starts)) if mae_starts else None,
        mean_mae_end_s=float(np.mean(mae_ends)) if mae_ends else None,
        mean_iou=float(np.mean(ious)) if ious else None,
        total_matched_pairs=total_matched,
        total_unmatched_pred=total_unmatched_pred,
        total_unmatched_truth=total_unmatched_truth,
        total_pred_segments=total_pred_segments,
        total_truth_segments=total_truth_segments,
    )


# =============================================================================
# Output Formatting
# =============================================================================


def print_report(per_video: List[VideoEvalResult], aggregate: AggregateResult) -> None:
    """Print formatted evaluation report."""
    print("\n" + "=" * 100)
    print("Per-Video Results")
    print("=" * 100)
    print(f"{'Video':<20} {'Time-P':>8} {'Time-R':>8} {'Time-F1':>8} {'MAE-S':>8} {'MAE-E':>8} {'IoU':>6} {'#Pred':>6} {'#True':>6} {'Delta':>6}")
    print("-" * 100)

    # Sort by F1
    sorted_results = sorted(per_video, key=lambda r: r.time_metrics.f1)

    for r in sorted_results:
        mae_s = f"{r.boundary_metrics.mae_start_s:.2f}s" if r.boundary_metrics.mae_start_s is not None else "N/A"
        mae_e = f"{r.boundary_metrics.mae_end_s:.2f}s" if r.boundary_metrics.mae_end_s is not None else "N/A"
        iou = f"{r.boundary_metrics.mean_iou:.3f}" if r.boundary_metrics.mean_iou is not None else "N/A"
        delta = f"{r.segment_count_delta:+d}"

        print(f"{r.video_id:<20} {r.time_metrics.precision:>8.3f} {r.time_metrics.recall:>8.3f} {r.time_metrics.f1:>8.3f} "
              f"{mae_s:>8} {mae_e:>8} {iou:>6} {r.num_pred_segments:>6} {r.num_truth_segments:>6} {delta:>6}")

    print("-" * 100)

    # Aggregate
    print(f"\n{'Aggregate Results':^100}")
    print("-" * 100)
    print(f"  Videos evaluated: {aggregate.num_videos}")
    print(f"  Total segments: pred={aggregate.total_pred_segments}, truth={aggregate.total_truth_segments}")
    print(f"  Matched pairs: {aggregate.total_matched_pairs}, unmatched pred: {aggregate.total_unmatched_pred}, unmatched truth: {aggregate.total_unmatched_truth}")
    print()
    print(f"  Time-level (micro):  P={aggregate.micro_precision:.4f}  R={aggregate.micro_recall:.4f}  F1={aggregate.micro_f1:.4f}")
    print(f"  Time-level (macro):  P={aggregate.macro_precision:.4f}  R={aggregate.macro_recall:.4f}  F1={aggregate.macro_f1:.4f}")
    print(f"  Segment-level (IoU): P={aggregate.seg_precision:.4f}  R={aggregate.seg_recall:.4f}  F1={aggregate.seg_f1:.4f}")
    if aggregate.mean_mae_start_s is not None:
        print(f"  Boundary MAE:        start={aggregate.mean_mae_start_s:.2f}s  end={aggregate.mean_mae_end_s:.2f}s")
    if aggregate.mean_iou is not None:
        print(f"  Mean IoU (matched):  {aggregate.mean_iou:.4f}")
    print("=" * 100)


def save_report_json(
    per_video: List[VideoEvalResult],
    aggregate: AggregateResult,
    output_path: Path,
) -> None:
    """Save evaluation report to JSON."""
    report = {
        "aggregate": {
            "num_videos": aggregate.num_videos,
            "micro": {
                "precision": aggregate.micro_precision,
                "recall": aggregate.micro_recall,
                "f1": aggregate.micro_f1,
            },
            "segment": {
                "precision": aggregate.seg_precision,
                "recall": aggregate.seg_recall,
                "f1": aggregate.seg_f1,
            },
            "macro": {
                "precision": aggregate.macro_precision,
                "recall": aggregate.macro_recall,
                "f1": aggregate.macro_f1,
            },
            "boundary": {
                "mean_mae_start_s": aggregate.mean_mae_start_s,
                "mean_mae_end_s": aggregate.mean_mae_end_s,
                "mean_iou": aggregate.mean_iou,
                "total_matched_pairs": aggregate.total_matched_pairs,
                "total_unmatched_pred": aggregate.total_unmatched_pred,
                "total_unmatched_truth": aggregate.total_unmatched_truth,
            },
            "segment_counts": {
                "total_pred": aggregate.total_pred_segments,
                "total_truth": aggregate.total_truth_segments,
            },
        },
        "per_video": [
            {
                "video_id": r.video_id,
                "time_precision": r.time_metrics.precision,
                "time_recall": r.time_metrics.recall,
                "time_f1": r.time_metrics.f1,
                "mae_start_s": r.boundary_metrics.mae_start_s,
                "mae_end_s": r.boundary_metrics.mae_end_s,
                "mean_iou": r.boundary_metrics.mean_iou,
                "matched_pairs": r.boundary_metrics.matched_pairs,
                "num_pred_segments": r.num_pred_segments,
                "num_truth_segments": r.num_truth_segments,
                "segment_count_delta": r.segment_count_delta,
            }
            for r in per_video
        ],
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved report to {output_path}")




# =============================================================================
# Main
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate call segmenter predictions against ground truth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions", type=str, required=True,
                        help="S3 prefix (s3://...) or local directory with prediction JSONs")
    parser.add_argument("--ground-truth", type=str, default="s3",
                        help="'s3' for S3 labels, or path to local labels file")
    parser.add_argument("--ground-truth-prefix", type=str, default="labeling/corrected_boundaries/v1/",
                        help="S3 prefix for ground truth labels (if --ground-truth=s3)")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for segment matching")
    parser.add_argument("--video-list", type=Path, default=None,
                        help="Optional file of video IDs (one per line) to evaluate")
    parser.add_argument("--video-ids", type=str, default=None,
                        help="Optional comma-separated list of video IDs to evaluate")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path for JSON report")

    args = parser.parse_args()

    s3 = get_s3_client()

    # Load predictions
    if args.predictions.startswith("s3://"):
        # Parse s3://bucket/prefix format
        parts = args.predictions[5:].split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        if bucket != S3_BUCKET:
            logger.warning(f"Predictions bucket {bucket} differs from default {S3_BUCKET}")
        predictions = load_predictions_s3(s3, prefix)
    else:
        predictions = load_predictions_local(Path(args.predictions))

    logger.info(f"Loaded predictions for {len(predictions)} videos")

    # Load ground truth
    if args.ground_truth == "s3":
        ground_truth = load_ground_truth_s3(s3, args.ground_truth_prefix)
    else:
        ground_truth = load_ground_truth_local(Path(args.ground_truth))

    logger.info(f"Loaded ground truth for {len(ground_truth)} videos")

    # Find common videos
    common_videos = set(predictions.keys()) & set(ground_truth.keys())
    if not common_videos:
        logger.error("No common videos between predictions and ground truth!")
        return 1

    # Apply user filtering
    if args.video_list:
        if not args.video_list.exists():
            logger.error(f"Video list not found: {args.video_list}")
            return 1
        requested = {ln.strip() for ln in args.video_list.read_text().splitlines() if ln.strip() and not ln.strip().startswith('#')}
        common_videos &= requested
    if args.video_ids:
        requested = {v.strip() for v in args.video_ids.split(",") if v.strip()}
        common_videos &= requested

    if not common_videos:
        logger.error("No videos left after applying --video-list/--video-ids filter")
        return 1

    pred_only = set(predictions.keys()) - common_videos
    truth_only = set(ground_truth.keys()) - common_videos

    if pred_only:
        logger.warning(f"Videos in predictions only: {len(pred_only)}")
    if truth_only:
        logger.warning(f"Videos in ground truth only: {len(truth_only)}")

    logger.info(f"Evaluating {len(common_videos)} common videos")

    # Evaluate each video
    per_video_results: List[VideoEvalResult] = []
    for video_id in sorted(common_videos):
        result = evaluate_video(
            video_id=video_id,
            pred_segments=predictions[video_id],
            truth_segments=ground_truth[video_id],
            iou_threshold=args.iou_threshold,
        )
        per_video_results.append(result)

    # Aggregate results
    aggregate = aggregate_results(per_video_results)

    # Print report
    print_report(per_video_results, aggregate)

    # Save JSON report
    if args.output:
        save_report_json(per_video_results, aggregate, args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
