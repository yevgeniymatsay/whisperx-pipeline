#!/usr/bin/env python3
"""Build Call Segmenter Training Dataset.

Pulls chunks from S3, aligns diarization in absolute time, and outputs
windowed rows with diarization features + labels for ML training.

Usage:
    python scripts/build_call_segmenter_dataset.py \
        --labels labels.json \
        --output-dir data/call_segmenter/v1 \
        --win-s 1.0 \
        --hop-s 0.5 \
        --ignore-s 0.75 \
        --exclude-drift-above 2.0
"""

import argparse
import csv
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

import boto3
import pandas as pd

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import S3_BUCKET, AWS_REGION
from pipeline.transcriber import DiarizationSegment
from pipeline.call_segmenter.host_detector import (
    detect_host_speaker,
    HostDetectionResult,
)
from pipeline.call_segmenter.features import merge_adjacent_segments
from pipeline.call_segmenter.window_generator import (
    WindowConfig,
    CallBoundary,
    generate_windows,
    compute_call_sanity_stats,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ChunkMetadata:
    """Metadata for a single chunk from S3."""
    chunk_id: str
    start_abs: float
    end_abs: float
    duration_s: float
    content_type: str  # "narration_only" or "call_like"
    speaker_count: int


@dataclass
class VideoData:
    """All data loaded for a single video."""
    video_id: str
    run_id: str
    mp3_key: str
    creator_id: str
    chunks: List[ChunkMetadata]
    segments: List[DiarizationSegment]
    call_boundaries: List[CallBoundary]
    timeline_end: float
    host_result: Optional[HostDetectionResult] = None


@dataclass
class DatasetConfig:
    """Configuration for dataset building."""
    win_s: float = 1.0
    hop_s: float = 0.5
    ignore_s: float = 0.75
    context_10s: float = 10.0
    context_30s: float = 30.0
    host_bin_s: float = 30.0
    drift_threshold_s: float = 2.0
    merge_gap_s: float = 0.2
    ambiguity_threshold: float = 0.15
    low_nonhost_threshold: float = 0.05


@dataclass
class VideoReport:
    """Report data for a single video."""
    video_id: str
    run_id: str
    creator_id: str
    mp3_key: str
    windows: int
    calls: int
    class_balance_in_call: float
    ignored_windows: int
    host_speakers: List[str]
    host_confidence: float
    is_host_ambiguous: bool
    timebase_drift_s: Optional[float]
    chunk_count: int
    chunk_gaps: List[Dict[str, float]]
    call_sanity: List[Dict]
    errors: List[str] = field(default_factory=list)


@dataclass
class DatasetReport:
    """Full report for the dataset build."""
    total_videos: int
    total_windows: int
    class_balance: Dict[str, float]
    ignored_windows: int
    videos: Dict[str, Dict]
    anomalies: Dict[str, List[str]]
    config: Dict


# =============================================================================
# S3 Loading Functions
# =============================================================================


def get_s3_client():
    """Get boto3 S3 client."""
    return boto3.client("s3", region_name=AWS_REGION)


def load_latest_run_id(s3_client, video_id: str) -> Optional[str]:
    """Load run_id from latest/{video_id}.json pointer."""
    key = f"latest/{video_id}.json"
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        data = json.loads(response["Body"].read())
        return data.get("run_id")
    except s3_client.exceptions.NoSuchKey:
        logger.warning(f"No latest pointer found for video {video_id}")
        return None
    except Exception as e:
        logger.error(f"Error loading latest pointer for {video_id}: {e}")
        return None


def load_video_manifest(s3_client, video_id: str, run_id: str) -> Optional[Dict]:
    """Load video manifest from S3."""
    key = f"runs/{video_id}/{run_id}/video_manifest.json"
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        return json.loads(response["Body"].read())
    except Exception as e:
        logger.error(f"Error loading manifest for {video_id}/{run_id}: {e}")
        return None


def list_chunks(s3_client, video_id: str, run_id: str) -> List[str]:
    """List all chunk IDs for a video/run."""
    prefix = f"runs/{video_id}/{run_id}/chunks/"
    chunk_ids = set()

    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            # Extract chunk_id from path like runs/.../chunks/{chunk_id}/...
            parts = obj["Key"].split("/")
            if len(parts) >= 5:
                chunk_ids.add(parts[4])

    return sorted(chunk_ids)


def load_chunk_metadata(
    s3_client, video_id: str, run_id: str, chunk_id: str
) -> Optional[ChunkMetadata]:
    """Load chunk_metadata.json for a chunk."""
    key = f"runs/{video_id}/{run_id}/chunks/{chunk_id}/chunk_metadata.json"
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        data = json.loads(response["Body"].read())
        return ChunkMetadata(
            chunk_id=chunk_id,
            start_abs=data["chunk_time_offset_s"],
            end_abs=data["chunk_time_offset_s"] + data["duration_s"],
            duration_s=data["duration_s"],
            content_type=data.get("content_type", "unknown"),
            speaker_count=data.get("speaker_count", 0),
        )
    except Exception as e:
        logger.error(f"Error loading chunk metadata {key}: {e}")
        return None


def load_diarization_segments(
    s3_client, video_id: str, run_id: str, chunk_id: str
) -> List[DiarizationSegment]:
    """Load diarization_segments.json with absolute timestamps.

    The pipeline's WhisperXTranscriber already adds chunk_time_offset_s when
    creating segments, so stored JSON has absolute timestamps (t0_abs, t1_abs).
    """
    key = f"runs/{video_id}/{run_id}/chunks/{chunk_id}/diarization_segments.json"
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        data = json.loads(response["Body"].read())
        segments = []
        for seg in data.get("segments", []):
            # Timestamps are already absolute - no adjustment needed
            segments.append(DiarizationSegment(
                t0_abs=seg["t0_abs"],
                t1_abs=seg["t1_abs"],
                spk=seg["spk"],
            ))
        return segments
    except s3_client.exceptions.NoSuchKey:
        logger.warning(f"No diarization segments for {chunk_id}")
        return []
    except Exception as e:
        logger.error(f"Error loading diarization {key}: {e}")
        return []


def extract_creator_id(mp3_key: str) -> str:
    """Extract creator_id from mp3_key.

    Pattern: audio/pretraining/{creator} - {video_id}.mp3
    """
    # Get filename from key
    filename = mp3_key.split("/")[-1]

    # Try to extract creator before " - "
    match = re.match(r"(.+?) - ", filename)
    if match:
        return match.group(1).strip()

    # Fallback: use full filename without extension
    return filename.replace(".mp3", "")


def get_mp3_key_for_video(s3_client, video_id: str) -> Optional[str]:
    """Find the mp3 key for a video by listing audio/pretraining/."""
    prefix = "audio/pretraining/"
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if video_id in key and key.endswith(".mp3"):
                return key

    return None


def get_mp3_duration_s(s3_client, mp3_key: str) -> Optional[float]:
    """Get MP3 duration using HEAD request and ffprobe if available.

    Returns None if duration cannot be determined.
    """
    # Try to get duration from S3 metadata (if stored)
    try:
        head = s3_client.head_object(Bucket=S3_BUCKET, Key=mp3_key)
        if "x-amz-meta-duration-s" in head.get("Metadata", {}):
            return float(head["Metadata"]["x-amz-meta-duration-s"])
    except Exception:
        pass

    # We can't easily get duration without downloading - return None
    # The caller can use chunk end times as approximation
    return None


# =============================================================================
# Label Loading
# =============================================================================


def load_labels_json(path: Path) -> Dict[str, List[CallBoundary]]:
    """Load labels from labels.json format.

    Format: [{"video_id": "...", "calls": [{"start": ..., "end": ...}]}]
    """
    with open(path) as f:
        data = json.load(f)

    labels = {}
    for video in data:
        video_id = video["video_id"]
        boundaries = [
            CallBoundary(start=c["start"], end=c["end"])
            for c in video.get("calls", [])
        ]
        labels[video_id] = boundaries

    return labels


def load_labels_csv(path: Path) -> Dict[str, List[CallBoundary]]:
    """Load labels from corrected_boundaries.csv format.

    Format: video_id,call_index,start_s,end_s,corrected_at
    """
    labels: Dict[str, List[CallBoundary]] = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row["video_id"]
            boundary = CallBoundary(
                start=float(row["start_s"]),
                end=float(row["end_s"]),
            )
            if video_id not in labels:
                labels[video_id] = []
            labels[video_id].append(boundary)

    # Sort boundaries by start time for each video
    for video_id in labels:
        labels[video_id].sort(key=lambda b: b.start)

    return labels


def load_labels(path: Path) -> Dict[str, List[CallBoundary]]:
    """Load labels from either JSON or CSV format."""
    if path.suffix == ".json":
        return load_labels_json(path)
    elif path.suffix == ".csv":
        return load_labels_csv(path)
    else:
        raise ValueError(f"Unsupported label format: {path.suffix}")


def validate_labels(labels: Dict[str, List[CallBoundary]]) -> List[str]:
    """Validate labels and return list of issues."""
    issues = []

    for video_id, boundaries in labels.items():
        for i, b in enumerate(boundaries):
            # Check start < end
            if b.start >= b.end:
                issues.append(
                    f"{video_id}: call {i} has start >= end ({b.start} >= {b.end})"
                )

            # Check for overlaps with next boundary
            if i < len(boundaries) - 1:
                next_b = boundaries[i + 1]
                if b.end > next_b.start:
                    issues.append(
                        f"{video_id}: calls {i} and {i+1} overlap "
                        f"({b.end} > {next_b.start})"
                    )

    return issues


# =============================================================================
# Core Processing
# =============================================================================


def load_video_data(
    s3_client,
    video_id: str,
    labels: Dict[str, List[CallBoundary]],
    config: DatasetConfig,
) -> Optional[VideoData]:
    """Load all data for a video from S3."""
    # Step 2: Resolve run_id
    run_id = load_latest_run_id(s3_client, video_id)
    if not run_id:
        logger.error(f"Could not resolve run_id for {video_id}")
        return None

    # Get mp3_key and creator_id
    mp3_key = get_mp3_key_for_video(s3_client, video_id)
    if not mp3_key:
        logger.warning(f"Could not find mp3 for {video_id}, using fallback creator_id")
        mp3_key = f"audio/pretraining/unknown - {video_id}.mp3"
    creator_id = extract_creator_id(mp3_key)

    # Step 3: Load and order chunks
    chunk_ids = list_chunks(s3_client, video_id, run_id)
    if not chunk_ids:
        logger.error(f"No chunks found for {video_id}/{run_id}")
        return None

    chunks = []
    for chunk_id in chunk_ids:
        meta = load_chunk_metadata(s3_client, video_id, run_id, chunk_id)
        if meta:
            chunks.append(meta)

    # Sort by start_abs
    chunks.sort(key=lambda c: c.start_abs)

    # Step 4: Load and merge diarization
    all_segments = []
    for chunk in chunks:
        segs = load_diarization_segments(
            s3_client, video_id, run_id, chunk.chunk_id
        )
        all_segments.extend(segs)

    # Sort by time and optionally merge adjacent segments
    all_segments.sort(key=lambda s: s.t0_abs)
    if config.merge_gap_s > 0:
        all_segments = merge_adjacent_segments(all_segments, config.merge_gap_s)

    # Get timeline end from chunks
    timeline_end = max(c.end_abs for c in chunks) if chunks else 0.0

    # Get call boundaries for this video
    call_boundaries = labels.get(video_id, [])

    return VideoData(
        video_id=video_id,
        run_id=run_id,
        mp3_key=mp3_key,
        creator_id=creator_id,
        chunks=chunks,
        segments=all_segments,
        call_boundaries=call_boundaries,
        timeline_end=timeline_end,
    )


def check_chunk_ordering(chunks: List[ChunkMetadata]) -> List[Dict[str, float]]:
    """Check for gaps or overlaps between chunks."""
    gaps = []
    for i in range(len(chunks) - 1):
        curr = chunks[i]
        next_c = chunks[i + 1]
        gap = next_c.start_abs - curr.end_abs
        if abs(gap) > 0.1:  # Report gaps/overlaps > 100ms
            gaps.append({
                "after_chunk": curr.chunk_id,
                "before_chunk": next_c.chunk_id,
                "gap_s": round(gap, 3),
            })
    return gaps


def process_video(
    video_data: VideoData,
    config: DatasetConfig,
) -> Tuple[pd.DataFrame, VideoReport]:
    """Process a video and generate windows + report."""
    errors = []

    # Step 5: Identify host speaker
    host_result = detect_host_speaker(
        segments=video_data.segments,
        timeline_end=video_data.timeline_end,
        bin_size_s=config.host_bin_s,
        ambiguity_threshold=config.ambiguity_threshold,
    )
    video_data.host_result = host_result

    # Step 6 & 7: Generate windows with features
    window_config = WindowConfig(
        win_s=config.win_s,
        hop_s=config.hop_s,
        ignore_s=config.ignore_s,
        context_10s=config.context_10s,
        context_30s=config.context_30s,
    )

    windows = generate_windows(
        segments=video_data.segments,
        host_speaker=host_result.host_speaker,
        call_boundaries=video_data.call_boundaries,
        timeline_start=0.0,
        timeline_end=video_data.timeline_end,
        config=window_config,
    )

    # Convert to DataFrame rows
    rows = []
    for window in windows:
        row = window.to_dict()
        row["video_id"] = video_data.video_id
        row["run_id"] = video_data.run_id
        row["creator_id"] = video_data.creator_id
        row["mp3_key"] = video_data.mp3_key
        rows.append(row)

    df = pd.DataFrame(rows)

    # Add chunk_content_type (for debugging - leaky!)
    # Find which chunk each window's t_mid falls into
    chunk_content_types = []
    for _, row in df.iterrows():
        t_mid = row["t_mid"]
        content_type = "unknown"
        for chunk in video_data.chunks:
            if chunk.start_abs <= t_mid < chunk.end_abs:
                content_type = chunk.content_type
                break
        chunk_content_types.append(content_type)
    df["chunk_content_type"] = chunk_content_types

    # Check chunk ordering
    chunk_gaps = check_chunk_ordering(video_data.chunks)

    # Compute call sanity stats
    call_sanity = []
    for boundary in video_data.call_boundaries:
        stats = compute_call_sanity_stats(
            video_data.segments, host_result.host_speaker, boundary
        )
        call_sanity.append(stats)

    # Compute report stats
    in_call_count = (df["y"] == 1).sum()
    total_count = len(df)
    ignored_count = (df["ignore"] == 1).sum()

    report = VideoReport(
        video_id=video_data.video_id,
        run_id=video_data.run_id,
        creator_id=video_data.creator_id,
        mp3_key=video_data.mp3_key,
        windows=total_count,
        calls=len(video_data.call_boundaries),
        class_balance_in_call=in_call_count / total_count if total_count > 0 else 0,
        ignored_windows=ignored_count,
        host_speakers=[s.speaker_id for s in host_result.top_speakers],
        host_confidence=host_result.host_confidence,
        is_host_ambiguous=host_result.is_ambiguous,
        timebase_drift_s=None,  # Computed later if mp3 duration available
        chunk_count=len(video_data.chunks),
        chunk_gaps=chunk_gaps,
        call_sanity=call_sanity,
        errors=errors,
    )

    return df, report


# =============================================================================
# Output Functions
# =============================================================================


def get_git_sha() -> str:
    """Get current git SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        return result.stdout.strip()[:12]
    except Exception:
        return "unknown"


def build_dataset_metadata(config: DatasetConfig) -> Dict:
    """Build dataset metadata document."""
    return {
        "version": "v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_sha(),
        "config": {
            "win_s": config.win_s,
            "hop_s": config.hop_s,
            "ignore_s": config.ignore_s,
            "context_10s": config.context_10s,
            "context_30s": config.context_30s,
            "host_bin_s": config.host_bin_s,
            "drift_threshold_s": config.drift_threshold_s,
        },
        "non_feature_columns": [
            "video_id",
            "run_id",
            "creator_id",
            "mp3_key",
            "t_start",
            "t_end",
            "t_mid",
            "chunk_content_type",
            "y",
            "ignore",
            "dist_to_boundary_s",
        ],
        "feature_columns": [
            "speech_frac",
            "host_speech_frac",
            "nonhost_speech_frac",
            "num_active_speakers",
            "nonhost_active",
            "speaker_switches_10s",
            "unique_nonhost_speakers_30s",
            "nonhost_turns_30s",
            "avg_nonhost_turn_len_30s",
        ],
        "group_split_column": "creator_id",
    }


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def build_report(
    video_reports: Dict[str, VideoReport],
    config: DatasetConfig,
) -> Dict:
    """Build the full dataset report."""
    total_windows = sum(r.windows for r in video_reports.values())
    total_ignored = sum(r.ignored_windows for r in video_reports.values())
    total_in_call = sum(
        r.windows * r.class_balance_in_call for r in video_reports.values()
    )

    # Identify anomalies
    anomalies = {
        "timebase_drift": [],
        "ambiguous_host": [],
        "low_nonhost_calls": [],
        "huge_chunk_gaps": [],
    }

    for video_id, report in video_reports.items():
        # Timebase drift
        if report.timebase_drift_s and report.timebase_drift_s > config.drift_threshold_s:
            anomalies["timebase_drift"].append(video_id)

        # Ambiguous host
        if report.is_host_ambiguous:
            anomalies["ambiguous_host"].append(video_id)

        # Low non-host calls
        for call_stat in report.call_sanity:
            if call_stat["nonhost_frac"] < config.low_nonhost_threshold:
                anomalies["low_nonhost_calls"].append(
                    f"{video_id}:{call_stat['start']:.1f}-{call_stat['end']:.1f}"
                )

        # Huge chunk gaps (> 5s)
        for gap in report.chunk_gaps:
            if abs(gap["gap_s"]) > 5.0:
                anomalies["huge_chunk_gaps"].append(
                    f"{video_id}:{gap['after_chunk']}->{gap['before_chunk']} ({gap['gap_s']:.1f}s)"
                )

    return {
        "total_videos": len(video_reports),
        "total_windows": total_windows,
        "class_balance": {
            "in_call": round(total_in_call / total_windows, 4) if total_windows else 0,
            "out_of_call": round(1 - total_in_call / total_windows, 4) if total_windows else 0,
        },
        "ignored_windows": total_ignored,
        "videos": {
            vid: {
                "windows": r.windows,
                "calls": r.calls,
                "host_speakers": r.host_speakers,
                "host_confidence": round(r.host_confidence, 3),
                "timebase_drift_s": r.timebase_drift_s,
                "chunk_gaps": r.chunk_gaps,
                "call_sanity": r.call_sanity,
            }
            for vid, r in video_reports.items()
        },
        "anomalies": anomalies,
    }


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Build call segmenter training dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to labels file (labels.json or corrected_boundaries.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/call_segmenter/v1"),
        help="Output directory for dataset",
    )
    parser.add_argument("--win-s", type=float, default=1.0, help="Window size in seconds")
    parser.add_argument("--hop-s", type=float, default=0.5, help="Hop size in seconds")
    parser.add_argument(
        "--ignore-s", type=float, default=0.75, help="Ignore zone around boundaries"
    )
    parser.add_argument(
        "--exclude-drift-above",
        type=float,
        default=None,
        help="Exclude videos with timebase drift above threshold",
    )
    parser.add_argument(
        "--video-ids",
        type=str,
        nargs="*",
        help="Process only these video IDs (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data but don't write outputs",
    )

    args = parser.parse_args()

    # Build config
    config = DatasetConfig(
        win_s=args.win_s,
        hop_s=args.hop_s,
        ignore_s=args.ignore_s,
        drift_threshold_s=args.exclude_drift_above or 2.0,
    )

    # Step 1: Load labels
    logger.info(f"Loading labels from {args.labels}")
    labels = load_labels(args.labels)
    logger.info(f"Loaded labels for {len(labels)} videos")

    # Validate labels
    label_issues = validate_labels(labels)
    if label_issues:
        logger.warning(f"Found {len(label_issues)} label issues:")
        for issue in label_issues[:10]:
            logger.warning(f"  {issue}")

    # Filter to specific videos if requested
    video_ids = args.video_ids if args.video_ids else list(labels.keys())
    logger.info(f"Processing {len(video_ids)} videos")

    # Initialize S3 client
    s3_client = get_s3_client()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process videos
    all_dfs = []
    video_reports: Dict[str, VideoReport] = {}
    index_entries = []

    for i, video_id in enumerate(video_ids):
        logger.info(f"[{i+1}/{len(video_ids)}] Processing {video_id}")

        # Load video data
        video_data = load_video_data(s3_client, video_id, labels, config)
        if not video_data:
            logger.error(f"  Failed to load data for {video_id}")
            continue

        logger.info(
            f"  Loaded {len(video_data.chunks)} chunks, "
            f"{len(video_data.segments)} segments, "
            f"{len(video_data.call_boundaries)} labeled calls"
        )

        # Process video
        df, report = process_video(video_data, config)
        logger.info(
            f"  Generated {len(df)} windows, "
            f"{report.class_balance_in_call:.1%} in-call, "
            f"host={report.host_speakers[0] if report.host_speakers else 'unknown'} "
            f"(conf={report.host_confidence:.2f})"
        )

        # Store results
        all_dfs.append(df)
        video_reports[video_id] = report

        # Add to index
        index_entries.append({
            "video_id": video_id,
            "run_id": video_data.run_id,
            "creator_id": video_data.creator_id,
            "mp3_key": video_data.mp3_key,
            "windows": len(df),
            "calls": len(video_data.call_boundaries),
        })

        # Write per-video parquet
        if not args.dry_run:
            video_parquet = args.output_dir / f"windows_{video_id}.parquet"
            df.to_parquet(video_parquet, index=False)
            logger.info(f"  Wrote {video_parquet}")

    # Combine all DataFrames
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Combined dataset: {len(combined_df)} windows")
    else:
        logger.error("No data to write!")
        return 1

    # Filter by drift if requested
    if args.exclude_drift_above:
        excluded = [
            vid for vid, r in video_reports.items()
            if r.timebase_drift_s and r.timebase_drift_s > args.exclude_drift_above
        ]
        if excluded:
            logger.info(f"Excluding {len(excluded)} videos with drift > {args.exclude_drift_above}s")
            combined_df = combined_df[~combined_df["video_id"].isin(excluded)]

    # Write outputs
    if not args.dry_run:
        # Combined parquet
        combined_path = args.output_dir / "windows_all.parquet"
        combined_df.to_parquet(combined_path, index=False)
        logger.info(f"Wrote {combined_path}")

        # Index
        index_path = args.output_dir / "index.jsonl"
        with open(index_path, "w") as f:
            for entry in index_entries:
                f.write(json.dumps(entry) + "\n")
        logger.info(f"Wrote {index_path}")

        # Report
        report = build_report(video_reports, config)
        report_path = args.output_dir / "report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        logger.info(f"Wrote {report_path}")

        # Metadata
        meta = build_dataset_metadata(config)
        meta_path = args.output_dir / "dataset_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, cls=NumpyEncoder)
        logger.info(f"Wrote {meta_path}")

    # Summary
    logger.info("=" * 60)
    logger.info("Dataset Build Complete")
    logger.info(f"  Total videos: {len(video_reports)}")
    logger.info(f"  Total windows: {len(combined_df)}")
    in_call = (combined_df["y"] == 1).sum()
    logger.info(f"  Class balance: {in_call/len(combined_df):.1%} in-call")
    ignored = (combined_df["ignore"] == 1).sum()
    logger.info(f"  Ignored windows: {ignored} ({ignored/len(combined_df):.1%})")

    # Report anomalies
    report_data = build_report(video_reports, config)
    for anomaly_type, items in report_data["anomalies"].items():
        if items:
            logger.warning(f"  {anomaly_type}: {len(items)} items")

    return 0


if __name__ == "__main__":
    sys.exit(main())
