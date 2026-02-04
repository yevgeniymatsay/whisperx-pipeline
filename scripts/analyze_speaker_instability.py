#!/usr/bin/env python3
"""Analyze Speaker ID Instability Across Chunks.

Computes metrics showing how pyannote speaker labels reset at chunk boundaries,
causing the same person to get different SPEAKER_XX labels in different chunks.

Usage:
    python scripts/analyze_speaker_instability.py

Output:
    data/call_segmenter/analysis/speaker_id_instability.csv
"""

import csv
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import S3_BUCKET, AWS_REGION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ChunkData:
    """Data for a single chunk."""
    chunk_id: str
    start_abs: float
    end_abs: float
    segments: List[dict]  # List of {t0_abs, t1_abs, spk}


@dataclass
class VideoMetrics:
    """Speaker instability metrics for a video."""
    video_id: str
    num_chunks: int
    flip_count: int
    flip_rate: float
    boundary_switches: int
    total_unique_speakers: int
    dominant_speaker_changes: str  # Serialized list


def get_s3_client():
    """Get boto3 S3 client."""
    return boto3.client("s3", region_name=AWS_REGION)


def list_labeled_videos(s3_client) -> List[str]:
    """List all video IDs with corrected boundaries in S3."""
    prefix = "labeling/corrected_boundaries/v1/"
    video_ids = []

    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".json"):
                # Extract video_id from key
                video_id = key[len(prefix):-5]  # Remove prefix and .json
                if video_id:
                    video_ids.append(video_id)

    return video_ids


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


def list_chunks(s3_client, video_id: str, run_id: str) -> List[str]:
    """List all chunk IDs for a video/run."""
    prefix = f"runs/{video_id}/{run_id}/chunks/"
    chunk_ids = set()

    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            parts = obj["Key"].split("/")
            if len(parts) >= 5:
                chunk_ids.add(parts[4])

    return sorted(chunk_ids)


def load_chunk_data(s3_client, video_id: str, run_id: str, chunk_id: str) -> Optional[ChunkData]:
    """Load chunk metadata and diarization segments."""
    # Load chunk_metadata.json
    meta_key = f"runs/{video_id}/{run_id}/chunks/{chunk_id}/chunk_metadata.json"
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=meta_key)
        meta = json.loads(response["Body"].read())
        start_abs = meta["chunk_time_offset_s"]
        end_abs = start_abs + meta["duration_s"]
    except Exception as e:
        logger.warning(f"Error loading chunk metadata {meta_key}: {e}")
        return None

    # Load diarization_segments.json
    dia_key = f"runs/{video_id}/{run_id}/chunks/{chunk_id}/diarization_segments.json"
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=dia_key)
        dia_data = json.loads(response["Body"].read())
        segments = dia_data.get("segments", [])
    except s3_client.exceptions.NoSuchKey:
        segments = []
    except Exception as e:
        logger.warning(f"Error loading diarization {dia_key}: {e}")
        segments = []

    return ChunkData(
        chunk_id=chunk_id,
        start_abs=start_abs,
        end_abs=end_abs,
        segments=segments,
    )


def compute_dominant_speaker(segments: List[dict]) -> Optional[str]:
    """Compute the dominant speaker (most total duration) in a chunk."""
    if not segments:
        return None

    speaker_durations: Dict[str, float] = defaultdict(float)
    for seg in segments:
        duration = seg["t1_abs"] - seg["t0_abs"]
        speaker_durations[seg["spk"]] += duration

    if not speaker_durations:
        return None

    return max(speaker_durations.keys(), key=lambda spk: speaker_durations[spk])


def compute_video_metrics(video_id: str, chunks: List[ChunkData]) -> VideoMetrics:
    """Compute speaker ID instability metrics for a video."""
    # Sort chunks by start time
    chunks = sorted(chunks, key=lambda c: c.start_abs)

    num_chunks = len(chunks)

    # Compute dominant speaker per chunk
    dominant_speakers = []
    for chunk in chunks:
        dom = compute_dominant_speaker(chunk.segments)
        dominant_speakers.append((chunk.chunk_id, dom))

    # Compute flip_count: number of times dominant speaker changes between adjacent chunks
    flip_count = 0
    dominant_changes = []
    for i in range(1, len(dominant_speakers)):
        prev_dom = dominant_speakers[i-1][1]
        curr_dom = dominant_speakers[i][1]
        if prev_dom is not None and curr_dom is not None and prev_dom != curr_dom:
            flip_count += 1
            dominant_changes.append(f"{dominant_speakers[i-1][0]}:{prev_dom}->{curr_dom}")

    # Compute flip_rate
    flip_rate = flip_count / (num_chunks - 1) if num_chunks > 1 else 0.0

    # Compute boundary_switches: speaker transitions at chunk boundaries
    boundary_switches = 0
    for i in range(len(chunks) - 1):
        curr_chunk = chunks[i]
        next_chunk = chunks[i + 1]

        # Get last segment of current chunk
        if curr_chunk.segments:
            last_seg = max(curr_chunk.segments, key=lambda s: s["t1_abs"])
            last_spk = last_seg["spk"]
        else:
            continue

        # Get first segment of next chunk
        if next_chunk.segments:
            first_seg = min(next_chunk.segments, key=lambda s: s["t0_abs"])
            first_spk = first_seg["spk"]
        else:
            continue

        if last_spk != first_spk:
            boundary_switches += 1

    # Compute total unique speakers across all chunks
    # This inflates if the same person gets different labels in different chunks
    all_speakers = set()
    for chunk in chunks:
        for seg in chunk.segments:
            all_speakers.add(seg["spk"])
    total_unique_speakers = len(all_speakers)

    return VideoMetrics(
        video_id=video_id,
        num_chunks=num_chunks,
        flip_count=flip_count,
        flip_rate=round(flip_rate, 4),
        boundary_switches=boundary_switches,
        total_unique_speakers=total_unique_speakers,
        dominant_speaker_changes=";".join(dominant_changes[:10]),  # Limit for CSV
    )


def analyze_feature_impact(metrics: List[VideoMetrics]) -> str:
    """Generate markdown analysis of impact on 9 features for worst videos."""
    # Sort by flip_rate descending
    worst = sorted(metrics, key=lambda m: m.flip_rate, reverse=True)[:10]

    lines = [
        "",
        "## Impact Analysis on 9 Features (Worst 10 Videos by Flip Rate)",
        "",
        "The speaker ID instability directly impacts the following features:",
        "",
        "### Affected Features:",
        "",
        "1. **host_speech_frac** - If the 'host' speaker gets a different label in a new chunk,",
        "   their speech is no longer counted as host speech, artificially reducing this metric.",
        "",
        "2. **nonhost_speech_frac** - Inverse of above; host speech may be misclassified as non-host.",
        "",
        "3. **num_active_speakers** - Inflated when the same person appears with multiple labels.",
        "",
        "4. **nonhost_active** - May incorrectly show non-host activity when it's actually the host.",
        "",
        "5. **speaker_switches_10s** - Inflated at chunk boundaries due to label resets.",
        "",
        "6. **unique_nonhost_speakers_30s** - Severely inflated; same person counted multiple times.",
        "",
        "7. **nonhost_turns_30s** - Inflated when host turns are misclassified as non-host.",
        "",
        "8. **avg_nonhost_turn_len_30s** - Distorted by incorrect speaker attribution.",
        "",
        "9. **speech_frac** - Generally unaffected (speaker-agnostic).",
        "",
        "### Worst 10 Videos:",
        "",
    ]

    for m in worst:
        lines.append(f"- **{m.video_id}**: flip_rate={m.flip_rate:.2%}, "
                    f"chunks={m.num_chunks}, boundary_switches={m.boundary_switches}, "
                    f"unique_speakers={m.total_unique_speakers}")

    lines.extend([
        "",
        "### Recommendations:",
        "",
        "1. **Global speaker clustering**: Implement cross-chunk speaker embedding clustering",
        "   to assign consistent IDs across the entire video.",
        "",
        "2. **Boundary smoothing**: When a new chunk starts, compare speaker embeddings with",
        "   the previous chunk to maintain label consistency.",
        "",
        "3. **Feature engineering**: Consider chunk-aware features that account for potential",
        "   speaker ID resets at boundaries.",
        "",
    ])

    return "\n".join(lines)


def main():
    """Main analysis function."""
    output_dir = Path("data/call_segmenter/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "speaker_id_instability.csv"

    s3 = get_s3_client()

    # Step 1: List labeled videos
    logger.info("Listing labeled videos from S3...")
    video_ids = list_labeled_videos(s3)
    logger.info(f"Found {len(video_ids)} labeled videos")

    if not video_ids:
        logger.error("No labeled videos found!")
        return 1

    all_metrics = []

    # Step 2-4: Process each video
    for i, video_id in enumerate(video_ids):
        logger.info(f"[{i+1}/{len(video_ids)}] Processing {video_id}")

        # Get run_id
        run_id = load_latest_run_id(s3, video_id)
        if not run_id:
            logger.warning(f"  Skipping {video_id} - no run_id found")
            continue

        # List chunks
        chunk_ids = list_chunks(s3, video_id, run_id)
        if not chunk_ids:
            logger.warning(f"  Skipping {video_id} - no chunks found")
            continue

        logger.info(f"  Found {len(chunk_ids)} chunks")

        # Load chunk data
        chunks = []
        for chunk_id in chunk_ids:
            chunk_data = load_chunk_data(s3, video_id, run_id, chunk_id)
            if chunk_data:
                chunks.append(chunk_data)

        if not chunks:
            logger.warning(f"  Skipping {video_id} - no chunk data loaded")
            continue

        # Compute metrics
        metrics = compute_video_metrics(video_id, chunks)
        all_metrics.append(metrics)

        logger.info(f"  flip_rate={metrics.flip_rate:.2%}, "
                   f"boundary_switches={metrics.boundary_switches}, "
                   f"unique_speakers={metrics.total_unique_speakers}")

    # Step 5: Write CSV output
    logger.info(f"Writing output to {output_file}")

    # Sort by flip_rate descending
    all_metrics.sort(key=lambda m: m.flip_rate, reverse=True)

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "video_id",
            "num_chunks",
            "flip_count",
            "flip_rate",
            "boundary_switches",
            "total_unique_speakers",
            "dominant_speaker_changes",
        ])

        for m in all_metrics:
            writer.writerow([
                m.video_id,
                m.num_chunks,
                m.flip_count,
                m.flip_rate,
                m.boundary_switches,
                m.total_unique_speakers,
                m.dominant_speaker_changes,
            ])

    # Step 6: Add markdown analysis at end
    md_analysis = analyze_feature_impact(all_metrics)

    # Write markdown section to separate file
    md_file = output_dir / "speaker_id_instability_analysis.md"
    with open(md_file, "w") as f:
        f.write(f"# Speaker ID Instability Analysis\n\n")
        f.write(f"Generated from {len(all_metrics)} labeled videos.\n")
        f.write(md_analysis)

    logger.info(f"Wrote markdown analysis to {md_file}")

    # Summary
    logger.info("=" * 60)
    logger.info("Analysis Complete")
    logger.info(f"  Videos analyzed: {len(all_metrics)}")
    if all_metrics:
        avg_flip = sum(m.flip_rate for m in all_metrics) / len(all_metrics)
        avg_boundary = sum(m.boundary_switches for m in all_metrics) / len(all_metrics)
        avg_unique = sum(m.total_unique_speakers for m in all_metrics) / len(all_metrics)
        logger.info(f"  Avg flip_rate: {avg_flip:.2%}")
        logger.info(f"  Avg boundary_switches: {avg_boundary:.1f}")
        logger.info(f"  Avg unique_speakers: {avg_unique:.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
