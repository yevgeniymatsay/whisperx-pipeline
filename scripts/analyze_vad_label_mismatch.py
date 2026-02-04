#!/usr/bin/env python3
"""VAD Gap vs Labels Mismatch Analysis.

Analyzes how well VAD chunk coverage aligns with human-labeled call boundaries.
Outputs metrics to data/call_segmenter/analysis/vad_gap_label_mismatch.json

Usage:
    python scripts/analyze_vad_label_mismatch.py
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

import boto3

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import S3_BUCKET, AWS_REGION

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ChunkCoverage:
    """Coverage interval from a chunk."""
    start_abs: float
    end_abs: float

@dataclass
class LabeledCall:
    """A labeled call boundary."""
    video_id: str
    call_index: int
    start: float
    end: float

@dataclass
class CallAnalysis:
    """Analysis result for a single call."""
    video_id: str
    call_index: int
    call_start: float
    call_end: float
    covered_fraction: float
    delta_start_to_first_speech_s: Optional[float]
    start_in_gap: bool
    end_in_gap: bool

# =============================================================================
# S3 Loading Functions
# =============================================================================

def get_s3_client():
    return boto3.client("s3", region_name=AWS_REGION)

def load_labeled_videos(s3_client) -> Dict[str, List[Dict]]:
    """Load all labeled boundaries from S3."""
    prefix = "labeling/corrected_boundaries/v1/"
    labels = {}

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
                response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
                doc = json.loads(response["Body"].read())
                boundaries = doc.get("boundaries", [])
                if boundaries:
                    labels[video_id] = boundaries
            except Exception as e:
                print(f"Warning: Error loading {video_id}: {e}")

    return labels

def load_latest_run_id(s3_client, video_id: str) -> Optional[str]:
    """Load run_id from latest pointer."""
    key = f"latest/{video_id}.json"
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        data = json.loads(response["Body"].read())
        return data.get("run_id")
    except Exception:
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

def load_chunk_metadata(s3_client, video_id: str, run_id: str, chunk_id: str) -> Optional[ChunkCoverage]:
    """Load chunk coverage interval."""
    key = f"runs/{video_id}/{run_id}/chunks/{chunk_id}/chunk_metadata.json"
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        data = json.loads(response["Body"].read())
        start = data["chunk_time_offset_s"]
        end = start + data["duration_s"]
        return ChunkCoverage(start_abs=start, end_abs=end)
    except Exception:
        return None

def load_diarization_segments(s3_client, video_id: str, run_id: str, chunk_id: str) -> List[Dict]:
    """Load diarization segments for a chunk."""
    key = f"runs/{video_id}/{run_id}/chunks/{chunk_id}/diarization_segments.json"
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        data = json.loads(response["Body"].read())
        return data.get("segments", [])
    except Exception:
        return []

# =============================================================================
# Analysis Functions
# =============================================================================

def point_in_coverage(t: float, coverages: List[ChunkCoverage]) -> bool:
    """Check if a point falls inside any chunk coverage."""
    for cov in coverages:
        if cov.start_abs <= t <= cov.end_abs:
            return True
    return False

def compute_overlap(call_start: float, call_end: float, coverages: List[ChunkCoverage]) -> float:
    """Compute fraction of call covered by chunks."""
    call_duration = call_end - call_start
    if call_duration <= 0:
        return 0.0

    # Sort coverages by start
    sorted_covs = sorted(coverages, key=lambda c: c.start_abs)

    # Compute union of coverage intervals
    merged = []
    for cov in sorted_covs:
        if not merged:
            merged.append([cov.start_abs, cov.end_abs])
        else:
            last = merged[-1]
            if cov.start_abs <= last[1]:
                last[1] = max(last[1], cov.end_abs)
            else:
                merged.append([cov.start_abs, cov.end_abs])

    # Compute overlap with call
    total_overlap = 0.0
    for start, end in merged:
        overlap_start = max(call_start, start)
        overlap_end = min(call_end, end)
        if overlap_end > overlap_start:
            total_overlap += overlap_end - overlap_start

    return total_overlap / call_duration

def find_first_speech_in_call(call_start: float, call_end: float, segments: List[Dict]) -> Optional[float]:
    """Find the first diarization segment start time inside the call."""
    for seg in sorted(segments, key=lambda s: s["t0_abs"]):
        if seg["t0_abs"] >= call_start and seg["t0_abs"] < call_end:
            return seg["t0_abs"]
    return None

def analyze_video(
    s3_client,
    video_id: str,
    boundaries: List[Dict]
) -> List[CallAnalysis]:
    """Analyze all labeled calls for a video."""
    results = []

    # Get run_id
    run_id = load_latest_run_id(s3_client, video_id)
    if not run_id:
        print(f"  Warning: No run_id for {video_id}")
        return results

    # Load chunks
    chunk_ids = list_chunks(s3_client, video_id, run_id)
    if not chunk_ids:
        print(f"  Warning: No chunks for {video_id}")
        return results

    # Load chunk coverages
    coverages = []
    for chunk_id in chunk_ids:
        cov = load_chunk_metadata(s3_client, video_id, run_id, chunk_id)
        if cov:
            coverages.append(cov)

    # Load all diarization segments
    all_segments = []
    for chunk_id in chunk_ids:
        segs = load_diarization_segments(s3_client, video_id, run_id, chunk_id)
        all_segments.extend(segs)
    all_segments.sort(key=lambda s: s["t0_abs"])

    # Analyze each labeled call
    for i, boundary in enumerate(boundaries):
        call_start = boundary["start_s"]
        call_end = boundary["end_s"]
        call_index = boundary.get("call_index", i)

        # Check if start/end fall in gaps
        start_in_gap = not point_in_coverage(call_start, coverages)
        end_in_gap = not point_in_coverage(call_end, coverages)

        # Compute covered fraction
        covered_fraction = compute_overlap(call_start, call_end, coverages)

        # Find first speech inside call
        first_speech = find_first_speech_in_call(call_start, call_end, all_segments)
        delta_start_to_first_speech = None
        if first_speech is not None:
            delta_start_to_first_speech = first_speech - call_start

        results.append(CallAnalysis(
            video_id=video_id,
            call_index=call_index,
            call_start=call_start,
            call_end=call_end,
            covered_fraction=covered_fraction,
            delta_start_to_first_speech_s=delta_start_to_first_speech,
            start_in_gap=start_in_gap,
            end_in_gap=end_in_gap,
        ))

    return results

# =============================================================================
# Main
# =============================================================================

def main():
    print("VAD Gap vs Labels Mismatch Analysis")
    print("=" * 60)

    s3_client = get_s3_client()

    # Load labeled videos
    print("Loading labeled boundaries from S3...")
    labels = load_labeled_videos(s3_client)
    print(f"Found {len(labels)} labeled videos")

    # Analyze each video
    all_analyses: List[CallAnalysis] = []
    for i, (video_id, boundaries) in enumerate(labels.items()):
        print(f"[{i+1}/{len(labels)}] Analyzing {video_id} ({len(boundaries)} calls)")
        analyses = analyze_video(s3_client, video_id, boundaries)
        all_analyses.extend(analyses)

    print(f"\nTotal labeled calls analyzed: {len(all_analyses)}")

    if not all_analyses:
        print("No calls to analyze!")
        return 1

    # Compute summary statistics
    total_calls = len(all_analyses)
    calls_with_start_in_gap = sum(1 for a in all_analyses if a.start_in_gap)
    calls_with_end_in_gap = sum(1 for a in all_analyses if a.end_in_gap)
    avg_covered_fraction = sum(a.covered_fraction for a in all_analyses) / total_calls if total_calls else 0

    # Average delta to first speech (only for calls that have speech)
    delta_values = [a.delta_start_to_first_speech_s for a in all_analyses
                    if a.delta_start_to_first_speech_s is not None]
    avg_delta = sum(delta_values) / len(delta_values) if delta_values else 0

    # Check parquet file
    parquet_path = Path("data/call_segmenter/v1/windows_all.parquet")
    y1_zero_speech = 0
    y1_zero_speakers = 0

    if parquet_path.exists():
        print(f"\nAnalyzing parquet file: {parquet_path}")
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)

            # Windows where y=1 AND speech_frac==0
            if "y" in df.columns and "speech_frac" in df.columns:
                y1_zero_speech = len(df[(df["y"] == 1) & (df["speech_frac"] == 0)])

            # Windows where y=1 AND num_active_speakers==0
            if "y" in df.columns and "num_active_speakers" in df.columns:
                y1_zero_speakers = len(df[(df["y"] == 1) & (df["num_active_speakers"] == 0)])

            print(f"  y=1 windows with zero speech_frac: {y1_zero_speech}")
            print(f"  y=1 windows with zero speakers: {y1_zero_speakers}")
        except ImportError:
            print("  pandas not available, skipping parquet analysis")
    else:
        print(f"\nParquet file not found: {parquet_path}")

    # Find worst 25 calls by covered_fraction
    sorted_analyses = sorted(all_analyses, key=lambda a: a.covered_fraction)
    worst_25 = sorted_analyses[:25]

    # Build output
    output = {
        "summary": {
            "total_labeled_calls": total_calls,
            "calls_with_start_in_gap": calls_with_start_in_gap,
            "calls_with_end_in_gap": calls_with_end_in_gap,
            "avg_covered_fraction": round(avg_covered_fraction, 4),
            "avg_delta_start_to_first_speech_s": round(avg_delta, 2),
            "y1_windows_with_zero_speech": y1_zero_speech,
            "y1_windows_with_zero_speakers": y1_zero_speakers,
        },
        "worst_25_calls": [
            {
                "video_id": a.video_id,
                "call_index": a.call_index,
                "call_start": round(a.call_start, 2),
                "call_end": round(a.call_end, 2),
                "covered_fraction": round(a.covered_fraction, 4),
                "delta_start_to_first_speech_s": round(a.delta_start_to_first_speech_s, 2)
                    if a.delta_start_to_first_speech_s is not None else None,
                "start_in_gap": a.start_in_gap,
                "end_in_gap": a.end_in_gap,
            }
            for a in worst_25
        ]
    }

    # Write output
    output_dir = Path("data/call_segmenter/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "vad_gap_label_mismatch.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  Total labeled calls: {total_calls}")
    print(f"  Calls with start in VAD gap: {calls_with_start_in_gap} ({100*calls_with_start_in_gap/total_calls:.1f}%)")
    print(f"  Calls with end in VAD gap: {calls_with_end_in_gap} ({100*calls_with_end_in_gap/total_calls:.1f}%)")
    print(f"  Average covered fraction: {avg_covered_fraction:.2%}")
    print(f"  Average delta to first speech: {avg_delta:.2f}s")
    print(f"\nOutput written to: {output_path}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
