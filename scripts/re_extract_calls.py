#!/usr/bin/env python3
"""Re-extract call audio using human-corrected boundaries.

This script reads corrected boundaries from the speaker-labeler app
and re-extracts audio segments at the corrected positions. Optionally
re-runs WhisperX transcription on the newly extracted segments.

Usage:
    # Re-extract all corrected videos
    python scripts/re_extract_calls.py

    # Re-extract specific video
    python scripts/re_extract_calls.py --video-id VIDEO_ID

    # Also re-transcribe after extraction
    python scripts/re_extract_calls.py --retranscribe

    # Dry run to see what would be done
    python scripts/re_extract_calls.py --dry-run
"""
import argparse
import csv
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import boto3

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
BOUNDARIES_FILE = REPO_ROOT / "apps" / "speaker-labeler" / "backend" / "corrected_boundaries.csv"

# S3 config - import from pipeline or use defaults
try:
    sys.path.insert(0, str(REPO_ROOT))
    from pipeline.config import S3_BUCKET, AWS_REGION
except ImportError:
    S3_BUCKET = "rezora-whisperx-us-east-1-864981718771"
    AWS_REGION = "us-east-1"


@dataclass
class CallBoundary:
    """A corrected call boundary."""
    video_id: str
    call_index: int
    start_s: float
    end_s: float
    corrected_at: str


def load_corrected_boundaries(video_id: Optional[str] = None) -> list[CallBoundary]:
    """Load corrected boundaries from CSV, optionally filtered by video_id."""
    if not BOUNDARIES_FILE.exists():
        return []

    boundaries = []
    with open(BOUNDARIES_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if video_id is None or row["video_id"] == video_id:
                boundaries.append(CallBoundary(
                    video_id=row["video_id"],
                    call_index=int(row["call_index"]),
                    start_s=float(row["start_s"]),
                    end_s=float(row["end_s"]),
                    corrected_at=row["corrected_at"],
                ))
    return boundaries


def get_corrected_video_ids() -> set[str]:
    """Get set of video IDs that have corrected boundaries."""
    boundaries = load_corrected_boundaries()
    return set(b.video_id for b in boundaries)


def extract_audio_segment(
    input_path: str,
    output_path: str,
    start_s: float,
    end_s: float,
) -> bool:
    """Extract audio segment using ffmpeg."""
    duration = end_s - start_s
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ss", str(start_s),
        "-t", str(duration),
        "-acodec", "libmp3lame",
        "-q:a", "2",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def find_audio_file(s3_client, bucket: str, video_id: str) -> str | None:
    """Find the audio file for a video in S3."""
    paginator = s3_client.get_paginator("list_objects_v2")

    for prefix in ["audio/pretraining/", "audio/expired_listing/", "audio/"]:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".mp3"):
                    filename = key.split("/")[-1]
                    # Check if video_id is in filename
                    if video_id in filename:
                        return key
    return None


def get_latest_run_id(s3_client, bucket: str, video_id: str) -> str | None:
    """Get the latest run_id for a video."""
    paginator = s3_client.get_paginator("list_objects_v2")
    prefix = f"runs/{video_id}/"

    run_ids = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for common_prefix in page.get("CommonPrefixes", []):
            # Extract run_id from path like "runs/VIDEO_ID/RUN_ID/"
            parts = common_prefix["Prefix"].rstrip("/").split("/")
            if len(parts) >= 3:
                run_ids.add(parts[2])

    if not run_ids:
        return None

    # Return the latest (assuming run_ids are timestamps)
    return sorted(run_ids)[-1]


def re_extract_video(
    s3_client,
    bucket: str,
    video_id: str,
    boundaries: list[CallBoundary],
    run_id: str | None = None,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Re-extract calls for a video using corrected boundaries.

    Returns (extracted_count, error_count)
    """
    extracted = 0
    errors = 0

    # Find audio file
    audio_key = find_audio_file(s3_client, bucket, video_id)
    if not audio_key:
        print(f"  ERROR: No audio file found for {video_id}")
        return 0, len(boundaries)

    # Get or create run_id
    if not run_id:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"  Audio: {audio_key}")
    print(f"  Run ID: {run_id}")
    print(f"  Boundaries to process: {len(boundaries)}")

    if dry_run:
        for b in boundaries:
            print(f"    Call {b.call_index}: {b.start_s:.1f}s - {b.end_s:.1f}s ({b.end_s - b.start_s:.1f}s)")
        return len(boundaries), 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        local_audio = tmpdir / "full.mp3"

        # Download full audio
        try:
            print(f"  Downloading audio...")
            s3_client.download_file(bucket, audio_key, str(local_audio))
        except Exception as e:
            print(f"  ERROR downloading audio: {e}")
            return 0, len(boundaries)

        # Extract each call segment
        for boundary in boundaries:
            call_id = f"call_{boundary.call_index:03d}"

            # Extract audio segment
            local_segment = tmpdir / f"{call_id}.mp3"
            if not extract_audio_segment(
                str(local_audio),
                str(local_segment),
                boundary.start_s,
                boundary.end_s,
            ):
                print(f"    ERROR extracting {call_id}")
                errors += 1
                continue

            # Upload extracted audio
            output_key = f"runs/{video_id}/{run_id}/calls/{call_id}/audio.mp3"
            try:
                s3_client.upload_file(str(local_segment), bucket, output_key)
            except Exception as e:
                print(f"    ERROR uploading {call_id}: {e}")
                errors += 1
                continue

            # Create metadata file with boundary info
            metadata = {
                "video_id": video_id,
                "call_index": boundary.call_index,
                "start_s": boundary.start_s,
                "end_s": boundary.end_s,
                "duration_s": boundary.end_s - boundary.start_s,
                "corrected_at": boundary.corrected_at,
                "extracted_at": datetime.now().isoformat(),
                "source": "human_corrected",
            }
            metadata_path = tmpdir / f"{call_id}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            metadata_key = f"runs/{video_id}/{run_id}/calls/{call_id}/boundary_metadata.json"
            try:
                s3_client.upload_file(str(metadata_path), bucket, metadata_key)
            except Exception as e:
                print(f"    WARNING: Could not upload metadata for {call_id}: {e}")

            duration = boundary.end_s - boundary.start_s
            print(f"    Extracted {call_id}: {boundary.start_s:.1f}s - {boundary.end_s:.1f}s ({duration:.1f}s)")
            extracted += 1

    return extracted, errors


def main():
    parser = argparse.ArgumentParser(
        description="Re-extract call audio using human-corrected boundaries"
    )
    parser.add_argument(
        "--video-id",
        help="Process only this video ID"
    )
    parser.add_argument(
        "--bucket",
        default=S3_BUCKET,
        help=f"S3 bucket (default: {S3_BUCKET})"
    )
    parser.add_argument(
        "--run-id",
        help="Use specific run ID (default: generate timestamp)"
    )
    parser.add_argument(
        "--retranscribe",
        action="store_true",
        help="Also re-run WhisperX transcription after extraction"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without doing it"
    )
    args = parser.parse_args()

    # Get corrected video IDs
    if args.video_id:
        video_ids = {args.video_id}
        all_boundaries = load_corrected_boundaries(args.video_id)
        if not all_boundaries:
            print(f"No corrected boundaries found for video: {args.video_id}")
            return
    else:
        video_ids = get_corrected_video_ids()
        if not video_ids:
            print("No videos with corrected boundaries found.")
            print("Use the speaker-labeler app to correct call boundaries first.")
            return
        all_boundaries = load_corrected_boundaries()

    print(f"Found {len(video_ids)} videos with corrected boundaries")
    print(f"Total corrected calls: {len(all_boundaries)}")

    # Group boundaries by video
    by_video: dict[str, list[CallBoundary]] = {}
    for b in all_boundaries:
        if b.video_id not in by_video:
            by_video[b.video_id] = []
        by_video[b.video_id].append(b)

    # Sort boundaries within each video by call_index
    for vid in by_video:
        by_video[vid].sort(key=lambda x: x.call_index)

    s3 = boto3.client("s3", region_name=AWS_REGION)

    total_extracted = 0
    total_errors = 0

    for video_id in sorted(by_video.keys()):
        boundaries = by_video[video_id]
        print(f"\nProcessing: {video_id} ({len(boundaries)} calls)")

        extracted, errors = re_extract_video(
            s3,
            args.bucket,
            video_id,
            boundaries,
            run_id=args.run_id,
            dry_run=args.dry_run,
        )
        total_extracted += extracted
        total_errors += errors

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Summary:")
    print(f"  Videos processed: {len(by_video)}")
    print(f"  Calls extracted: {total_extracted}")
    print(f"  Errors: {total_errors}")

    if args.retranscribe and not args.dry_run and total_extracted > 0:
        print("\nRe-transcription requested but not yet implemented.")
        print("Run WhisperX manually on the extracted segments:")
        print(f"  python -m pipeline.cli --video-id VIDEO_ID --run-id RUN_ID")


if __name__ == "__main__":
    main()
