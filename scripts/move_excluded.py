#!/usr/bin/env python3
"""
Move excluded videos from audio/pretraining/ to audio/pretraining-excluded/ in S3.

Reads video IDs from data/excluded_videos.txt and moves matching audio files.

Usage:
  python scripts/move_excluded_videos.py           # Dry-run (preview)
  python scripts/move_excluded_videos.py --execute # Actually move files
"""

import argparse
from pathlib import Path

import boto3

# Paths
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
EXCLUDED_PATH = DATA_DIR / "excluded_videos.txt"

# S3 config
BUCKET = "rezora-whisperx-us-east-1-864981718771"
SOURCE_PREFIX = "audio/pretraining/"
DEST_PREFIX = "audio/pretraining-excluded/"


def load_excluded_video_ids() -> set[str]:
    """Load excluded video IDs from file."""
    if not EXCLUDED_PATH.exists():
        return set()
    with open(EXCLUDED_PATH) as f:
        return {line.strip() for line in f if line.strip()}


def find_matching_files(s3_client, video_ids: set[str]) -> dict[str, str]:
    """Find S3 files matching the video IDs.

    Returns dict mapping video_id -> full S3 key.
    """
    matches: dict[str, str] = {}
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=BUCKET, Prefix=SOURCE_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key.split("/")[-1]

            # Check if any video ID is in the filename
            for vid in video_ids:
                if vid in filename:
                    matches[vid] = key
                    break

    return matches


def move_files(s3_client, matches: dict[str, str], execute: bool) -> None:
    """Move files from source to destination prefix."""
    if not matches:
        print("No files to move.")
        return

    print(f"\n{'=' * 60}")
    print(f"{'EXECUTING MOVE' if execute else 'DRY RUN - Preview only'}")
    print(f"{'=' * 60}\n")

    moved = 0
    for vid, source_key in sorted(matches.items()):
        filename = source_key.split("/")[-1]
        dest_key = f"{DEST_PREFIX}{filename}"

        print(f"  {source_key}")
        print(f"    → {dest_key}")

        if execute:
            # Copy to destination
            s3_client.copy_object(
                Bucket=BUCKET,
                CopySource={"Bucket": BUCKET, "Key": source_key},
                Key=dest_key,
            )
            # Delete from source
            s3_client.delete_object(Bucket=BUCKET, Key=source_key)
            moved += 1
        print()

    print(f"{'=' * 60}")
    if execute:
        print(f"Moved {moved} file(s)")
    else:
        print(f"Would move {len(matches)} file(s)")
        print("\nRun with --execute to actually move files")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Move excluded videos to separate S3 prefix"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move files (default is dry-run)",
    )
    args = parser.parse_args()

    # Load excluded video IDs
    video_ids = load_excluded_video_ids()
    if not video_ids:
        print(f"No excluded videos found in {EXCLUDED_PATH}")
        print("Use the TUI (scripts/select_whisperx_videos.py) to mark videos with 'x'")
        return

    print(f"Found {len(video_ids)} excluded video ID(s) in {EXCLUDED_PATH.name}")

    # Find matching files in S3
    s3 = boto3.client("s3")
    print(f"\nSearching for files in s3://{BUCKET}/{SOURCE_PREFIX}...")
    matches = find_matching_files(s3, video_ids)

    # Report any video IDs that weren't found
    not_found = video_ids - set(matches.keys())
    if not_found:
        print(f"\nWarning: {len(not_found)} video ID(s) not found in S3:")
        for vid in sorted(not_found):
            print(f"  - {vid}")

    print(f"\nFound {len(matches)} matching file(s) to move")

    # Move (or preview) the files
    move_files(s3, matches, args.execute)


if __name__ == "__main__":
    main()
