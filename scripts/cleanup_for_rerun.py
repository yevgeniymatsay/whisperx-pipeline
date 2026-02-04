#!/usr/bin/env python3
"""
Cleanup S3 data for pipeline rerun with corrected diarization settings.

Two modes:
1. --all-runs: Delete ALL runs/ and latest/ folders (all videos processed with incorrect settings)
2. --export-labeled: Export 35 labeled video IDs to data/labeled_videos.txt

Usage:
    python scripts/cleanup_for_rerun.py --all-runs           # Dry run - show what would be deleted
    python scripts/cleanup_for_rerun.py --all-runs --execute # Actually delete
    python scripts/cleanup_for_rerun.py --export-labeled     # Export labeled video list
"""

import argparse
from pathlib import Path

import boto3

# S3 config
S3_BUCKET = "rezora-whisperx-us-east-1-864981718771"

# Paths
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
LABELED_VIDEOS_PATH = DATA_DIR / "labeled_videos.txt"

# Prefixes to delete (all runs used incorrect speaker settings)
PREFIXES_TO_DELETE = [
    "runs/",
    "latest/",
]

# Prefix for labeled videos
LABELED_BOUNDARIES_PREFIX = "labeling/corrected_boundaries/v1/"


def count_objects(s3_client, prefix: str) -> int:
    """Count objects under a prefix."""
    paginator = s3_client.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        count += page.get("KeyCount", 0)
    return count


def delete_prefix(s3_client, prefix: str, execute: bool = False) -> int:
    """Delete all objects under a prefix. Returns count of deleted objects."""
    paginator = s3_client.get_paginator("list_objects_v2")
    deleted = 0

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        contents = page.get("Contents", [])
        if not contents:
            continue

        if execute:
            # Batch delete (up to 1000 at a time)
            delete_request = {
                "Objects": [{"Key": obj["Key"]} for obj in contents],
                "Quiet": True,
            }
            s3_client.delete_objects(Bucket=S3_BUCKET, Delete=delete_request)

        deleted += len(contents)

    return deleted


def export_labeled_videos(s3_client) -> list[str]:
    """Get list of labeled video IDs from S3."""
    paginator = s3_client.get_paginator("list_objects_v2")
    video_ids = []

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=LABELED_BOUNDARIES_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Extract video_id from path like labeling/corrected_boundaries/v1/{video_id}.json
            filename = key.split("/")[-1]
            if filename.endswith(".json"):
                video_id = filename.replace(".json", "")
                video_ids.append(video_id)

    return sorted(video_ids)


def main():
    parser = argparse.ArgumentParser(
        description="Cleanup S3 data for pipeline rerun"
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Delete ALL runs/ and latest/ folders",
    )
    parser.add_argument(
        "--export-labeled",
        action="store_true",
        help="Export labeled video IDs to data/labeled_videos.txt",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete (default is dry-run)",
    )
    args = parser.parse_args()

    if not args.all_runs and not args.export_labeled:
        parser.error("Must specify --all-runs or --export-labeled")

    s3 = boto3.client("s3")

    if args.export_labeled:
        print(f"Exporting labeled video IDs from s3://{S3_BUCKET}/{LABELED_BOUNDARIES_PREFIX}...")
        video_ids = export_labeled_videos(s3)

        print(f"Found {len(video_ids)} labeled videos:")
        for vid in video_ids[:10]:
            print(f"  - {vid}")
        if len(video_ids) > 10:
            print(f"  ... and {len(video_ids) - 10} more")

        # Ensure data directory exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        with open(LABELED_VIDEOS_PATH, "w") as f:
            for vid in video_ids:
                f.write(f"{vid}\n")

        print(f"\n✓ Exported {len(video_ids)} video IDs to {LABELED_VIDEOS_PATH}")

    if args.all_runs:
        print(f"\n{'=' * 60}")
        print(f"{'EXECUTING DELETE' if args.execute else 'DRY RUN - Preview only'}")
        print(f"{'=' * 60}\n")

        total_deleted = 0

        for prefix in PREFIXES_TO_DELETE:
            count = count_objects(s3, prefix)
            print(f"s3://{S3_BUCKET}/{prefix}")
            print(f"  Objects to delete: {count}")

            if count > 0:
                deleted = delete_prefix(s3, prefix, execute=args.execute)
                total_deleted += deleted
                if args.execute:
                    print(f"  ✓ Deleted {deleted} objects")
                else:
                    print(f"  Would delete {deleted} objects")

            print()

        print(f"{'=' * 60}")
        if args.execute:
            print(f"Total deleted: {total_deleted} objects")
        else:
            print(f"Total to delete: {total_deleted} objects")
            print(f"\nRun with --execute to actually delete")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
