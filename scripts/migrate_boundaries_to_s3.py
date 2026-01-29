#!/usr/bin/env python3
"""One-time migration of corrected_boundaries.csv to S3.

Reads existing CSV file and uploads each video's boundaries as JSON to S3.

Usage:
    python scripts/migrate_boundaries_to_s3.py \
        --input apps/speaker-labeler/backend/corrected_boundaries.csv

Options:
    --input PATH      Path to existing CSV file
    --bucket NAME     S3 bucket (default: S3_BUCKET env or rezora-whisperx-us-east-1-864981718771)
    --prefix PREFIX   S3 key prefix (default: labeling/corrected_boundaries/v1/)
    --dry-run         Print what would be uploaded without actually uploading
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

import boto3
from botocore.exceptions import ClientError

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_BUCKET = os.environ.get("S3_BUCKET", "rezora-whisperx-us-east-1-864981718771")
DEFAULT_PREFIX = "labeling/corrected_boundaries/v1/"


def load_csv(csv_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load boundaries from CSV and group by video_id.

    Returns:
        Dict mapping video_id to list of boundary dicts.
    """
    by_video: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row["video_id"]
            by_video[video_id].append({
                "call_index": int(row["call_index"]),
                "start_s": float(row["start_s"]),
                "end_s": float(row["end_s"]),
                "corrected_at": row["corrected_at"],
            })

    # Sort each video's boundaries by start time
    for video_id in by_video:
        by_video[video_id].sort(key=lambda x: x["start_s"])

    return dict(by_video)


def upload_to_s3(
    s3_client,
    bucket: str,
    prefix: str,
    video_id: str,
    boundaries: List[Dict[str, Any]],
    dry_run: bool = False,
) -> bool:
    """Upload boundaries for a video to S3.

    Returns:
        True if successful (or dry run), False on error.
    """
    key = f"{prefix}{video_id}.json"
    doc = {
        "video_id": video_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "boundaries": boundaries,
    }
    body = json.dumps(doc, indent=2)

    if dry_run:
        print(f"  [DRY RUN] Would upload to s3://{bucket}/{key}")
        print(f"            {len(boundaries)} boundaries, {len(body)} bytes")
        return True

    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
        print(f"  Uploaded s3://{bucket}/{key} ({len(boundaries)} boundaries)")
        return True
    except ClientError as e:
        print(f"  ERROR uploading {video_id}: {e}")
        return False


def verify_upload(
    s3_client,
    bucket: str,
    prefix: str,
    video_id: str,
    expected_count: int,
) -> bool:
    """Verify that uploaded boundaries can be read back.

    Returns:
        True if verification passes, False otherwise.
    """
    key = f"{prefix}{video_id}.json"
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        doc = json.loads(response["Body"].read())
        actual_count = len(doc.get("boundaries", []))
        if actual_count != expected_count:
            print(f"  VERIFY FAILED: {video_id} - expected {expected_count}, got {actual_count}")
            return False
        return True
    except ClientError as e:
        print(f"  VERIFY FAILED: {video_id} - {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Migrate corrected boundaries from CSV to S3")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to existing corrected_boundaries.csv",
    )
    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help=f"S3 bucket name (default: {DEFAULT_BUCKET})",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help=f"S3 key prefix (default: {DEFAULT_PREFIX})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without actually uploading",
    )
    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    print(f"Loading boundaries from: {args.input}")
    by_video = load_csv(args.input)

    if not by_video:
        print("No boundaries found in CSV file.")
        sys.exit(0)

    print(f"Found {len(by_video)} videos with corrected boundaries:")
    for video_id, boundaries in sorted(by_video.items()):
        print(f"  {video_id}: {len(boundaries)} calls")

    print()
    print(f"Target: s3://{args.bucket}/{args.prefix}")
    if args.dry_run:
        print("[DRY RUN MODE - no changes will be made]")
    print()

    # Initialize S3 client
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))

    # Upload each video's boundaries
    print("Uploading to S3...")
    success_count = 0
    for video_id, boundaries in sorted(by_video.items()):
        if upload_to_s3(s3, args.bucket, args.prefix, video_id, boundaries, args.dry_run):
            success_count += 1

    print()
    print(f"Uploaded {success_count}/{len(by_video)} videos")

    # Verify uploads (skip in dry run)
    if not args.dry_run:
        print()
        print("Verifying uploads...")
        verify_count = 0
        for video_id, boundaries in sorted(by_video.items()):
            if verify_upload(s3, args.bucket, args.prefix, video_id, len(boundaries)):
                verify_count += 1

        print()
        print(f"Verified {verify_count}/{len(by_video)} videos")

        if verify_count == len(by_video):
            print()
            print("SUCCESS: All boundaries migrated and verified!")
            print()
            print("Next steps:")
            print(f"  1. Delete the old CSV file: rm {args.input}")
            print("  2. Restart the backend to use S3 storage")
        else:
            print()
            print("WARNING: Some verifications failed. Check errors above.")
            sys.exit(1)


if __name__ == "__main__":
    main()
