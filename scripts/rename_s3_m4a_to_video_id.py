#!/usr/bin/env python3
"""One-shot: rename S3 m4a files from 'Title - VIDEO_ID.m4a' to 'VIDEO_ID.m4a'.

Skips .temp.m4a files and files already in VIDEO_ID.m4a format.

Usage:
    python scripts/rename_s3_m4a_to_video_id.py          # Dry-run
    python scripts/rename_s3_m4a_to_video_id.py --execute # Actually rename
"""
from __future__ import annotations

import argparse
import re

import boto3

BUCKET = "rezora-whisperx-us-east-1-864981718771"
PREFIX = "audio_m4a_aac/pretraining/"

# Match the 11-char YouTube video ID at the end of the filename.
_VIDEO_ID_RE = re.compile(r"[- ]([A-Za-z0-9_-]{11})\.m4a$")
# Match files already in VIDEO_ID.m4a format (just the bare ID).
_ALREADY_CLEAN_RE = re.compile(r"^[A-Za-z0-9_-]{11}\.m4a$")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rename S3 m4a files to VIDEO_ID.m4a")
    parser.add_argument("--execute", action="store_true", help="Actually rename (default: dry-run)")
    args = parser.parse_args()

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    to_rename: list[tuple[str, str]] = []  # (old_key, new_key)
    skipped_temp = 0
    skipped_clean = 0
    skipped_no_id = 0

    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key.rsplit("/", 1)[-1]

            if filename.endswith(".temp.m4a"):
                skipped_temp += 1
                continue

            if _ALREADY_CLEAN_RE.match(filename):
                skipped_clean += 1
                continue

            m = _VIDEO_ID_RE.search(filename)
            if not m:
                print(f"  SKIP (no video ID): {filename}")
                skipped_no_id += 1
                continue

            video_id = m.group(1)
            new_key = f"{PREFIX}{video_id}.m4a"
            to_rename.append((key, new_key))

    print(f"To rename:      {len(to_rename)}")
    print(f"Already clean:  {skipped_clean}")
    print(f"Skipped .temp:  {skipped_temp}")
    print(f"Skipped no-ID:  {skipped_no_id}")
    print()

    if not args.execute:
        print("DRY RUN - would rename:")
        for old, new in to_rename[:10]:
            print(f"  {old.rsplit('/', 1)[-1]}")
            print(f"    -> {new.rsplit('/', 1)[-1]}")
        if len(to_rename) > 10:
            print(f"  ... and {len(to_rename) - 10} more")
        print("\nRun with --execute to apply.")
        return

    succeeded = 0
    failed = 0
    for i, (old_key, new_key) in enumerate(to_rename, 1):
        try:
            s3.copy_object(
                Bucket=BUCKET,
                CopySource={"Bucket": BUCKET, "Key": old_key},
                Key=new_key,
            )
            s3.delete_object(Bucket=BUCKET, Key=old_key)
            succeeded += 1
            if i % 20 == 0 or i == len(to_rename):
                print(f"  [{i}/{len(to_rename)}] renamed")
        except Exception as e:
            failed += 1
            print(f"  FAILED: {old_key} -> {new_key}: {e}")

    print(f"\nDone: {succeeded} renamed, {failed} failed")


if __name__ == "__main__":
    main()
