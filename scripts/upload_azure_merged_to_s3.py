#!/usr/bin/env python3
"""Upload Azure merged diarization outputs (merged.diarized.json) to S3.

This is a thin helper to make the upstream A/B reproducible:
  - local layout: <local_root>/<video_id>/merged.diarized.json
  - s3 layout:    s3://<bucket>/<s3_prefix>/<video_id>.json

We upload JSON only (no audio), and do not require Azure credentials.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import boto3

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import S3_BUCKET, AWS_REGION

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_s3_client():
    return boto3.client("s3", region_name=AWS_REGION)


def load_video_ids(video_list: Optional[Path], local_root: Path) -> List[str]:
    if video_list is not None:
        out: List[str] = []
        for line in video_list.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
        return out

    vids = []
    for p in sorted(local_root.iterdir()):
        if p.is_dir():
            vids.append(p.name)
    return vids


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload Azure merged diarize outputs to S3")
    parser.add_argument("--local-root", type=Path, required=True,
                        help="Local root containing <video_id>/merged.diarized.json")
    parser.add_argument("--s3-out-prefix", type=str, required=True,
                        help="S3 key prefix to write <video_id>.json under (within the bucket)")
    parser.add_argument("--video-list", type=Path, default=None,
                        help="Optional file with video IDs to upload (one per line)")
    parser.add_argument("--dry-run", action="store_true", help="Log actions without uploading")
    args = parser.parse_args()

    local_root = Path(args.local_root)
    if not local_root.exists():
        raise FileNotFoundError(f"--local-root not found: {local_root}")

    s3_prefix = str(args.s3_out_prefix).rstrip("/")
    video_ids = load_video_ids(args.video_list, local_root)
    if not video_ids:
        logger.error("No video_ids found to upload")
        return 2

    s3 = get_s3_client()

    uploaded = 0
    skipped = 0
    for idx, vid in enumerate(video_ids):
        src = local_root / vid / "merged.diarized.json"
        if not src.exists():
            logger.warning(f"[{idx+1}/{len(video_ids)}] Missing merged.diarized.json for {vid}: {src}")
            skipped += 1
            continue

        try:
            # Ensure it is valid JSON before upload (catches truncated files).
            _ = json.loads(src.read_text())
        except Exception as e:
            logger.warning(f"[{idx+1}/{len(video_ids)}] Invalid JSON for {vid}: {e}")
            skipped += 1
            continue

        dst_key = f"{s3_prefix}/{vid}.json"
        logger.info(f"[{idx+1}/{len(video_ids)}] Upload {src} -> s3://{S3_BUCKET}/{dst_key}")
        if not args.dry_run:
            s3.upload_file(str(src), S3_BUCKET, dst_key)
        uploaded += 1

    logger.info(f"Done. uploaded={uploaded} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

