#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import AWS_REGION, S3_BUCKET
from pipeline.call_extractor_wavlm.io import build_audio_index, s3_list_keys, utc_now_compact

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_EVAL_VIDEO_IDS = [
    # Hard tiny-gap / zero-gap
    "Qa-ppZFUp0g",
    "DEt3IRqqUVs",
    "CfMJ01KP_ns",
    "D4uiHjHW4AU",
    "-POaWp9_UaM",
    # Long/high-signal
    "FkgGv2iMjEo",
    "4ipwbOJRMck",
    "9g2gwfWIbUk",
    # No-call negatives
    "15QzruVINDc",
    "4OweikRF7bg",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create call extractor train/eval split config from S3 labels.")
    parser.add_argument(
        "--label-prefix",
        type=str,
        default="labeling/corrected_boundaries/v2/",
        help="S3 prefix containing {video_id}.json label files",
    )
    parser.add_argument(
        "--audio-prefix",
        action="append",
        default=["audio/pretraining/", "audio/expired_listing/", "audio/"],
        help="S3 prefix to scan for MP3s (repeatable)",
    )
    parser.add_argument(
        "--eval-video-id",
        action="append",
        default=None,
        help="Override eval list (repeatable). If omitted, use the built-in v1 list.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("configs/call_extractor/split_v2.config.json"),
        help="Output path (must end with .config.json to be tracked)",
    )
    parser.add_argument("--include-missing-audio", action="store_true", help="Keep videos even if audio not found")
    args = parser.parse_args()

    eval_ids = args.eval_video_id if args.eval_video_id else DEFAULT_EVAL_VIDEO_IDS
    eval_ids = list(dict.fromkeys(eval_ids))

    label_keys = [k for k in s3_list_keys(S3_BUCKET, args.label_prefix, region=AWS_REGION) if k.endswith(".json")]
    video_ids = sorted([Path(k).stem for k in label_keys])
    logger.info(f"Found {len(video_ids)} label videos under s3://{S3_BUCKET}/{args.label_prefix}")

    audio_index = build_audio_index(S3_BUCKET, args.audio_prefix, region=AWS_REGION)
    missing_audio = sorted([vid for vid in video_ids if vid not in audio_index])
    if missing_audio:
        logger.warning(f"Missing audio for {len(missing_audio)} labeled videos (example: {missing_audio[:5]})")

    excluded: list[str] = []
    if not args.include_missing_audio:
        excluded = missing_audio
        video_ids = [vid for vid in video_ids if vid not in set(excluded)]

    eval_set = [vid for vid in eval_ids if vid in set(video_ids)]
    missing_eval = [vid for vid in eval_ids if vid not in set(video_ids)]
    if missing_eval:
        logger.warning(f"Eval IDs missing from label set (will be ignored): {missing_eval}")

    train_set = sorted([vid for vid in video_ids if vid not in set(eval_set)])

    out: Dict[str, Any] = {
        "version": "split_v2",
        "generated_at_utc": utc_now_compact(),
        "bucket": S3_BUCKET,
        "aws_region": AWS_REGION,
        "label_prefix": args.label_prefix,
        "audio_prefixes": args.audio_prefix,
        "eval_video_ids": eval_set,
        "train_video_ids": train_set,
        "excluded_video_ids": excluded,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    logger.info(f"Wrote {args.out} (train={len(train_set)}, eval={len(eval_set)}, excluded={len(excluded)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
