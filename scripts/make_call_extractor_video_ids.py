#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import AWS_REGION, S3_BUCKET
from pipeline.call_extractor_wavlm.io import build_audio_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="List video_ids from S3 MP3 keys for batch call extraction.")
    parser.add_argument(
        "--audio-prefix",
        action="append",
        default=["audio/pretraining/"],
        help="S3 prefix to scan for MP3s (repeatable). Default: audio/pretraining/",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/call_extractor/wavlm_large_v1/video_ids_all.txt"),
        help="Output newline-delimited video_id file",
    )
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of video_ids to write")
    parser.add_argument("--num-shards", type=int, default=1, help="Shard count (>=1)")
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index in [0, num-shards)")
    args = parser.parse_args()

    if int(args.num_shards) < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not (0 <= int(args.shard_index) < int(args.num_shards)):
        raise SystemExit("--shard-index must be in [0, num-shards)")

    audio_index = build_audio_index(S3_BUCKET, args.audio_prefix, region=AWS_REGION)
    video_ids = list(audio_index.keys())
    logger.info(f"Found {len(video_ids)} unique video_ids across prefixes={args.audio_prefix}")

    if args.shuffle:
        rng = random.Random(int(args.seed))
        rng.shuffle(video_ids)
    else:
        video_ids.sort()

    if int(args.num_shards) > 1:
        video_ids = [vid for i, vid in enumerate(video_ids) if (i % int(args.num_shards)) == int(args.shard_index)]
        logger.info(f"Shard {args.shard_index}/{args.num_shards}: {len(video_ids)} video_ids")

    if args.limit is not None:
        video_ids = video_ids[: int(args.limit)]
        logger.info(f"Limit: {len(video_ids)} video_ids")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(video_ids) + "\n")
    logger.info(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

