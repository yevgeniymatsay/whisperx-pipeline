#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import AWS_REGION, S3_BUCKET
from pipeline.call_extractor_wavlm.decode import DecodeConfig, probabilities_to_segments
from pipeline.call_extractor_wavlm.io import cache_key_for_s3_prefix, s3_download_if_missing, s3_upload_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_decode_cfg(path: Path) -> DecodeConfig:
    return DecodeConfig(**_load_json(path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Decode call segments from saved per-frame probs (*.npz).")
    parser.add_argument("--split-config", type=Path, default=Path("configs/call_extractor/split_v2.config.json"))
    parser.add_argument("--output-config", type=Path, default=Path("configs/call_extractor/output_wavlm_large_v1.config.json"))
    parser.add_argument("--decode-config", type=Path, default=Path("configs/call_extractor/decode_wavlm_large_v1.config.json"))
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default=None,
        help="Override S3 prefix for downloads/uploads (default: output-config's s3_output_prefix)",
    )
    parser.add_argument(
        "--probs-dir",
        type=Path,
        default=None,
        help="Local dir containing {video_id}.npz (default: <local_artifacts_dir>/predictions)",
    )
    parser.add_argument(
        "--segments-dir",
        type=Path,
        default=None,
        help="Local dir to write {video_id}.json (default: <local_artifacts_dir>/predictions)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=["eval", "train", "all"],
        default="eval",
        help="Which split videos to process (default: eval)",
    )
    parser.add_argument("--upload", action=argparse.BooleanOptionalAction, default=True, help="Upload segments to S3")
    args = parser.parse_args()

    split_cfg = _load_json(args.split_config)
    out_cfg = _load_json(args.output_config)
    decode_cfg = _load_decode_cfg(args.decode_config)

    s3_prefix = str(args.s3_prefix or out_cfg.get("s3_output_prefix", "call_extractor/wavlm_large_v1/")).rstrip("/")

    probs_dir = args.probs_dir
    segments_dir = args.segments_dir
    if probs_dir is None or segments_dir is None:
        cache_key = cache_key_for_s3_prefix(s3_prefix)
        base = Path(out_cfg.get("local_artifacts_dir", "artifacts/call_extractor/wavlm_large_v1")) / "predictions" / cache_key
        probs_dir = probs_dir or base
        segments_dir = segments_dir or base

    probs_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)

    if args.subset == "eval":
        video_ids = list(split_cfg["eval_video_ids"])
    elif args.subset == "train":
        video_ids = list(split_cfg["train_video_ids"])
    else:
        video_ids = list(split_cfg["eval_video_ids"]) + list(split_cfg["train_video_ids"])

    def load_probs(vid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        local = probs_dir / f"{vid}.npz"
        if not local.exists():
            s3_key = f"{s3_prefix}/probs/{vid}.npz"
            s3_download_if_missing(S3_BUCKET, s3_key, local, region=AWS_REGION)
        arr = np.load(local)
        return arr["times_s"], arr["in_call"], arr["start"], arr["end"]

    for vid in video_ids:
        vid = str(vid)
        times_s, in_call, start, end = load_probs(vid)
        segs = probabilities_to_segments(times_s=times_s, in_call_p=in_call, start_p=start, end_p=end, cfg=decode_cfg)

        out = {
            "video_id": vid,
            "segments": segs,
            "decode_config": decode_cfg.__dict__,
        }
        local_out = segments_dir / f"{vid}.json"
        local_out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

        if args.upload:
            s3_key = f"{s3_prefix}/segments/{vid}.json"
            s3_upload_file(bucket=S3_BUCKET, key=s3_key, src_path=local_out, region=AWS_REGION)
        logger.info(f"{vid}: segments={len(segs)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
