#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import AWS_REGION, S3_BUCKET
from pipeline.call_extractor_wavlm.audio_cache import ensure_flac_cached, write_flac_segment_from_cached_flac
from pipeline.call_extractor_wavlm.io import cache_key_for_s3_prefix, build_audio_index, s3_download_if_missing, s3_upload_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract predicted call clips as FLAC 16kHz mono and upload to S3.")
    parser.add_argument("--split-config", type=Path, default=Path("configs/call_extractor/split_v1.config.json"))
    parser.add_argument("--output-config", type=Path, default=Path("configs/call_extractor/output_wavlm_large_v1.config.json"))
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default=None,
        help="Override S3 prefix for downloading segments and uploading clips (default: output-config's s3_output_prefix)",
    )
    parser.add_argument(
        "--segments-dir",
        type=Path,
        default=None,
        help="Local directory containing {video_id}.json (default: <local_artifacts_dir>/predictions)",
    )
    parser.add_argument(
        "--video-ids-file",
        type=Path,
        default=None,
        help="Optional newline-delimited video_id list to process (overrides --subset/split-config lists)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=["eval", "train", "all"],
        default="eval",
        help="Which split videos to process (default: eval)",
    )
    parser.add_argument("--upload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-jsonl", type=Path, default=Path("artifacts/call_extractor/wavlm_large_v1/extracted_calls.jsonl"))
    args = parser.parse_args()

    split_cfg = _load_json(args.split_config)
    out_cfg = _load_json(args.output_config)
    s3_prefix = str(args.s3_prefix or out_cfg.get("s3_output_prefix", "call_extractor/wavlm_large_v1/")).rstrip("/")

    cache_dir = Path(out_cfg.get("local_cache_dir", ".cache/call_extractor_wavlm"))
    audio_cache_dir = cache_dir / "audio"
    flac_cache_dir = cache_dir / "audio_flac"
    audio_cache_dir.mkdir(parents=True, exist_ok=True)
    flac_cache_dir.mkdir(parents=True, exist_ok=True)

    segments_dir = args.segments_dir
    if segments_dir is None:
        cache_key = cache_key_for_s3_prefix(s3_prefix)
        segments_dir = (
            Path(out_cfg.get("local_artifacts_dir", "artifacts/call_extractor/wavlm_large_v1"))
            / "predictions"
            / cache_key
        )
    segments_dir.mkdir(parents=True, exist_ok=True)

    if args.video_ids_file:
        video_ids = [ln.strip() for ln in args.video_ids_file.read_text().splitlines() if ln.strip() != ""]
    else:
        if args.subset == "eval":
            video_ids = list(split_cfg["eval_video_ids"])
        elif args.subset == "train":
            video_ids = list(split_cfg["train_video_ids"])
        else:
            video_ids = list(split_cfg["eval_video_ids"]) + list(split_cfg["train_video_ids"])

    audio_index = build_audio_index(S3_BUCKET, list(split_cfg["audio_prefixes"]), region=AWS_REGION)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    f_out = args.out_jsonl.open("w")
    try:
        for idx, vid in enumerate(video_ids):
            vid = str(vid)
            audio_key = audio_index.get(vid)
            if not audio_key:
                logger.warning(f"[{idx+1}/{len(video_ids)}] Skip {vid}: audio not found")
                continue

            audio_path = audio_cache_dir / f"{vid}.mp3"
            s3_download_if_missing(S3_BUCKET, audio_key, audio_path, region=AWS_REGION)

            flac_path = flac_cache_dir / f"{vid}.flac"
            ensure_flac_cached(mp3_path=audio_path, flac_path=flac_path, sr_hz=16000)

            seg_path = segments_dir / f"{vid}.json"
            if not seg_path.exists():
                s3_key = f"{s3_prefix}/segments/{vid}.json"
                s3_download_if_missing(S3_BUCKET, s3_key, seg_path, region=AWS_REGION)

            seg_json = json.loads(seg_path.read_text())
            segments = seg_json.get("segments", [])
            logger.info(f"[{idx+1}/{len(video_ids)}] {vid}: segments={len(segments)}")

            for si, s in enumerate(segments):
                start_s = float(s["start_s"])
                end_s = float(s["end_s"])
                name = f"{si:04d}_{int(start_s*1000):010d}_{int(end_s*1000):010d}.flac"

                local_clip_dir = Path(out_cfg.get("local_artifacts_dir", "artifacts/call_extractor/wavlm_large_v1")) / "clips" / vid
                local_clip_path = local_clip_dir / name
                write_flac_segment_from_cached_flac(
                    flac_path,
                    start_s=float(start_s),
                    end_s=float(end_s),
                    output_path=local_clip_path,
                    sr_hz=16000,
                )

                clip_key = f"{s3_prefix}/clips/{vid}/{name}"
                if args.upload:
                    s3_upload_file(bucket=S3_BUCKET, key=clip_key, src_path=local_clip_path, region=AWS_REGION)

                rec = {
                    "video_id": vid,
                    "segment_index": int(si),
                    "start_s": float(start_s),
                    "end_s": float(end_s),
                    "audio_s3_key": str(audio_key),
                    "clip_s3_key": str(clip_key),
                }
                f_out.write(json.dumps(rec, sort_keys=True) + "\n")
    finally:
        f_out.close()

    logger.info(f"Wrote {args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
