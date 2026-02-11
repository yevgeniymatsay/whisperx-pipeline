#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from pipeline.call_extractor_wavlm.audio_cache import ensure_flac_cached, read_flac_segment_float32
from pipeline.call_extractor_wavlm.io import (
    parse_video_id_from_mp3_key,
    s3_download_if_missing,
    s3_list_keys,
    s3_read_json,
    utc_now_compact,
)
from pipeline.config import AWS_REGION, S3_BUCKET

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClipSpec:
    video_id: str
    call_index: int
    start_s: float
    end_s: float


def _read_video_ids(path: Path) -> list[str]:
    raw = path.read_text().splitlines()
    video_ids = [line.strip() for line in raw if line.strip() and not line.strip().startswith("#")]
    # De-dup while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for vid in video_ids:
        if vid in seen:
            continue
        out.append(vid)
        seen.add(vid)
    return out


def _resolve_mp3_keys_for_video_ids(
    *,
    bucket: str,
    audio_prefixes: Iterable[str],
    video_ids: list[str],
) -> dict[str, str]:
    remaining = set(video_ids)
    resolved: dict[str, str] = {}
    for prefix in audio_prefixes:
        logger.info(f"Scanning S3 for MP3 keys under prefix={prefix!r} (need {len(remaining)} video_ids)")
        for key in s3_list_keys(bucket, prefix, region=AWS_REGION):
            if not key.endswith(".mp3"):
                continue
            vid = parse_video_id_from_mp3_key(key)
            if not vid or vid not in remaining:
                continue
            resolved[vid] = key
            remaining.remove(vid)
            logger.info(f"Resolved audio: {vid} -> s3://{bucket}/{key}")
            if not remaining:
                return resolved
    return resolved


def _load_clip_specs(
    *,
    bucket: str,
    labels_prefix: str,
    video_ids: list[str],
) -> list[ClipSpec]:
    specs: list[ClipSpec] = []
    for vid in video_ids:
        key = f"{labels_prefix.rstrip('/')}/{vid}.json"
        data = s3_read_json(bucket, key, region=AWS_REGION)
        boundaries = data.get("boundaries") or []
        for b in boundaries:
            start_s = float(b["start_s"])
            end_s = float(b["end_s"])
            call_index = int(b.get("call_index", len(specs)))
            if end_s <= start_s:
                logger.warning(f"Skipping invalid boundary {vid} call_index={call_index}: start={start_s} end={end_s}")
                continue
            specs.append(ClipSpec(video_id=vid, call_index=call_index, start_s=start_s, end_s=end_s))
    specs.sort(key=lambda s: (s.video_id, s.call_index, s.start_s, s.end_s))
    return specs


def _select_clip_specs(
    specs: list[ClipSpec],
    *,
    max_clips: Optional[int],
    seed: int,
) -> list[ClipSpec]:
    if max_clips is None or len(specs) <= int(max_clips):
        return specs
    rng = random.Random(int(seed))
    idxs = list(range(len(specs)))
    rng.shuffle(idxs)
    picked = sorted(idxs[: int(max_clips)])
    return [specs[i] for i in picked]


def _write_wav_mono_16k(path: Path, audio_f32: np.ndarray, *, sr_hz: int = 16000) -> None:
    try:
        import soundfile as sf
    except Exception as e:  # pragma: no cover
        raise RuntimeError("soundfile is required: pip install soundfile") from e

    path.parent.mkdir(parents=True, exist_ok=True)
    audio_f32 = np.asarray(audio_f32, dtype=np.float32)
    if audio_f32.ndim != 1:
        raise ValueError(f"Expected mono audio (1D), got shape={audio_f32.shape}")
    sf.write(str(path), audio_f32, int(sr_hz), format="WAV", subtype="PCM_16")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build per-call audio clips (WAV 16k mono) from manual boundaries.")
    parser.add_argument(
        "--video-ids-file",
        type=Path,
        default=Path("configs/diarization_pilot/video_ids_5.txt"),
        help="Newline-delimited list of video IDs to include",
    )
    parser.add_argument(
        "--labels-prefix",
        type=str,
        default="labeling/corrected_boundaries/v2/",
        help="S3 prefix for manual boundary JSON files",
    )
    parser.add_argument(
        "--audio-prefix",
        action="append",
        default=["audio/pretraining/"],
        help="S3 prefix(es) to scan for MP3s (repeatable). Default: audio/pretraining/",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=35,
        help="Max number of call clips to build across all videos (default: 35). Use 0 for unlimited.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Random seed used when --max-clips limits selection")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: artifacts/diarization_pilot/pilot_{utc})",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/call_extractor_wavlm"),
        help="Cache dir for MP3/FLAC (default: .cache/call_extractor_wavlm)",
    )
    args = parser.parse_args()

    video_ids = _read_video_ids(Path(args.video_ids_file))
    if not video_ids:
        raise SystemExit(f"No video_ids found in {args.video_ids_file}")

    max_clips = None if int(args.max_clips) == 0 else int(args.max_clips)
    out_dir = Path(args.out_dir) if args.out_dir else Path("artifacts/diarization_pilot") / f"pilot_{utc_now_compact()}"
    clips_dir = out_dir / "clips"
    meta_path = out_dir / "metadata.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    specs_all = _load_clip_specs(bucket=S3_BUCKET, labels_prefix=str(args.labels_prefix), video_ids=video_ids)
    if not specs_all:
        raise SystemExit("No boundaries found for requested videos (did you pick no-call videos?)")

    specs = _select_clip_specs(specs_all, max_clips=max_clips, seed=int(args.seed))
    logger.info(f"Selected {len(specs)} clips (from {len(specs_all)} total boundaries across {len(video_ids)} videos)")

    resolved = _resolve_mp3_keys_for_video_ids(bucket=S3_BUCKET, audio_prefixes=args.audio_prefix, video_ids=sorted({s.video_id for s in specs}))
    missing_audio = sorted({s.video_id for s in specs if s.video_id not in resolved})
    if missing_audio:
        raise SystemExit(f"Missing audio MP3 keys for video_ids: {missing_audio}")

    mp3_dir = Path(args.cache_dir) / "audio_mp3"
    flac_dir = Path(args.cache_dir) / "audio_flac"
    mp3_dir.mkdir(parents=True, exist_ok=True)
    flac_dir.mkdir(parents=True, exist_ok=True)

    meta_f = meta_path.open("w", encoding="utf-8")
    try:
        for i, spec in enumerate(specs):
            mp3_key = resolved[spec.video_id]
            mp3_path = mp3_dir / f"{spec.video_id}.mp3"
            flac_path = flac_dir / f"{spec.video_id}.flac"

            s3_download_if_missing(S3_BUCKET, mp3_key, mp3_path, region=AWS_REGION)
            ensure_flac_cached(mp3_path=mp3_path, flac_path=flac_path, sr_hz=16000)

            clip_id = f"{spec.video_id}_call{spec.call_index:03d}"
            clip_path = clips_dir / f"{clip_id}.wav"
            duration_s = float(spec.end_s - spec.start_s)
            audio = read_flac_segment_float32(flac_path, start_s=float(spec.start_s), duration_s=float(duration_s), sr_hz=16000)
            _write_wav_mono_16k(clip_path, audio, sr_hz=16000)

            row = {
                "clip_id": clip_id,
                "video_id": spec.video_id,
                "call_index": int(spec.call_index),
                "offset_s": float(spec.start_s),
                "clip_duration_s": float(duration_s),
                "clip_path": str(clip_path),
                "labels_s3_key": f"{str(args.labels_prefix).rstrip('/')}/{spec.video_id}.json",
                "audio_s3_key": mp3_key,
            }
            meta_f.write(json.dumps(row) + "\n")
            if (i + 1) % 10 == 0 or (i + 1) == len(specs):
                logger.info(f"Wrote {i+1}/{len(specs)} clips -> {clips_dir}")
    finally:
        meta_f.close()

    logger.info(f"Done. Output: {out_dir}")
    logger.info(f"Metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

