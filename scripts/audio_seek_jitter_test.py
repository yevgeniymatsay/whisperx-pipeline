#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import AWS_REGION, S3_BUCKET
from pipeline.call_extractor_wavlm.audio_cache import ensure_flac_cached, read_flac_segment_float32
from pipeline.call_extractor_wavlm.io import (
    build_audio_index,
    decode_audio_segment_to_float32,
    ffprobe_duration_s,
    s3_download_if_missing,
    s3_upload_file,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _next_pow2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def _xcorr_lag_samples(a: np.ndarray, b: np.ndarray, *, max_lag_samples: int) -> int:
    """Return lag (in samples) maximizing normalized cross-correlation within +/- max_lag_samples.

    Positive lag means `a` is best aligned to `b` when shifting `a` right (i.e., `a` occurs later).
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size != b.size or a.size == 0:
        return 0

    a = a - float(a.mean())
    b = b - float(b.mean())
    a_std = float(a.std())
    b_std = float(b.std())
    if a_std == 0.0 or b_std == 0.0:
        return 0
    a = a / a_std
    b = b / b_std

    L = int(a.size)
    n = _next_pow2(2 * L - 1)
    A = np.fft.rfft(a, n)
    B = np.fft.rfft(b, n)
    corr = np.fft.irfft(A * np.conj(B), n)
    # Convert circular to linear full correlation (lags -(L-1)..(L-1))
    corr_full = np.concatenate([corr[-(L - 1) :], corr[:L]])
    lags = np.arange(-(L - 1), L, dtype=np.int64)

    m = int(max_lag_samples)
    if m <= 0:
        return 0
    mask = (lags >= -m) & (lags <= m)
    if not mask.any():
        return 0
    idx = int(np.argmax(corr_full[mask]))
    return int(lags[mask][idx])


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantify MP3 chunk-seek jitter vs FLAC sample-accurate slicing.")
    parser.add_argument("--video-id", type=str, required=True)
    parser.add_argument("--split-config", type=Path, default=Path("configs/call_extractor/split_v1.config.json"))
    parser.add_argument("--output-config", type=Path, default=Path("configs/call_extractor/output_wavlm_large_v1.config.json"))
    parser.add_argument("--sr-hz", type=int, default=16000)
    parser.add_argument("--dur-s", type=float, default=10.0)
    parser.add_argument("--n-segments", type=int, default=20)
    parser.add_argument("--max-lag-ms", type=float, default=200.0)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write report JSON here (default: <local_artifacts_dir>/seek_jitter/{video_id}.json)",
    )
    parser.add_argument("--upload", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    split_cfg = _load_json(args.split_config)
    out_cfg = _load_json(args.output_config)
    video_id = str(args.video_id)

    cache_dir = Path(out_cfg.get("local_cache_dir", ".cache/call_extractor_wavlm"))
    mp3_cache_dir = cache_dir / "audio"
    flac_cache_dir = cache_dir / "audio_flac"
    mp3_cache_dir.mkdir(parents=True, exist_ok=True)
    flac_cache_dir.mkdir(parents=True, exist_ok=True)

    audio_index = build_audio_index(S3_BUCKET, list(split_cfg["audio_prefixes"]), region=AWS_REGION)
    audio_key = audio_index.get(video_id)
    if not audio_key:
        raise RuntimeError(f"Audio not found for video_id={video_id}")

    mp3_path = mp3_cache_dir / f"{video_id}.mp3"
    s3_download_if_missing(S3_BUCKET, audio_key, mp3_path, region=AWS_REGION)
    flac_path = flac_cache_dir / f"{video_id}.flac"
    ensure_flac_cached(mp3_path=mp3_path, flac_path=flac_path, sr_hz=int(args.sr_hz))

    duration_s = float(ffprobe_duration_s(flac_path))
    seg_dur_s = float(args.dur_s)
    if duration_s <= seg_dur_s + 0.5:
        raise RuntimeError(f"Audio too short for test: duration_s={duration_s} dur_s={seg_dur_s}")

    rng = np.random.default_rng(1337 ^ (hash(video_id) & 0xFFFF_FFFF))
    max_start = float(duration_s) - float(seg_dur_s) - 0.1
    starts = rng.uniform(0.0, max_start, size=int(args.n_segments)).astype(np.float64)
    starts.sort()

    max_lag_samples = int(math.ceil(float(args.max_lag_ms) * float(args.sr_hz) / 1000.0))
    rows: list[dict[str, Any]] = []
    for s in starts.tolist():
        a = decode_audio_segment_to_float32(mp3_path, start_s=float(s), duration_s=float(seg_dur_s), sr_hz=int(args.sr_hz))
        b = read_flac_segment_float32(flac_path, start_s=float(s), duration_s=float(seg_dur_s), sr_hz=int(args.sr_hz))
        L = min(int(a.size), int(b.size))
        a = a[:L]
        b = b[:L]
        lag = _xcorr_lag_samples(a, b, max_lag_samples=max_lag_samples)
        rows.append(
            {
                "start_s": float(s),
                "lag_samples": int(lag),
                "lag_ms": float(1000.0 * float(lag) / float(args.sr_hz)),
            }
        )

    lags_ms = np.array([r["lag_ms"] for r in rows], dtype=np.float64)
    report = {
        "video_id": video_id,
        "audio_s3_key": str(audio_key),
        "sr_hz": int(args.sr_hz),
        "dur_s": float(seg_dur_s),
        "n_segments": int(args.n_segments),
        "max_lag_ms": float(args.max_lag_ms),
        "summary": {
            "median_lag_ms": float(np.median(lags_ms)),
            "mean_lag_ms": float(lags_ms.mean()),
            "std_lag_ms": float(lags_ms.std()),
            "max_abs_lag_ms": float(np.max(np.abs(lags_ms))),
        },
        "samples": rows,
    }

    if args.out_json is None:
        out_dir = Path(out_cfg.get("local_artifacts_dir", "artifacts/call_extractor/wavlm_large_v1")) / "seek_jitter"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{video_id}.json"
    else:
        out_path = args.out_json
        out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    logger.info(f"Wrote {out_path}")

    if args.upload:
        s3_prefix = str(out_cfg.get("s3_output_prefix", "call_extractor/wavlm_large_v1/")).rstrip("/")
        dst_key = f"{s3_prefix}/diagnostics/seek_jitter/{video_id}.json"
        s3_upload_file(bucket=S3_BUCKET, key=dst_key, src_path=out_path, region=AWS_REGION)
        logger.info(f"Uploaded s3://{S3_BUCKET}/{dst_key}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
