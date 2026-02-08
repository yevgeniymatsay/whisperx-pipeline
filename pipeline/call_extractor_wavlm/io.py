from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from hashlib import sha1
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

import boto3
import numpy as np

from ..config import AWS_REGION

logger = logging.getLogger(__name__)

_YT_ID_RE = re.compile(r" - ([A-Za-z0-9_-]{11})\.mp3$")


@dataclass(frozen=True)
class S3Path:
    bucket: str
    key: str


def get_s3(region: str = AWS_REGION):
    return boto3.client("s3", region_name=region)


def s3_list_keys(bucket: str, prefix: str, *, region: str = AWS_REGION) -> Iterator[str]:
    s3 = get_s3(region=region)
    continuation: Optional[str] = None
    while True:
        kwargs: Dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if continuation:
            kwargs["ContinuationToken"] = continuation
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []) or []:
            yield obj["Key"]
        if resp.get("IsTruncated"):
            continuation = resp.get("NextContinuationToken")
        else:
            break


def s3_read_json(bucket: str, key: str, *, region: str = AWS_REGION) -> Dict[str, Any]:
    s3 = get_s3(region=region)
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    return json.loads(body)


def s3_download_if_missing(
    bucket: str,
    key: str,
    dst_path: Path,
    *,
    region: str = AWS_REGION,
) -> Path:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        return dst_path
    s3 = get_s3(region=region)
    s3.download_file(bucket, key, str(dst_path))
    return dst_path


def s3_upload_file(
    *,
    bucket: str,
    key: str,
    src_path: Path,
    region: str = AWS_REGION,
) -> None:
    s3 = get_s3(region=region)
    s3.upload_file(str(src_path), bucket, key)


def s3_upload_directory(
    *,
    bucket: str,
    prefix: str,
    src_dir: Path,
    region: str = AWS_REGION,
) -> None:
    s3 = get_s3(region=region)
    src_dir = src_dir.resolve()
    for path in src_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src_dir).as_posix()
        key = f"{prefix.rstrip('/')}/{rel}"
        s3.upload_file(str(path), bucket, key)


def parse_video_id_from_mp3_key(key: str) -> Optional[str]:
    m = _YT_ID_RE.search(key)
    if not m:
        return None
    return m.group(1)


def build_audio_index(
    bucket: str,
    audio_prefixes: Iterable[str],
    *,
    region: str = AWS_REGION,
) -> dict[str, str]:
    """Build video_id -> s3_key index by scanning known audio prefixes once."""
    index: dict[str, str] = {}
    for prefix in audio_prefixes:
        for key in s3_list_keys(bucket, prefix, region=region):
            if not key.endswith(".mp3"):
                continue
            video_id = parse_video_id_from_mp3_key(key)
            if not video_id:
                continue
            if video_id in index and index[video_id] != key:
                logger.warning(f"Multiple mp3 keys for {video_id}: {index[video_id]} vs {key} (keeping first)")
                continue
            index[video_id] = key
    return index


def resolve_audio_key(video_id: str, audio_index: dict[str, str]) -> Optional[str]:
    return audio_index.get(video_id)


def ffprobe_duration_s(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(proc.stdout.strip())


def decode_audio_segment_to_float32(
    input_path: Path,
    *,
    start_s: float,
    duration_s: float,
    sr_hz: int = 16000,
) -> np.ndarray:
    if duration_s <= 0:
        raise ValueError(f"duration_s must be > 0 (got {duration_s})")
    if start_s < 0:
        raise ValueError(f"start_s must be >= 0 (got {start_s})")

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(input_path),
        "-ss",
        str(float(start_s)),
        "-t",
        str(float(duration_s)),
        "-ar",
        str(int(sr_hz)),
        "-ac",
        "1",
        "-f",
        "f32le",
        "-",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True)
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f"Decoded 0 samples from {input_path} @ start={start_s} dur={duration_s}")
    return audio


def write_flac_segment(
    input_path: Path,
    *,
    start_s: float,
    end_s: float,
    output_path: Path,
    sr_hz: int = 16000,
) -> None:
    if end_s <= start_s:
        raise ValueError(f"end_s must be > start_s (got start_s={start_s}, end_s={end_s})")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_s = float(end_s - start_s)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(input_path),
        "-ss",
        str(float(start_s)),
        "-t",
        str(duration_s),
        "-ar",
        str(int(sr_hz)),
        "-ac",
        "1",
        "-c:a",
        "flac",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def git_short_sha(repo_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip()
    except Exception:
        return "unknown"


def utc_now_compact() -> str:
    import datetime as _dt

    return _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


_CACHE_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def cache_key_for_s3_prefix(prefix: str) -> str:
    """Return a short, filesystem-safe key for caching artifacts per S3 prefix.

    This prevents cross-run contamination when multiple runs write `{video_id}.npz/json`
    under a shared local directory.
    """
    p = str(prefix).strip("/")
    if p == "":
        return "root"
    tail = p.split("/")[-1] or "prefix"
    tail = _CACHE_SAFE_RE.sub("_", tail).strip("_") or "prefix"
    h = sha1(p.encode("utf-8")).hexdigest()[:10]
    return f"{tail}_{h}"
