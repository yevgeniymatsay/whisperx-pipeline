from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


def ensure_flac_cached(
    *,
    mp3_path: Path,
    flac_path: Path,
    sr_hz: int = 16000,
) -> Path:
    """Decode the full MP3 to a canonical 16kHz mono FLAC if missing.

    This avoids MP3 chunk-seek jitter during training/inference by enabling sample-accurate slicing from FLAC.
    """
    flac_path.parent.mkdir(parents=True, exist_ok=True)
    if flac_path.exists() and flac_path.stat().st_size > 0:
        return flac_path

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(mp3_path),
        "-vn",
        "-ar",
        str(int(sr_hz)),
        "-ac",
        "1",
        "-c:a",
        "flac",
        str(flac_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    if not flac_path.exists() or flac_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to create FLAC cache: {flac_path}")
    return flac_path


def read_flac_segment_float32(
    flac_path: Path,
    *,
    start_s: float,
    duration_s: float,
    sr_hz: int = 16000,
) -> np.ndarray:
    """Read a segment from a FLAC file using sample-accurate slicing."""
    if duration_s <= 0:
        raise ValueError(f"duration_s must be > 0 (got {duration_s})")
    if start_s < 0:
        raise ValueError(f"start_s must be >= 0 (got {start_s})")

    try:
        import soundfile as sf
    except Exception as e:  # pragma: no cover
        raise RuntimeError("soundfile is required for FLAC slicing: pip install soundfile") from e

    start_frame = int(round(float(start_s) * float(sr_hz)))
    n_frames = int(round(float(duration_s) * float(sr_hz)))
    if n_frames <= 0:
        raise ValueError(f"Requested 0 frames (start_s={start_s}, duration_s={duration_s}, sr_hz={sr_hz})")

    with sf.SoundFile(str(flac_path)) as f:
        if int(f.samplerate) != int(sr_hz):
            raise ValueError(f"FLAC sample rate mismatch: expected {sr_hz}, got {f.samplerate} ({flac_path})")
        if int(f.channels) != 1:
            raise ValueError(f"FLAC channel mismatch: expected 1, got {f.channels} ({flac_path})")

        start_frame = max(0, min(int(start_frame), int(f.frames)))
        f.seek(int(start_frame))
        audio = f.read(int(n_frames), dtype="float32", always_2d=False)

    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f"Decoded 0 samples from {flac_path} @ start={start_s} dur={duration_s}")
    return audio


def write_flac_segment_from_cached_flac(
    flac_path: Path,
    *,
    start_s: float,
    end_s: float,
    output_path: Path,
    sr_hz: int = 16000,
) -> None:
    """Write a FLAC clip using sample-accurate slicing from the cached FLAC."""
    if end_s <= start_s:
        raise ValueError(f"end_s must be > start_s (got start_s={start_s}, end_s={end_s})")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration_s = float(end_s - start_s)
    audio = read_flac_segment_float32(flac_path, start_s=float(start_s), duration_s=float(duration_s), sr_hz=int(sr_hz))

    try:
        import soundfile as sf
    except Exception as e:  # pragma: no cover
        raise RuntimeError("soundfile is required for FLAC writing: pip install soundfile") from e

    sf.write(str(output_path), audio, int(sr_hz), format="FLAC", subtype="PCM_16")
