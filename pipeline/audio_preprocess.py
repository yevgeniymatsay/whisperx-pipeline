# scripts/whisperx_pipeline/audio_preprocess.py
"""Audio preprocessing: convert to 16kHz mono WAV."""
import subprocess
from typing import BinaryIO, Optional
import threading
import numpy as np
import soundfile as sf


def convert_to_wav(input_path: str, output_path: str) -> None:
    """Convert audio file to 16kHz mono WAV using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def decode_audio_to_float32(input_path: str, sr_hz: int = 16000) -> np.ndarray:
    """Decode an audio file to mono float32 PCM using ffmpeg.

    This avoids writing large intermediate WAV files to disk, which is important
    for long videos and low disk-space environments.
    """
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        input_path,
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
    # ffmpeg can succeed but still output nothing for malformed inputs.
    if audio.size == 0:
        raise ValueError(f"Decoded 0 samples from {input_path}")
    return audio


def decode_audio_stream_to_float32(
    stream: BinaryIO,
    *,
    sr_hz: int = 16000,
    max_duration_s: Optional[float] = None,
) -> np.ndarray:
    """Decode an audio stream (e.g., S3 StreamingBody) to mono float32 PCM using ffmpeg.

    This avoids writing the source audio to disk. The decoded PCM is still held in
    memory as a numpy array, which is required for the call segmenter features.
    """
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        "pipe:0",
    ]
    if max_duration_s is not None:
        cmd += ["-t", str(float(max_duration_s))]
    cmd += [
        "-ar",
        str(int(sr_hz)),
        "-ac",
        "1",
        "-f",
        "f32le",
        "-",
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None

    out_chunks: list[bytes] = []
    err_chunks: list[bytes] = []

    def _drain(src, dst: list[bytes]) -> None:
        while True:
            buf = src.read(1024 * 1024)
            if not buf:
                break
            dst.append(buf)

    t_out = threading.Thread(target=_drain, args=(proc.stdout, out_chunks), daemon=True)
    t_err = threading.Thread(target=_drain, args=(proc.stderr, err_chunks), daemon=True)
    t_out.start()
    t_err.start()

    try:
        while True:
            if proc.poll() is not None:
                break
            chunk = stream.read(1024 * 1024)  # 1 MiB
            if not chunk:
                break
            try:
                proc.stdin.write(chunk)
            except BrokenPipeError:
                # ffmpeg exited early (e.g., due to -t); stop feeding input.
                break
        try:
            proc.stdin.close()
        except Exception:
            pass

        ret = proc.wait()
    finally:
        try:
            stream.close()
        except Exception:
            pass

    t_out.join(timeout=30)
    t_err.join(timeout=30)
    out = b"".join(out_chunks)
    err = b"".join(err_chunks)

    if ret != 0:
        raise RuntimeError(f"ffmpeg failed (exit={ret}): {err.decode('utf-8', errors='ignore')[:400]}")

    audio = np.frombuffer(out, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("Decoded 0 samples from stream")
    return audio


def load_audio(path: str) -> np.ndarray:
    """Load audio file as numpy array."""
    data, sr = sf.read(path, dtype="float32")
    if sr != 16000:
        raise ValueError(f"Expected 16kHz, got {sr}Hz")
    return data
