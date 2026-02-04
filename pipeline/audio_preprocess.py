# scripts/whisperx_pipeline/audio_preprocess.py
"""Audio preprocessing: convert to 16kHz mono WAV."""
import subprocess
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


def load_audio(path: str) -> np.ndarray:
    """Load audio file as numpy array."""
    data, sr = sf.read(path, dtype="float32")
    if sr != 16000:
        raise ValueError(f"Expected 16kHz, got {sr}Hz")
    return data
