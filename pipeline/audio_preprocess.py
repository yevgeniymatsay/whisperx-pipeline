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


def load_audio(path: str) -> np.ndarray:
    """Load audio file as numpy array."""
    data, sr = sf.read(path, dtype="float32")
    if sr != 16000:
        raise ValueError(f"Expected 16kHz, got {sr}Hz")
    return data
