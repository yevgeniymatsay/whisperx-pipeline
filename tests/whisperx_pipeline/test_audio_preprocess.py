# tests/whisperx_pipeline/test_audio_preprocess.py
import pytest
import tempfile
import numpy as np
import soundfile as sf
from pipeline.audio_preprocess import convert_to_wav, load_audio


def test_convert_to_wav_creates_16k_mono():
    """Convert MP3 to 16kHz mono WAV."""
    # Create a test WAV (we'll test with WAV since MP3 needs encoding)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        # 1 second of silence at 44.1kHz stereo
        sr = 44100
        audio = np.zeros((sr, 2), dtype=np.float32)
        sf.write(f.name, audio, sr)
        input_path = f.name

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        output_path = f.name

    convert_to_wav(input_path, output_path)

    # Verify output is 16kHz mono
    data, sr = sf.read(output_path)
    assert sr == 16000
    assert len(data.shape) == 1  # mono


def test_load_audio_returns_numpy_array():
    """Load audio returns numpy array at 16kHz."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sr = 16000
        audio = np.random.randn(sr).astype(np.float32)
        sf.write(f.name, audio, sr)
        path = f.name

    result = load_audio(path)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
