# tests/whisperx_pipeline/test_transcriber.py
import pytest
from pipeline.transcriber import WhisperXTranscriber, WordOutput
from pipeline.config import WhisperXConfig


def test_word_output_dataclass():
    """WordOutput has required fields."""
    word = WordOutput(
        i=0,
        t0=0.5,
        t1=0.8,
        t0_abs=100.5,
        t1_abs=100.8,
        text="Hello",
        text_norm="hello",
        spk="SPEAKER_00"
    )
    assert word.text_norm == "hello"


def test_transcriber_config():
    """Transcriber accepts config."""
    config = WhisperXConfig(model="large-v3", batch_size=8)
    transcriber = WhisperXTranscriber(config)
    assert transcriber.config.model == "large-v3"


# Integration test (requires GPU, skip in CI)
@pytest.mark.skip(reason="Requires GPU and model download")
def test_transcriber_produces_words():
    """Transcriber produces word list with timestamps."""
    config = WhisperXConfig(model="large-v3")
    transcriber = WhisperXTranscriber(config)
    # Would test with actual audio
