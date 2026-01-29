# tests/whisperx_pipeline/test_vad_chunker.py
import pytest
from pipeline.vad_chunker import VADChunker, Chunk
from pipeline.config import VADConfig


def test_chunk_dataclass():
    """Chunk has required fields."""
    chunk = Chunk(
        chunk_id="video_0_30000",
        start_ms=0,
        end_ms=30000,
        start_s=0.0,
        end_s=30.0,
        duration_s=30.0
    )
    assert chunk.chunk_id == "video_0_30000"
    assert chunk.duration_s == 30.0


def test_chunker_respects_min_max():
    """Chunks respect min/max duration."""
    config = VADConfig(min_chunk_s=30.0, max_chunk_s=60.0)
    chunker = VADChunker(config)

    # Mock: 3 speech segments with gaps
    speech_segments = [
        {"start": 0.0, "end": 25.0},   # First speech
        {"start": 28.0, "end": 55.0},  # Gap of 3s (>2s threshold)
        {"start": 58.0, "end": 120.0}, # Long segment
    ]

    chunks = chunker.compute_chunks(
        speech_segments=speech_segments,
        total_duration_s=120.0,
        video_id="test"
    )

    # Should merge first two (gap < min_chunk), split long one
    for chunk in chunks:
        assert chunk.duration_s >= 20.0  # Allow some slack for padding
        assert chunk.duration_s <= 65.0  # max + padding


def test_chunker_adds_padding():
    """Chunks have padding around boundaries."""
    config = VADConfig(padding_s=0.5, gap_threshold_s=2.0, min_chunk_s=10.0, max_chunk_s=600.0)
    chunker = VADChunker(config)

    speech_segments = [
        {"start": 1.0, "end": 10.0},
        {"start": 15.0, "end": 25.0},  # Gap of 5s
    ]

    chunks = chunker.compute_chunks(speech_segments, 30.0, "test")

    # First chunk should start at 0.5 (1.0 - 0.5 padding)
    assert chunks[0].start_s == 0.5
