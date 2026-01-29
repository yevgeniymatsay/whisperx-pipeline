# tests/whisperx_pipeline/test_chunk_writer.py
import pytest
import json
import tempfile
from pathlib import Path
from pipeline.chunk_writer import ChunkWriter
from pipeline.transcriber import WordOutput, DiarizationSegment
from pipeline.quality_metrics import QualityMetrics, QualityReason
from pipeline.config import PipelineConfig


def test_chunk_writer_creates_words_json():
    """ChunkWriter creates words.json with correct schema."""
    config = PipelineConfig()
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = ChunkWriter(base_path=tmpdir, config=config)

        words = [
            WordOutput(i=0, t0=0.5, t1=0.8, t0_abs=0.5, t1_abs=0.8,
                      text="Hello", text_norm="hello", spk="SPEAKER_00")
        ]
        segments = [
            DiarizationSegment(t0_abs=0.5, t1_abs=2.0, spk="SPEAKER_00")
        ]
        metrics = QualityMetrics(
            overlap_ratio=0.05, speaker_switches_per_min=20,
            median_turn_s=2.0, micro_turn_ratio=0.1,
            narrator_ratio=0.2, speaker_flip_suspected=False
        )

        writer.write_chunk(
            video_id="test_video",
            chunk_id="test_video_0_30000",
            chunk_time_offset_s=0.0,
            words=words,
            diarization_segments=segments,
            quality_metrics=metrics,
            quality_reasons=[],
            content_type="call_like",
            duration_s=30.0
        )

        # Verify files created (chunks/<chunk_id>/ per design)
        chunk_dir = Path(tmpdir) / "chunks" / "test_video_0_30000"
        assert (chunk_dir / "words.json").exists()
        assert (chunk_dir / "diarization_segments.json").exists()
        assert (chunk_dir / "chunk_metadata.json").exists()

        # Verify words.json schema
        with open(chunk_dir / "words.json") as f:
            data = json.load(f)
        assert data["video_id"] == "test_video"
        assert data["asr_model"] == "whisper-large-v3"
        assert len(data["words"]) == 1
        assert data["words"][0]["text_norm"] == "hello"
