# tests/whisperx_pipeline/test_pipeline.py
import pytest
from pipeline.pipeline import WhisperXPipeline
from pipeline.config import PipelineConfig


def test_pipeline_config():
    """Pipeline accepts config."""
    config = PipelineConfig()
    pipeline = WhisperXPipeline(config)
    assert pipeline.config.whisperx.model == "large-v3"


def test_pipeline_generates_run_id():
    """Pipeline generates unique run IDs."""
    config = PipelineConfig()
    pipeline = WhisperXPipeline(config)

    run_id_1 = pipeline._generate_run_id()
    run_id_2 = pipeline._generate_run_id()

    assert run_id_1 != run_id_2
    assert len(run_id_1) > 10  # Timestamp format
