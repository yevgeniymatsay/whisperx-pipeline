"""WhisperX pipeline configuration."""
import os
from dataclasses import dataclass, field


@dataclass
class VADConfig:
    gap_threshold_s: float = 2.0
    min_chunk_s: float = 30.0
    max_chunk_s: float = 600.0
    padding_s: float = 0.5


@dataclass
class WhisperXConfig:
    model: str = "large-v3"
    batch_size: int = 32  # Increased for g5.2xlarge (24GB A10G GPU)
    compute_type: str = "float16"
    device: str = "cuda"
    language: str = "en"  # Force English, skip language detection
    min_speakers: int = 1  # Allow single speaker (voicemails, monologues)
    max_speakers: int = 5  # Allow up to 5 speakers (transfers, multi-party calls)


@dataclass
class QualityConfig:
    min_quality_score: float = 0.4
    max_narrator_ratio: float = 0.7
    max_overlap_ratio: float = 0.3
    flip_detection_threshold: float = 0.4


@dataclass
class RoleClassifierConfig:
    confidence_threshold: float = 0.25
    model_version: str = "1.0.0"


@dataclass
class PipelineConfig:
    vad: VADConfig = field(default_factory=VADConfig)
    whisperx: WhisperXConfig = field(default_factory=WhisperXConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    role_classifier: RoleClassifierConfig = field(default_factory=RoleClassifierConfig)
    pipeline_version: str = "1.0.0"


# Legacy bucket (us-east-2) - keep for old pipeline data
S3_BUCKET_LEGACY = "rezora-data-pipeline-864981718771"

# New WhisperX bucket (us-east-1)
# Can be overridden by S3_BUCKET environment variable (for Docker/Batch jobs)
_DEFAULT_S3_BUCKET = "rezora-whisperx-us-east-1-864981718771"
S3_BUCKET = os.environ.get("S3_BUCKET", _DEFAULT_S3_BUCKET)
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# DynamoDB tables
DYNAMODB_VIDEOS_TABLE = "whisperx-videos"
DYNAMODB_CALLS_TABLE = "whisperx-calls"

# Valid conversation types
CONVERSATION_TYPES = ["pretraining", "expired_listing"]
