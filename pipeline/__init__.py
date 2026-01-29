"""WhisperX transcription pipeline."""
from .config import PipelineConfig, S3_BUCKET, S3_BUCKET_LEGACY, AWS_REGION
from . import db

# v2 Role Classification (3-class: agent/user/narrator)
from .role_features import extract_all_speaker_features, extract_speaker_features
from .role_predictor_v2 import RolePredictorV2, RolePredictionV2
from .train_role_model_v2 import train_model_v2, load_labels_v2, FEATURE_COLUMNS

__all__ = [
    "PipelineConfig",
    "S3_BUCKET",
    "S3_BUCKET_LEGACY",
    "AWS_REGION",
    "db",
    # v2 Role Classification
    "extract_all_speaker_features",
    "extract_speaker_features",
    "RolePredictorV2",
    "RolePredictionV2",
    "train_model_v2",
    "load_labels_v2",
    "FEATURE_COLUMNS",
]
