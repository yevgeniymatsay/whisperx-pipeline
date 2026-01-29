"""Call Segmenter - Dataset builder for training call boundary detection models.

This package builds windowed training data from WhisperX diarization output
with human-corrected call boundary labels.

Modules:
    features: Diarization feature computation (speech fractions, switches, etc.)
    host_detector: Host speaker identification using time coverage bins
    window_generator: Window generation with label assignment

The dataset builder produces Parquet files with:
    - Metadata columns (video_id, timestamps, etc.)
    - Label columns (y, ignore, dist_to_boundary_s)
    - Feature columns (diarization-derived, safe for training)
"""

from .features import compute_window_features, compute_rolling_features
from .host_detector import detect_host_speaker, HostDetectionResult
from .window_generator import generate_windows, assign_labels, WindowConfig

__all__ = [
    "compute_window_features",
    "compute_rolling_features",
    "detect_host_speaker",
    "HostDetectionResult",
    "generate_windows",
    "assign_labels",
    "WindowConfig",
]
