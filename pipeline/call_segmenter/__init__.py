"""Call Segmenter - windowed dataset utilities for call boundary detection.

This package builds role-agnostic training data from WhisperX diarization (and
optionally ASR text) aligned with human-corrected call boundaries.

Important: This package must not encode "host/nonhost" assumptions. Speaker
identity is treated as arbitrary labels from diarization; all features should
be invariant to speaker role.
"""

from .features import compute_window_features, compute_rolling_features, merge_adjacent_segments
from .window_generator import generate_windows, assign_labels, WindowConfig

__all__ = [
    "compute_window_features",
    "compute_rolling_features",
    "merge_adjacent_segments",
    "generate_windows",
    "assign_labels",
    "WindowConfig",
]
