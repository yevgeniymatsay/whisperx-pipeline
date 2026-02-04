"""Boundary validation and data conversion utilities.

S3 storage is handled by boundary_store.py.
This module provides validation and format conversion.
"""
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class CallBoundary:
    """A corrected call boundary."""
    video_id: str
    call_index: int
    start_s: float
    end_s: float
    corrected_at: str


def boundaries_to_dict(boundaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert boundaries list to API response format.

    Args:
        boundaries: List of boundary dicts from S3 store

    Returns:
        List of boundary dicts for API response
    """
    return [
        {
            "call_index": b.get("call_index", i),
            "start_s": b["start_s"],
            "end_s": b["end_s"],
            "corrected_at": b.get("corrected_at", ""),
            "session_id": b.get("session_id"),
        }
        for i, b in enumerate(boundaries)
    ]


def validate_boundaries(boundaries: List[Dict[str, float]]) -> tuple:
    """Validate boundaries for overlaps and invalid ranges.

    Args:
        boundaries: List of dicts with start_s and end_s

    Returns:
        (is_valid: bool, errors: List[str])
    """
    errors = []

    # Sort by start time
    sorted_boundaries = sorted(boundaries, key=lambda x: x["start_s"])

    for i, b in enumerate(sorted_boundaries):
        # Check valid range
        if b["start_s"] >= b["end_s"]:
            errors.append(f"Boundary {i+1}: start >= end ({b['start_s']:.2f} >= {b['end_s']:.2f})")

        # Check minimum duration (1 second)
        if b["end_s"] - b["start_s"] < 1.0:
            errors.append(f"Boundary {i+1}: duration too short ({b['end_s'] - b['start_s']:.2f}s < 1s)")

        # Check for overlaps with next boundary
        if i < len(sorted_boundaries) - 1:
            next_b = sorted_boundaries[i + 1]
            if b["end_s"] > next_b["start_s"]:
                errors.append(
                    f"Boundaries {i+1} and {i+2} overlap: "
                    f"{b['end_s']:.2f} > {next_b['start_s']:.2f}"
                )

    return len(errors) == 0, errors
