"""Boundary management for call boundary editor."""
import csv
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

BOUNDARIES_FILE = os.path.join(os.path.dirname(__file__), "corrected_boundaries.csv")


@dataclass
class CallBoundary:
    """A corrected call boundary."""
    video_id: str
    call_index: int
    start_s: float
    end_s: float
    corrected_at: str


def load_corrected_boundaries(video_id: Optional[str] = None) -> List[CallBoundary]:
    """Load corrected boundaries from CSV, optionally filtered by video_id."""
    if not os.path.exists(BOUNDARIES_FILE):
        return []

    boundaries = []
    with open(BOUNDARIES_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if video_id is None or row["video_id"] == video_id:
                boundaries.append(CallBoundary(
                    video_id=row["video_id"],
                    call_index=int(row["call_index"]),
                    start_s=float(row["start_s"]),
                    end_s=float(row["end_s"]),
                    corrected_at=row["corrected_at"],
                ))
    return boundaries


def save_corrected_boundaries(video_id: str, boundaries: List[Dict[str, Any]]) -> None:
    """Save corrected boundaries for a video (replaces existing for that video)."""
    # Load all existing boundaries except for this video
    existing = [b for b in load_corrected_boundaries() if b.video_id != video_id]

    # Add new boundaries for this video
    now = datetime.now().isoformat()
    for i, b in enumerate(boundaries):
        existing.append(CallBoundary(
            video_id=video_id,
            call_index=i,
            start_s=b["start_s"],
            end_s=b["end_s"],
            corrected_at=now,
        ))

    # Sort by video_id, then call_index
    existing.sort(key=lambda x: (x.video_id, x.call_index))

    # Write back to CSV
    with open(BOUNDARIES_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "call_index", "start_s", "end_s", "corrected_at"])
        writer.writeheader()
        for b in existing:
            writer.writerow({
                "video_id": b.video_id,
                "call_index": b.call_index,
                "start_s": b.start_s,
                "end_s": b.end_s,
                "corrected_at": b.corrected_at,
            })


def get_corrected_video_ids() -> set:
    """Get set of video IDs that have corrected boundaries."""
    boundaries = load_corrected_boundaries()
    return set(b.video_id for b in boundaries)


def boundaries_to_dict(boundaries: List[CallBoundary]) -> List[Dict[str, Any]]:
    """Convert boundaries to dict format for API response."""
    return [
        {
            "call_index": b.call_index,
            "start_s": b.start_s,
            "end_s": b.end_s,
            "corrected_at": b.corrected_at,
        }
        for b in boundaries
    ]
