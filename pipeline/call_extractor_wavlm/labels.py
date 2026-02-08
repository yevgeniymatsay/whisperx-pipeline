from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy as np

from .types import CallBoundary, VideoLabels

logger = logging.getLogger(__name__)


def parse_video_labels(data: Dict[str, Any]) -> VideoLabels:
    video_id = str(data.get("video_id") or "")
    if not video_id:
        raise ValueError("Label JSON missing video_id")

    boundaries_raw = data.get("boundaries", [])
    boundaries: list[CallBoundary] = []
    for b in boundaries_raw:
        start_s = float(b["start_s"])
        end_s = float(b["end_s"])
        if end_s < start_s:
            raise ValueError(f"Invalid boundary for {video_id}: end_s < start_s ({start_s}..{end_s})")
        boundaries.append(CallBoundary(start_s=start_s, end_s=end_s))
    boundaries.sort(key=lambda x: (x.start_s, x.end_s))
    return VideoLabels(video_id=video_id, boundaries=boundaries)


@dataclass(frozen=True)
class TargetConfig:
    start_tolerance_s: float = 0.20
    end_tolerance_s: float = 0.20
    boundary_target_shape: str = "binary"  # "binary" | "triangle"


def _event_targets(
    t: np.ndarray,
    *,
    event_times_s: np.ndarray,
    tolerance_s: float,
    shape: str,
) -> np.ndarray:
    if event_times_s.size == 0:
        return np.zeros((t.shape[0],), dtype=np.float32)

    tol = float(tolerance_s)
    if tol <= 0:
        raise ValueError(f"tolerance_s must be > 0 (got {tolerance_s})")

    shape_n = str(shape).strip().lower()
    if shape_n == "binary":
        return np.any(np.abs(t[:, None] - event_times_s[None, :]) <= tol, axis=1).astype(np.float32)
    if shape_n == "triangle":
        # Soft peak target in [0, 1], with a maximum of 1.0 at the boundary time and
        # linearly decaying to 0.0 at |dt| >= tolerance_s.
        d = np.min(np.abs(t[:, None] - event_times_s[None, :]), axis=1)
        y = 1.0 - (d / tol)
        return np.clip(y, 0.0, 1.0).astype(np.float32)

    raise ValueError(f"Unknown boundary_target_shape: {shape!r}")


def make_targets_for_frames(
    frame_times_abs_s: np.ndarray,
    *,
    boundaries: Sequence[CallBoundary],
    cfg: TargetConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (in_call, start, end) targets for the given absolute frame times."""
    t = frame_times_abs_s.astype(np.float32, copy=False)
    in_call = np.zeros((t.shape[0],), dtype=np.float32)
    start = np.zeros((t.shape[0],), dtype=np.float32)
    end = np.zeros((t.shape[0],), dtype=np.float32)

    if not boundaries:
        return in_call, start, end

    starts = np.array([b.start_s for b in boundaries], dtype=np.float32)
    ends = np.array([b.end_s for b in boundaries], dtype=np.float32)

    start[:] = _event_targets(
        t,
        event_times_s=starts,
        tolerance_s=float(cfg.start_tolerance_s),
        shape=str(cfg.boundary_target_shape),
    )
    end[:] = _event_targets(
        t,
        event_times_s=ends,
        tolerance_s=float(cfg.end_tolerance_s),
        shape=str(cfg.boundary_target_shape),
    )

    for b in boundaries:
        in_call[(t >= float(b.start_s)) & (t <= float(b.end_s))] = 1.0

    return in_call, start, end


def make_core_mask(
    frame_times_abs_s: np.ndarray,
    *,
    core_start_abs_s: float,
    core_end_abs_s: float,
) -> np.ndarray:
    return (frame_times_abs_s >= float(core_start_abs_s)) & (frame_times_abs_s < float(core_end_abs_s))
