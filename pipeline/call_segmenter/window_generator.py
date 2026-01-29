"""Window generation and label assignment for call segmentation training.

Generates sliding windows over a timeline and assigns binary labels
based on human-corrected call boundaries.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Iterator, Tuple

from ..transcriber import DiarizationSegment
from .features import compute_window_features, WindowFeatures


@dataclass
class WindowConfig:
    """Configuration for window generation."""
    win_s: float = 1.0          # Window duration in seconds
    hop_s: float = 0.5          # Hop size in seconds
    ignore_s: float = 0.75      # Distance to boundary within which to ignore
    context_10s: float = 10.0   # Lookback for speaker switches
    context_30s: float = 30.0   # Lookback for non-host stats


@dataclass
class CallBoundary:
    """A labeled call boundary (from human annotations)."""
    start: float  # Call start time (absolute seconds)
    end: float    # Call end time (absolute seconds)


@dataclass
class Window:
    """A single window with metadata, features, and labels."""
    # Metadata
    t_start: float
    t_end: float
    t_mid: float

    # Labels
    y: int            # 1 = IN_CALL, 0 = OUT_OF_CALL
    ignore: int       # 1 if within ignore zone near boundary
    dist_to_boundary_s: float  # Distance to nearest call boundary

    # Features (populated separately)
    features: WindowFeatures = field(default_factory=lambda: WindowFeatures(
        speech_frac=0.0,
        host_speech_frac=0.0,
        nonhost_speech_frac=0.0,
        num_active_speakers=0,
        nonhost_active=0,
        speaker_switches_10s=0,
        unique_nonhost_speakers_30s=0,
        nonhost_turns_30s=0,
        avg_nonhost_turn_len_30s=0.0,
    ))

    def to_dict(self) -> Dict:
        """Convert to flat dict for DataFrame row."""
        result = {
            "t_start": self.t_start,
            "t_end": self.t_end,
            "t_mid": self.t_mid,
            "y": self.y,
            "ignore": self.ignore,
            "dist_to_boundary_s": self.dist_to_boundary_s,
        }
        result.update(self.features.to_dict())
        return result


def generate_window_times(
    timeline_start: float,
    timeline_end: float,
    config: WindowConfig,
) -> Iterator[Tuple[float, float, float]]:
    """Generate window timestamps (start, end, mid) over timeline.

    Yields:
        Tuples of (t_start, t_end, t_mid) for each window
    """
    t = timeline_start
    while t + config.win_s <= timeline_end:
        t_end = t + config.win_s
        t_mid = t + config.win_s / 2
        yield t, t_end, t_mid
        t += config.hop_s


def assign_labels(
    t_mid: float,
    call_boundaries: List[CallBoundary],
    ignore_s: float,
) -> Tuple[int, int, float]:
    """Assign label, ignore flag, and distance to nearest boundary.

    Args:
        t_mid: Window center time (absolute seconds)
        call_boundaries: List of call boundaries
        ignore_s: Distance threshold for ignore zone

    Returns:
        Tuple of (y, ignore, dist_to_boundary_s)
    """
    if not call_boundaries:
        return 0, 0, float("inf")

    # Check if t_mid is inside any call
    y = 0
    for boundary in call_boundaries:
        if boundary.start <= t_mid <= boundary.end:
            y = 1
            break

    # Find distance to nearest boundary (start or end of any call)
    min_dist = float("inf")
    for boundary in call_boundaries:
        dist_to_start = abs(t_mid - boundary.start)
        dist_to_end = abs(t_mid - boundary.end)
        min_dist = min(min_dist, dist_to_start, dist_to_end)

    # Set ignore flag if within ignore zone
    ignore = 1 if min_dist <= ignore_s else 0

    return y, ignore, min_dist


def generate_windows(
    segments: List[DiarizationSegment],
    host_speaker: str,
    call_boundaries: List[CallBoundary],
    timeline_start: float,
    timeline_end: float,
    config: WindowConfig,
) -> List[Window]:
    """Generate windows with features and labels.

    Args:
        segments: All diarization segments
        host_speaker: Identified host speaker ID
        call_boundaries: Human-labeled call boundaries
        timeline_start: Start of timeline (seconds)
        timeline_end: End of timeline (seconds)
        config: Window generation configuration

    Returns:
        List of Window objects ready for DataFrame conversion
    """
    windows: List[Window] = []

    for t_start, t_end, t_mid in generate_window_times(
        timeline_start, timeline_end, config
    ):
        # Compute context boundaries
        context_10s_start = max(timeline_start, t_end - config.context_10s)
        context_30s_start = max(timeline_start, t_end - config.context_30s)

        # Compute features
        features = compute_window_features(
            segments=segments,
            host_speaker=host_speaker,
            win_start=t_start,
            win_end=t_end,
            context_10s_start=context_10s_start,
            context_30s_start=context_30s_start,
        )

        # Assign labels
        y, ignore, dist = assign_labels(t_mid, call_boundaries, config.ignore_s)

        window = Window(
            t_start=t_start,
            t_end=t_end,
            t_mid=t_mid,
            y=y,
            ignore=ignore,
            dist_to_boundary_s=dist,
            features=features,
        )
        windows.append(window)

    return windows


def compute_call_sanity_stats(
    segments: List[DiarizationSegment],
    host_speaker: str,
    boundary: CallBoundary,
) -> Dict:
    """Compute sanity statistics for a single call.

    Used to detect potential labeling errors (e.g., calls with no non-host speech).

    Args:
        segments: All diarization segments
        host_speaker: Host speaker ID
        boundary: The call boundary to analyze

    Returns:
        Dict with sanity stats for this call
    """
    # Get segments within this call
    call_segs = [
        s for s in segments
        if s.t1_abs > boundary.start and s.t0_abs < boundary.end
    ]

    call_duration = boundary.end - boundary.start
    host_speech = 0.0
    nonhost_speech = 0.0
    nonhost_speakers: set = set()
    switches = 0

    # Sort segments by time
    sorted_segs = sorted(call_segs, key=lambda s: s.t0_abs)
    prev_speaker = None

    for seg in sorted_segs:
        # Clip segment to call bounds
        seg_start = max(seg.t0_abs, boundary.start)
        seg_end = min(seg.t1_abs, boundary.end)
        duration = seg_end - seg_start

        if seg.spk == host_speaker:
            host_speech += duration
        else:
            nonhost_speech += duration
            nonhost_speakers.add(seg.spk)

        # Count switches
        if prev_speaker is not None and seg.spk != prev_speaker:
            switches += 1
        prev_speaker = seg.spk

    nonhost_frac = nonhost_speech / call_duration if call_duration > 0 else 0.0

    return {
        "start": boundary.start,
        "end": boundary.end,
        "duration_s": call_duration,
        "nonhost_frac": round(nonhost_frac, 3),
        "nonhost_speakers": len(nonhost_speakers),
        "switches": switches,
        "host_speech_s": round(host_speech, 2),
        "nonhost_speech_s": round(nonhost_speech, 2),
    }
