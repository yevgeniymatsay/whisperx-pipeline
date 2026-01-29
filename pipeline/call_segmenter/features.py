"""Diarization feature computation for call segmentation.

Features are computed from diarization segments within sliding windows.
All features here are safe for ML training (no leakage from labels).
"""

from dataclasses import dataclass
from typing import List, Set, Dict, Tuple

from ..transcriber import DiarizationSegment


@dataclass
class WindowFeatures:
    """Features computed for a single time window."""
    # Basic speech fractions
    speech_frac: float          # Total speech time / window duration
    host_speech_frac: float     # Host speech time / window duration
    nonhost_speech_frac: float  # Non-host speech time / window duration

    # Speaker activity
    num_active_speakers: int    # Distinct speakers in window
    nonhost_active: int         # 1 if any non-host speaking, else 0

    # Rolling context features (lookback windows)
    speaker_switches_10s: int       # Speaker switches in trailing 10s
    unique_nonhost_speakers_30s: int  # Unique non-host in trailing 30s
    nonhost_turns_30s: int          # Non-host turn count in trailing 30s
    avg_nonhost_turn_len_30s: float # Avg non-host turn duration in 30s

    def to_dict(self) -> Dict[str, float]:
        """Convert to dict for DataFrame row."""
        return {
            "speech_frac": self.speech_frac,
            "host_speech_frac": self.host_speech_frac,
            "nonhost_speech_frac": self.nonhost_speech_frac,
            "num_active_speakers": self.num_active_speakers,
            "nonhost_active": self.nonhost_active,
            "speaker_switches_10s": self.speaker_switches_10s,
            "unique_nonhost_speakers_30s": self.unique_nonhost_speakers_30s,
            "nonhost_turns_30s": self.nonhost_turns_30s,
            "avg_nonhost_turn_len_30s": self.avg_nonhost_turn_len_30s,
        }


def _intersect_duration(
    seg_start: float,
    seg_end: float,
    win_start: float,
    win_end: float,
) -> float:
    """Compute intersection duration between segment and window."""
    overlap_start = max(seg_start, win_start)
    overlap_end = min(seg_end, win_end)
    return max(0.0, overlap_end - overlap_start)


def _segments_in_range(
    segments: List[DiarizationSegment],
    t_start: float,
    t_end: float,
) -> List[DiarizationSegment]:
    """Filter segments that overlap with time range."""
    return [
        s for s in segments
        if s.t1_abs > t_start and s.t0_abs < t_end
    ]


def compute_window_features(
    segments: List[DiarizationSegment],
    host_speaker: str,
    win_start: float,
    win_end: float,
    context_10s_start: float,
    context_30s_start: float,
) -> WindowFeatures:
    """Compute all diarization features for a window.

    Args:
        segments: All diarization segments (should be sorted by time)
        host_speaker: The identified host speaker ID
        win_start: Window start time (absolute seconds)
        win_end: Window end time (absolute seconds)
        context_10s_start: Start of 10s lookback context
        context_30s_start: Start of 30s lookback context

    Returns:
        WindowFeatures with all computed features
    """
    win_duration = win_end - win_start

    # Get segments overlapping the window
    window_segs = _segments_in_range(segments, win_start, win_end)

    # Compute speech fractions
    total_speech = 0.0
    host_speech = 0.0
    nonhost_speech = 0.0
    active_speakers: Set[str] = set()
    nonhost_speakers_in_window: Set[str] = set()

    for seg in window_segs:
        duration = _intersect_duration(seg.t0_abs, seg.t1_abs, win_start, win_end)
        total_speech += duration
        active_speakers.add(seg.spk)

        if seg.spk == host_speaker:
            host_speech += duration
        else:
            nonhost_speech += duration
            nonhost_speakers_in_window.add(seg.spk)

    # Compute fractions, capped at 1.0 (overlapping segments can exceed window)
    speech_frac = min(1.0, total_speech / win_duration) if win_duration > 0 else 0.0
    host_speech_frac = min(1.0, host_speech / win_duration) if win_duration > 0 else 0.0
    nonhost_speech_frac = min(1.0, nonhost_speech / win_duration) if win_duration > 0 else 0.0

    # Rolling context features
    (
        switches_10s,
        unique_nonhost_30s,
        nonhost_turns_30s,
        avg_turn_len_30s,
    ) = compute_rolling_features(
        segments, host_speaker, win_end, context_10s_start, context_30s_start
    )

    return WindowFeatures(
        speech_frac=speech_frac,
        host_speech_frac=host_speech_frac,
        nonhost_speech_frac=nonhost_speech_frac,
        num_active_speakers=len(active_speakers),
        nonhost_active=1 if nonhost_speakers_in_window else 0,
        speaker_switches_10s=switches_10s,
        unique_nonhost_speakers_30s=unique_nonhost_30s,
        nonhost_turns_30s=nonhost_turns_30s,
        avg_nonhost_turn_len_30s=avg_turn_len_30s,
    )


def compute_rolling_features(
    segments: List[DiarizationSegment],
    host_speaker: str,
    t_end: float,
    context_10s_start: float,
    context_30s_start: float,
) -> Tuple[int, int, int, float]:
    """Compute rolling context features looking back from t_end.

    Args:
        segments: All diarization segments
        host_speaker: Host speaker ID
        t_end: End of current window (lookback ends here)
        context_10s_start: Start of 10s lookback window
        context_30s_start: Start of 30s lookback window

    Returns:
        Tuple of:
            - speaker_switches_10s: Number of speaker transitions in 10s window
            - unique_nonhost_speakers_30s: Unique non-host speakers in 30s window
            - nonhost_turns_30s: Number of non-host turns in 30s window
            - avg_nonhost_turn_len_30s: Average non-host turn duration
    """
    # Get segments in each context window
    segs_10s = _segments_in_range(segments, context_10s_start, t_end)
    segs_30s = _segments_in_range(segments, context_30s_start, t_end)

    # Speaker switches in 10s (count transitions)
    switches_10s = _count_speaker_switches(segs_10s)

    # Non-host stats in 30s window
    nonhost_segs_30s = [s for s in segs_30s if s.spk != host_speaker]
    unique_nonhost_30s = len(set(s.spk for s in nonhost_segs_30s))
    nonhost_turns_30s = len(nonhost_segs_30s)

    # Average non-host turn length
    if nonhost_segs_30s:
        total_duration = sum(
            min(s.t1_abs, t_end) - max(s.t0_abs, context_30s_start)
            for s in nonhost_segs_30s
        )
        avg_turn_len_30s = total_duration / len(nonhost_segs_30s)
    else:
        avg_turn_len_30s = 0.0

    return switches_10s, unique_nonhost_30s, nonhost_turns_30s, avg_turn_len_30s


def _count_speaker_switches(segments: List[DiarizationSegment]) -> int:
    """Count speaker switches (transitions) in a list of segments.

    A switch occurs when consecutive segments have different speakers.
    Segments should be in chronological order.
    """
    if len(segments) < 2:
        return 0

    # Sort by start time to ensure chronological order
    sorted_segs = sorted(segments, key=lambda s: s.t0_abs)

    switches = 0
    prev_speaker = sorted_segs[0].spk
    for seg in sorted_segs[1:]:
        if seg.spk != prev_speaker:
            switches += 1
            prev_speaker = seg.spk

    return switches


def merge_adjacent_segments(
    segments: List[DiarizationSegment],
    max_gap_s: float = 0.2,
) -> List[DiarizationSegment]:
    """Merge adjacent segments from the same speaker with small gaps.

    This cleans up diarization output where the same speaker's speech
    may be split into multiple adjacent segments.

    Args:
        segments: List of segments (will be sorted by start time)
        max_gap_s: Maximum gap between segments to merge

    Returns:
        List of merged segments
    """
    if not segments:
        return []

    # Sort by start time
    sorted_segs = sorted(segments, key=lambda s: s.t0_abs)
    merged: List[DiarizationSegment] = []

    current = DiarizationSegment(
        t0_abs=sorted_segs[0].t0_abs,
        t1_abs=sorted_segs[0].t1_abs,
        spk=sorted_segs[0].spk,
    )

    for seg in sorted_segs[1:]:
        # Check if we can merge with current
        gap = seg.t0_abs - current.t1_abs
        if seg.spk == current.spk and gap <= max_gap_s:
            # Extend current segment
            current.t1_abs = max(current.t1_abs, seg.t1_abs)
        else:
            # Save current and start new
            merged.append(current)
            current = DiarizationSegment(
                t0_abs=seg.t0_abs,
                t1_abs=seg.t1_abs,
                spk=seg.spk,
            )

    # Don't forget the last segment
    merged.append(current)
    return merged
