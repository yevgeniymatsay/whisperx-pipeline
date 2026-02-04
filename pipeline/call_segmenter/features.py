"""Diarization feature computation for call segmentation.

All features must be role-agnostic: diarization speaker IDs are treated as
anonymous labels. Do NOT encode assumptions like "host/nonhost" or any other
pipeline heuristic that correlates with target labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from ..transcriber import DiarizationSegment


@dataclass
class WindowFeatures:
    """Role-agnostic features computed for a single time window."""

    # Basic activity
    speech_frac: float  # Total diarized speech time / window duration (capped at 1.0)
    num_active_speakers: int  # Distinct speakers with any speech in window

    # Short context
    speaker_switches_10s: int  # Speaker switches in trailing 10s

    # Distribution (identity-free)
    dominant_speech_frac: float  # Max single-speaker speech / total speech in window
    speech_entropy: float  # Normalized entropy over per-speaker speech shares (0..1)

    # Longer context (identity-free)
    unique_speakers_30s: int  # Unique speakers in trailing 30s
    turns_30s: int  # Count of diarization segments in trailing 30s
    avg_turn_len_30s: float  # Avg diarization segment length in trailing 30s

    def to_dict(self) -> Dict[str, float]:
        return {
            "speech_frac": self.speech_frac,
            "num_active_speakers": self.num_active_speakers,
            "speaker_switches_10s": self.speaker_switches_10s,
            "dominant_speech_frac": self.dominant_speech_frac,
            "speech_entropy": self.speech_entropy,
            "unique_speakers_30s": self.unique_speakers_30s,
            "turns_30s": self.turns_30s,
            "avg_turn_len_30s": self.avg_turn_len_30s,
        }


def _intersect_duration(
    seg_start: float,
    seg_end: float,
    win_start: float,
    win_end: float,
) -> float:
    overlap_start = max(seg_start, win_start)
    overlap_end = min(seg_end, win_end)
    return max(0.0, overlap_end - overlap_start)


def _segments_in_range(
    segments: List[DiarizationSegment],
    t_start: float,
    t_end: float,
) -> List[DiarizationSegment]:
    return [s for s in segments if s.t1_abs > t_start and s.t0_abs < t_end]


def compute_window_features(
    segments: List[DiarizationSegment],
    win_start: float,
    win_end: float,
    context_10s_start: float,
    context_30s_start: float,
) -> WindowFeatures:
    """Compute all diarization features for a window."""
    win_duration = win_end - win_start
    if win_duration <= 0:
        return WindowFeatures(
            speech_frac=0.0,
            num_active_speakers=0,
            speaker_switches_10s=0,
            dominant_speech_frac=0.0,
            speech_entropy=0.0,
            unique_speakers_30s=0,
            turns_30s=0,
            avg_turn_len_30s=0.0,
        )

    # Window segments (clipped by overlap duration)
    window_segs = _segments_in_range(segments, win_start, win_end)

    total_speech = 0.0
    speech_by_speaker: Dict[str, float] = {}
    active_speakers: Set[str] = set()

    for seg in window_segs:
        dur = _intersect_duration(seg.t0_abs, seg.t1_abs, win_start, win_end)
        if dur <= 0:
            continue
        total_speech += dur
        active_speakers.add(seg.spk)
        speech_by_speaker[seg.spk] = speech_by_speaker.get(seg.spk, 0.0) + dur

    speech_frac = min(1.0, total_speech / win_duration)

    # Dominance and entropy are computed on per-speaker *speech shares* (not window shares).
    dominant_speech_frac = 0.0
    speech_entropy = 0.0
    if total_speech > 0.0 and speech_by_speaker:
        dominant_speech_frac = max(speech_by_speaker.values()) / total_speech

        probs = [d / total_speech for d in speech_by_speaker.values() if d > 0.0]
        if len(probs) > 1:
            h = -sum(p * math.log(p) for p in probs)
            speech_entropy = float(h / math.log(len(probs)))  # normalize to 0..1

    switches_10s, unique_speakers_30s, turns_30s, avg_turn_len_30s = compute_rolling_features(
        segments=segments,
        t_end=win_end,
        context_10s_start=context_10s_start,
        context_30s_start=context_30s_start,
    )

    return WindowFeatures(
        speech_frac=float(speech_frac),
        num_active_speakers=len(active_speakers),
        speaker_switches_10s=int(switches_10s),
        dominant_speech_frac=float(dominant_speech_frac),
        speech_entropy=float(speech_entropy),
        unique_speakers_30s=int(unique_speakers_30s),
        turns_30s=int(turns_30s),
        avg_turn_len_30s=float(avg_turn_len_30s),
    )


def compute_rolling_features(
    segments: List[DiarizationSegment],
    t_end: float,
    context_10s_start: float,
    context_30s_start: float,
) -> Tuple[int, int, int, float]:
    """Compute rolling context features looking back from t_end."""
    segs_10s = _segments_in_range(segments, context_10s_start, t_end)
    segs_30s = _segments_in_range(segments, context_30s_start, t_end)

    switches_10s = _count_speaker_switches(segs_10s)

    unique_speakers_30s = len(set(s.spk for s in segs_30s))
    turns_30s = len(segs_30s)

    if segs_30s:
        total_dur = sum(
            _intersect_duration(s.t0_abs, s.t1_abs, context_30s_start, t_end)
            for s in segs_30s
        )
        avg_turn_len_30s = total_dur / len(segs_30s)
    else:
        avg_turn_len_30s = 0.0

    return switches_10s, unique_speakers_30s, turns_30s, avg_turn_len_30s


def _count_speaker_switches(segments: List[DiarizationSegment]) -> int:
    """Count speaker transitions in chronological diarization segments."""
    if len(segments) < 2:
        return 0

    sorted_segs = sorted(segments, key=lambda s: s.t0_abs)

    switches = 0
    prev = sorted_segs[0].spk
    for seg in sorted_segs[1:]:
        if seg.spk != prev:
            switches += 1
            prev = seg.spk
    return switches


def merge_adjacent_segments(
    segments: List[DiarizationSegment],
    max_gap_s: float = 0.2,
) -> List[DiarizationSegment]:
    """Merge adjacent segments from the same speaker with small gaps."""
    if not segments:
        return []

    sorted_segs = sorted(segments, key=lambda s: s.t0_abs)
    merged: List[DiarizationSegment] = []

    current = DiarizationSegment(
        t0_abs=sorted_segs[0].t0_abs,
        t1_abs=sorted_segs[0].t1_abs,
        spk=sorted_segs[0].spk,
    )

    for seg in sorted_segs[1:]:
        gap = seg.t0_abs - current.t1_abs
        if seg.spk == current.spk and gap <= max_gap_s:
            current.t1_abs = max(current.t1_abs, seg.t1_abs)
        else:
            merged.append(current)
            current = DiarizationSegment(t0_abs=seg.t0_abs, t1_abs=seg.t1_abs, spk=seg.spk)

    merged.append(current)
    return merged

