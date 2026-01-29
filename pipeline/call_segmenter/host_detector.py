"""Host speaker identification using time coverage bins.

The host (narrator/caller making the calls) typically speaks across the entire
recording while non-host speakers (recipients) appear in distinct call segments.
We identify the host by binning the timeline and finding the speaker present
in the most bins.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from collections import defaultdict

from ..transcriber import DiarizationSegment


@dataclass
class SpeakerStats:
    """Statistics for a single speaker."""
    speaker_id: str
    bins_present: int       # Number of time bins where speaker appears
    total_bins: int         # Total bins in recording
    coverage_ratio: float   # bins_present / total_bins
    total_speech_s: float   # Total speech duration


@dataclass
class HostDetectionResult:
    """Result of host speaker detection."""
    host_speaker: str
    host_confidence: float  # (coverage_top1 - coverage_top2) / coverage_top1
    top_speakers: List[SpeakerStats]  # Top speakers by bin coverage
    total_bins: int
    bin_size_s: float
    is_ambiguous: bool = field(default=False)  # True if confidence < threshold

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "host_speaker": self.host_speaker,
            "host_confidence": self.host_confidence,
            "top_speakers": [
                {
                    "speaker_id": s.speaker_id,
                    "bins_present": s.bins_present,
                    "coverage_ratio": s.coverage_ratio,
                    "total_speech_s": s.total_speech_s,
                }
                for s in self.top_speakers
            ],
            "total_bins": self.total_bins,
            "bin_size_s": self.bin_size_s,
            "is_ambiguous": self.is_ambiguous,
        }


def detect_host_speaker(
    segments: List[DiarizationSegment],
    timeline_end: float,
    bin_size_s: float = 30.0,
    ambiguity_threshold: float = 0.15,
    top_n: int = 3,
) -> HostDetectionResult:
    """Identify the host speaker using time bin coverage.

    The host is the speaker appearing in the most time bins. This works because
    the host speaks during transitions, narration, and across multiple calls,
    while non-host speakers only appear within their respective calls.

    Args:
        segments: List of diarization segments with absolute timestamps
        timeline_end: End of the timeline in seconds
        bin_size_s: Size of each time bin in seconds (default 30s)
        ambiguity_threshold: Confidence threshold below which host is ambiguous
        top_n: Number of top speakers to return in results

    Returns:
        HostDetectionResult with identified host and confidence metrics

    Example:
        >>> segments = [
        ...     DiarizationSegment(0.0, 5.0, "SPEAKER_00"),  # Host intro
        ...     DiarizationSegment(5.0, 30.0, "SPEAKER_01"), # Guest 1
        ...     DiarizationSegment(60.0, 65.0, "SPEAKER_00"), # Host transition
        ...     DiarizationSegment(65.0, 90.0, "SPEAKER_02"), # Guest 2
        ... ]
        >>> result = detect_host_speaker(segments, timeline_end=100.0)
        >>> result.host_speaker  # "SPEAKER_00" - appears in multiple bins
    """
    if not segments:
        return HostDetectionResult(
            host_speaker="SPEAKER_00",  # Default fallback
            host_confidence=0.0,
            top_speakers=[],
            total_bins=0,
            bin_size_s=bin_size_s,
            is_ambiguous=True,
        )

    # Calculate number of bins
    num_bins = max(1, int(timeline_end / bin_size_s) + 1)

    # Track which bins each speaker appears in
    speaker_bins: Dict[str, set] = defaultdict(set)
    speaker_speech_s: Dict[str, float] = defaultdict(float)

    for seg in segments:
        # Find all bins this segment overlaps with
        start_bin = int(seg.t0_abs / bin_size_s)
        end_bin = int(seg.t1_abs / bin_size_s)

        for bin_idx in range(start_bin, end_bin + 1):
            if 0 <= bin_idx < num_bins:
                speaker_bins[seg.spk].add(bin_idx)

        # Track total speech duration
        speaker_speech_s[seg.spk] += seg.t1_abs - seg.t0_abs

    # Build speaker stats sorted by bin coverage
    speaker_stats: List[SpeakerStats] = []
    for spk, bins in speaker_bins.items():
        speaker_stats.append(SpeakerStats(
            speaker_id=spk,
            bins_present=len(bins),
            total_bins=num_bins,
            coverage_ratio=len(bins) / num_bins,
            total_speech_s=speaker_speech_s[spk],
        ))

    # Sort by bins_present descending, then by total_speech_s as tiebreaker
    speaker_stats.sort(key=lambda x: (x.bins_present, x.total_speech_s), reverse=True)

    # Determine host and confidence
    if len(speaker_stats) == 0:
        return HostDetectionResult(
            host_speaker="SPEAKER_00",
            host_confidence=0.0,
            top_speakers=[],
            total_bins=num_bins,
            bin_size_s=bin_size_s,
            is_ambiguous=True,
        )

    host = speaker_stats[0]

    # Calculate confidence as relative gap between top two
    if len(speaker_stats) >= 2:
        second = speaker_stats[1]
        if host.coverage_ratio > 0:
            confidence = (host.coverage_ratio - second.coverage_ratio) / host.coverage_ratio
        else:
            confidence = 0.0
    else:
        # Only one speaker - high confidence
        confidence = 1.0

    return HostDetectionResult(
        host_speaker=host.speaker_id,
        host_confidence=confidence,
        top_speakers=speaker_stats[:top_n],
        total_bins=num_bins,
        bin_size_s=bin_size_s,
        is_ambiguous=confidence < ambiguity_threshold,
    )


def get_host_and_nonhost_segments(
    segments: List[DiarizationSegment],
    host_speaker: str,
) -> Tuple[List[DiarizationSegment], List[DiarizationSegment]]:
    """Split segments into host and non-host groups.

    Args:
        segments: All diarization segments
        host_speaker: The identified host speaker ID

    Returns:
        Tuple of (host_segments, nonhost_segments)
    """
    host_segs = [s for s in segments if s.spk == host_speaker]
    nonhost_segs = [s for s in segments if s.spk != host_speaker]
    return host_segs, nonhost_segs
