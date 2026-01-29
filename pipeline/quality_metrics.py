# scripts/whisperx_pipeline/quality_metrics.py
"""Quality metrics for diarization output."""
from dataclasses import dataclass
from typing import List, Tuple, Optional
from statistics import median


@dataclass
class QualityReason:
    type: str
    t0_abs: float
    t1_abs: float
    severity: float
    details: Optional[str] = None


@dataclass
class QualityMetrics:
    overlap_ratio: float
    speaker_switches_per_min: float
    median_turn_s: float
    micro_turn_ratio: float
    narrator_ratio: float
    speaker_flip_suspected: bool


def compute_narrator_ratio(words: List[dict], duration_s: float) -> float:
    """
    Compute narrator ratio: how much the chunk resembles narration vs dialogue.

    High ratio (>0.7) = likely narration/monologue (one speaker dominates with long turns)
    Low ratio (<0.3) = likely dialogue (back-and-forth conversation)

    Method:
    1. Compute talk time per speaker
    2. Compute average turn duration per speaker
    3. If dominant speaker has >80% talk time AND avg turn >10s, it's narration
    """
    if not words or duration_s <= 0:
        return 0.0

    # Group consecutive words by speaker into turns
    turns = []
    current_spk = None
    current_start = None
    current_end = None

    for w in words:
        if w["spk"] != current_spk:
            if current_spk is not None:
                turns.append({"spk": current_spk, "start": current_start, "end": current_end})
            current_spk = w["spk"]
            current_start = w["t0_abs"]
        current_end = w["t1_abs"]

    if current_spk is not None:
        turns.append({"spk": current_spk, "start": current_start, "end": current_end})

    if not turns:
        return 0.0

    # Compute per-speaker stats
    speaker_stats = {}
    for turn in turns:
        spk = turn["spk"]
        dur = turn["end"] - turn["start"]
        if spk not in speaker_stats:
            speaker_stats[spk] = {"talk_time": 0.0, "turn_count": 0, "turn_durations": []}
        speaker_stats[spk]["talk_time"] += dur
        speaker_stats[spk]["turn_count"] += 1
        speaker_stats[spk]["turn_durations"].append(dur)

    if len(speaker_stats) < 2:
        # Only one speaker = definitely narration
        return 1.0

    # Find dominant speaker
    total_talk = sum(s["talk_time"] for s in speaker_stats.values())
    if total_talk <= 0:
        return 0.0

    dominant_spk = max(speaker_stats.keys(), key=lambda s: speaker_stats[s]["talk_time"])
    dominant = speaker_stats[dominant_spk]

    talk_ratio = dominant["talk_time"] / total_talk
    avg_turn = dominant["talk_time"] / dominant["turn_count"] if dominant["turn_count"] > 0 else 0

    # Narrator pattern: >70% talk time AND avg turn >8s
    if talk_ratio > 0.7 and avg_turn > 8.0:
        return min(1.0, talk_ratio * (avg_turn / 10.0))

    # Moderate narration: >60% talk time AND avg turn >5s
    if talk_ratio > 0.6 and avg_turn > 5.0:
        return talk_ratio * 0.7

    # Dialogue pattern
    return talk_ratio * 0.3


def detect_speaker_flip(words: List[dict], window_s: float = 30.0) -> Tuple[bool, Optional[QualityReason]]:
    """
    Detect if speakers appear to swap roles mid-call.

    This is different from high switch rate (which can be normal rapid dialogue).
    Speaker flip = the "agent pattern" (longer turns, questions) moves from one speaker to another.

    Method:
    1. Split into first half and second half
    2. Compute "agent score" per speaker in each half
    3. If dominant agent flips, flag it
    """
    if not words:
        return False, None

    # Get time range
    min_t = min(w["t0_abs"] for w in words)
    max_t = max(w["t1_abs"] for w in words)
    mid_t = (min_t + max_t) / 2

    if max_t - min_t < 20:  # Too short to detect flip
        return False, None

    first_half = [w for w in words if w["t0_abs"] < mid_t]
    second_half = [w for w in words if w["t0_abs"] >= mid_t]

    def agent_score(word_list: List[dict]) -> dict:
        """Compute agent-like score per speaker (longer turns, more questions)."""
        turns = _group_into_turns(word_list)
        scores = {}
        for spk, spk_turns in _group_turns_by_speaker(turns).items():
            if not spk_turns:
                scores[spk] = 0
                continue
            avg_dur = sum(t["end"] - t["start"] for t in spk_turns) / len(spk_turns)
            question_ratio = sum(1 for t in spk_turns if "?" in t.get("text", "")) / len(spk_turns)
            scores[spk] = avg_dur * (1 + question_ratio)
        return scores

    first_scores = agent_score(first_half)
    second_scores = agent_score(second_half)

    if not first_scores or not second_scores:
        return False, None

    first_agent = max(first_scores.keys(), key=lambda s: first_scores.get(s, 0))
    second_agent = max(second_scores.keys(), key=lambda s: second_scores.get(s, 0))

    # Check if agent flipped AND the scores are meaningfully different
    if first_agent != second_agent:
        first_gap = first_scores.get(first_agent, 0) - first_scores.get(second_agent, 0)
        second_gap = second_scores.get(second_agent, 0) - second_scores.get(first_agent, 0)

        if first_gap > 1.0 and second_gap > 1.0:
            return True, QualityReason(
                type="speaker_flip_suspected",
                t0_abs=mid_t - 5,
                t1_abs=mid_t + 5,
                severity=min(1.0, (first_gap + second_gap) / 10),
                details=f"Agent pattern moved from {first_agent} to {second_agent}"
            )

    return False, None


def _group_into_turns(words: List[dict]) -> List[dict]:
    """Group consecutive words by speaker into turns."""
    if not words:
        return []
    turns = []
    current = {"spk": words[0]["spk"], "start": words[0]["t0_abs"], "end": words[0]["t1_abs"], "text": words[0].get("text", "")}
    for w in words[1:]:
        if w["spk"] == current["spk"]:
            current["end"] = w["t1_abs"]
            current["text"] += " " + w.get("text", "")
        else:
            turns.append(current)
            current = {"spk": w["spk"], "start": w["t0_abs"], "end": w["t1_abs"], "text": w.get("text", "")}
    turns.append(current)
    return turns


def _group_turns_by_speaker(turns: List[dict]) -> dict:
    """Group turns by speaker."""
    result = {}
    for t in turns:
        spk = t["spk"]
        if spk not in result:
            result[spk] = []
        result[spk].append(t)
    return result


def compute_quality_metrics(
    words: List[dict],
    diarization_segments: List[dict],
    duration_s: float
) -> Tuple[QualityMetrics, List[QualityReason]]:
    """Compute all quality metrics for a chunk."""
    reasons = []

    # Overlap ratio (from diarization segments)
    overlap_ratio = _compute_overlap_ratio(diarization_segments)

    # Speaker switches per minute
    turns = _group_into_turns(words)
    switches = len(turns) - 1 if turns else 0
    switches_per_min = (switches / duration_s) * 60 if duration_s > 0 else 0

    # Median turn duration
    turn_durations = [t["end"] - t["start"] for t in turns]
    median_turn = median(turn_durations) if turn_durations else 0

    # Micro-turn ratio (turns < 0.7s)
    micro_turns = sum(1 for d in turn_durations if d < 0.7)
    micro_ratio = micro_turns / len(turns) if turns else 0

    # Narrator ratio
    narrator_ratio = compute_narrator_ratio(words, duration_s)

    # Speaker flip detection
    flip_detected, flip_reason = detect_speaker_flip(words)
    if flip_reason:
        reasons.append(flip_reason)

    # Add high overlap reason if needed
    if overlap_ratio > 0.15:
        reasons.append(QualityReason(
            type="high_overlap",
            t0_abs=0,
            t1_abs=duration_s,
            severity=overlap_ratio
        ))

    return QualityMetrics(
        overlap_ratio=overlap_ratio,
        speaker_switches_per_min=switches_per_min,
        median_turn_s=median_turn,
        micro_turn_ratio=micro_ratio,
        narrator_ratio=narrator_ratio,
        speaker_flip_suspected=flip_detected
    ), reasons


def _compute_overlap_ratio(segments: List[dict]) -> float:
    """
    Compute ratio of overlapping speech from diarization segments.

    Uses sweep line algorithm to avoid overcounting when >2 segments overlap.
    """
    if len(segments) < 2:
        return 0.0

    total_duration = max(s["t1_abs"] for s in segments) - min(s["t0_abs"] for s in segments)
    if total_duration <= 0:
        return 0.0

    # Sweep line: create events (+1 at start, -1 at end)
    events = []
    for seg in segments:
        events.append((seg["t0_abs"], +1))  # segment starts
        events.append((seg["t1_abs"], -1))  # segment ends

    events.sort(key=lambda e: (e[0], -e[1]))  # Sort by time, starts before ends at same time

    overlap_time = 0.0
    active_speakers = 0
    prev_time = None

    for time, delta in events:
        if prev_time is not None and active_speakers >= 2:
            overlap_time += time - prev_time
        active_speakers += delta
        prev_time = time

    return overlap_time / total_duration
