# scripts/whisperx_pipeline/role_features.py
"""Feature extraction for role classification."""
from typing import List, Dict
from statistics import mean, median


def safe_mean(values: List[float]) -> float:
    return mean(values) if values else 0.0


def safe_median(values: List[float]) -> float:
    return median(values) if values else 0.0


def safe_max(values: List[float]) -> float:
    return max(values) if values else 0.0


def is_question(text: str) -> bool:
    """Robust question detection."""
    text_lower = text.strip().lower()
    if text.rstrip().endswith("?"):
        return True
    words = text_lower.split()[:3]
    interrogatives = {"is", "are", "do", "did", "can", "could", "would",
                     "what", "why", "how", "when", "where", "who", "whose", "which"}
    return any(w in interrogatives for w in words)


def extract_speaker_features(
    all_turns: List[dict],
    speaker: str,
    call_start_abs: float,
    window_s: float = 60.0
) -> Dict[str, float]:
    """Extract features for one speaker from first 60s."""
    # Filter to first 60s and this speaker
    first_60s = [t for t in all_turns if t["t0_abs"] < call_start_abs + window_s]
    spk_turns = [t for t in first_60s if t["spk"] == speaker]

    if not spk_turns:
        return {k: 0.0 for k in ["talk_time", "turn_count", "avg_turn_duration",
                                  "median_turn_duration", "max_turn_duration",
                                  "question_turn_count", "question_turn_rate",
                                  "long_turn_count", "short_turn_count",
                                  "word_count", "avg_words_per_turn", "first_speaker"]}

    first_turn_spk = first_60s[0]["spk"] if first_60s else None
    durations = [t["t1_abs"] - t["t0_abs"] for t in spk_turns]
    word_counts = [len(t["text"].split()) for t in spk_turns]
    turn_count = len(spk_turns)
    question_count = sum(1 for t in spk_turns if is_question(t["text"]))

    return {
        "talk_time": sum(durations),
        "turn_count": turn_count,
        "avg_turn_duration": safe_mean(durations),
        "median_turn_duration": safe_median(durations),
        "max_turn_duration": safe_max(durations),
        "question_turn_count": question_count,
        "question_turn_rate": question_count / turn_count if turn_count > 0 else 0,
        "long_turn_count": sum(1 for d in durations if d > 5),
        "short_turn_count": sum(1 for d in durations if d < 0.6),
        "word_count": sum(word_counts),
        "avg_words_per_turn": safe_mean(word_counts),
        "first_speaker": 1 if speaker == first_turn_spk else 0,
    }


def extract_call_features(
    call_id: str,
    spk_turns: List[dict],
    call_start_abs: float
) -> Dict[str, float]:
    """Extract call-level difference features."""
    # Hardcoded speaker ordering
    spk0, spk1 = "SPEAKER_00", "SPEAKER_01"

    feat0 = extract_speaker_features(spk_turns, spk0, call_start_abs)
    feat1 = extract_speaker_features(spk_turns, spk1, call_start_abs)

    diff = {f"diff_{k}": feat0[k] - feat1[k] for k in feat0}
    return {"call_id": call_id, "spk0": spk0, "spk1": spk1, **diff}


def extract_all_speaker_features(
    all_turns: List[dict],
    call_start_abs: float,
    window_s: float = 60.0
) -> Dict[str, Dict[str, float]]:
    """
    Extract features for ALL speakers in the call.

    Returns: {speaker_id: {feature_name: value, ...}, ...}

    This replaces the hardcoded SPEAKER_00/01 approach.
    Each speaker gets their own feature vector for classification.
    """
    # Find all unique speakers
    speakers = set(t["spk"] for t in all_turns)

    # Extract features for each speaker
    result = {}
    for speaker in speakers:
        features = extract_speaker_features(all_turns, speaker, call_start_abs, window_s)
        result[speaker] = features

    return result
