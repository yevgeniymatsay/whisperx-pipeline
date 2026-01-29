# tests/whisperx_pipeline/test_role_features.py
import pytest
from pipeline.role_features import (
    extract_speaker_features,
    extract_call_features,
    is_question
)


def test_is_question_punctuation():
    """Question mark detected."""
    assert is_question("How are you?")
    assert not is_question("I am fine.")


def test_is_question_interrogative():
    """Interrogative words detected."""
    assert is_question("Do you have time")
    assert is_question("What is your name")
    assert is_question("Can I help you")


def test_extract_speaker_features():
    """Extract features for one speaker."""
    turns = [
        {"spk": "SPEAKER_00", "t0_abs": 0, "t1_abs": 5, "text": "Hi is this John?"},
        {"spk": "SPEAKER_01", "t0_abs": 5, "t1_abs": 6, "text": "Yes"},
        {"spk": "SPEAKER_00", "t0_abs": 6, "t1_abs": 12, "text": "Great how are you doing today?"},
    ]

    features = extract_speaker_features(turns, "SPEAKER_00", call_start_abs=0)

    assert features["turn_count"] == 2
    assert features["first_speaker"] == 1
    assert features["question_turn_rate"] == 1.0  # Both turns have questions


def test_extract_call_features_difference():
    """Call features are differences between speakers."""
    turns = [
        {"spk": "SPEAKER_00", "t0_abs": 0, "t1_abs": 5, "text": "Hello there"},
        {"spk": "SPEAKER_01", "t0_abs": 5, "t1_abs": 6, "text": "Hi"},
    ]

    features = extract_call_features("call_1", turns, call_start_abs=0)

    assert "diff_talk_time" in features
    assert features["diff_talk_time"] == 4.0  # 5s - 1s


def test_extract_all_speaker_features():
    """Extract features for all speakers, not just SPEAKER_00/01."""
    from pipeline.role_features import extract_all_speaker_features

    turns = [
        {"spk": "SPEAKER_00", "t0_abs": 0, "t1_abs": 10, "text": "Hi, this is Mike calling about your listing. How are you today?"},
        {"spk": "SPEAKER_01", "t0_abs": 10, "t1_abs": 12, "text": "Good thanks"},
        {"spk": "SPEAKER_02", "t0_abs": 12, "t1_abs": 25, "text": "So you can see here the agent is building rapport. This is a great technique to use."},
        {"spk": "SPEAKER_00", "t0_abs": 25, "t1_abs": 32, "text": "I noticed your listing expired last week. What happened there?"},
        {"spk": "SPEAKER_01", "t0_abs": 32, "t1_abs": 34, "text": "Yeah unfortunately"},
    ]
    features = extract_all_speaker_features(turns, call_start_abs=0)

    # Should have features for all 3 speakers
    assert len(features) == 3
    assert "SPEAKER_00" in features
    assert "SPEAKER_01" in features
    assert "SPEAKER_02" in features

    # SPEAKER_00 (agent pattern): more questions, moderate talk time
    assert features["SPEAKER_00"]["question_turn_rate"] > 0  # Asked questions
    assert features["SPEAKER_00"]["turn_count"] == 2

    # SPEAKER_01 (user pattern): short turns, no questions
    assert features["SPEAKER_01"]["question_turn_rate"] == 0
    assert features["SPEAKER_01"]["avg_turn_duration"] < 3  # Short turns

    # SPEAKER_02 (narrator pattern): long turns, no questions, doesn't interact
    assert features["SPEAKER_02"]["turn_count"] == 1
    assert features["SPEAKER_02"]["talk_time"] > 10  # Long monologue
    assert features["SPEAKER_02"]["avg_turn_duration"] > 10


def test_extract_all_speaker_features_2_speakers():
    """Works with 2 speakers (backward compatible)."""
    from pipeline.role_features import extract_all_speaker_features

    turns = [
        {"spk": "SPEAKER_00", "t0_abs": 0, "t1_abs": 5, "text": "Hello there"},
        {"spk": "SPEAKER_01", "t0_abs": 5, "t1_abs": 7, "text": "Hi"},
    ]
    features = extract_all_speaker_features(turns, call_start_abs=0)

    assert len(features) == 2
    assert "SPEAKER_00" in features
    assert "SPEAKER_01" in features
