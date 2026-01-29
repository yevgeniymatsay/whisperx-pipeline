# tests/whisperx_pipeline/test_turn_builder.py
import pytest
from pipeline.turn_builder import TurnBuilder, Turn


def test_turn_dataclass():
    """Turn has required fields."""
    turn = Turn(
        turn_id=0,
        spk="SPEAKER_00",
        text="Hello there",
        t0_abs=0.0,
        t1_abs=1.5,
        word_span_start=0,
        word_span_end=2
    )
    assert turn.text == "Hello there"


def test_builder_groups_consecutive_words():
    """Builder groups consecutive same-speaker words into turns."""
    builder = TurnBuilder()

    words = [
        {"spk": "SPEAKER_00", "text": "Hi", "t0_abs": 0.0, "t1_abs": 0.3},
        {"spk": "SPEAKER_00", "text": "there", "t0_abs": 0.3, "t1_abs": 0.6},
        {"spk": "SPEAKER_01", "text": "Hello", "t0_abs": 0.8, "t1_abs": 1.1},
        {"spk": "SPEAKER_00", "text": "How", "t0_abs": 1.3, "t1_abs": 1.5},
        {"spk": "SPEAKER_00", "text": "are", "t0_abs": 1.5, "t1_abs": 1.7},
        {"spk": "SPEAKER_00", "text": "you", "t0_abs": 1.7, "t1_abs": 2.0},
    ]

    turns = builder.build_turns(words)

    assert len(turns) == 3
    assert turns[0].text == "Hi there"
    assert turns[0].spk == "SPEAKER_00"
    assert turns[1].text == "Hello"
    assert turns[1].spk == "SPEAKER_01"
    assert turns[2].text == "How are you"


def test_builder_tracks_word_spans():
    """Builder correctly tracks word span indices."""
    builder = TurnBuilder()

    words = [
        {"spk": "SPEAKER_00", "text": "One", "t0_abs": 0.0, "t1_abs": 0.3},
        {"spk": "SPEAKER_00", "text": "two", "t0_abs": 0.3, "t1_abs": 0.6},
        {"spk": "SPEAKER_01", "text": "Three", "t0_abs": 0.8, "t1_abs": 1.1},
    ]

    turns = builder.build_turns(words)

    assert turns[0].word_span_start == 0
    assert turns[0].word_span_end == 2  # exclusive
    assert turns[1].word_span_start == 2
    assert turns[1].word_span_end == 3


def test_builder_merges_small_gaps():
    """Builder merges same-speaker words with small gaps."""
    builder = TurnBuilder(max_gap_s=0.3)

    # Same speaker with small gap (0.2s) - should merge
    words = [
        {"spk": "SPEAKER_00", "text": "Hi", "t0_abs": 0.0, "t1_abs": 0.3},
        {"spk": "SPEAKER_00", "text": "there", "t0_abs": 0.5, "t1_abs": 0.8},  # 0.2s gap
    ]

    turns = builder.build_turns(words)
    assert len(turns) == 1
    assert turns[0].text == "Hi there"


def test_builder_splits_large_gaps():
    """Builder splits same-speaker words with large gaps (diarization artifacts)."""
    builder = TurnBuilder(max_gap_s=0.3)

    # Same speaker but large gap (2s) - should split
    words = [
        {"spk": "SPEAKER_00", "text": "First sentence", "t0_abs": 0.0, "t1_abs": 1.0},
        {"spk": "SPEAKER_00", "text": "Second sentence", "t0_abs": 3.0, "t1_abs": 4.0},  # 2s gap
    ]

    turns = builder.build_turns(words)
    assert len(turns) == 2
    assert turns[0].text == "First sentence"
    assert turns[1].text == "Second sentence"
