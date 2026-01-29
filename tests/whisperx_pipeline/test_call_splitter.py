# tests/whisperx_pipeline/test_call_splitter.py
import pytest
from pipeline.call_splitter import CallSplitter, CallBoundary
from pipeline.config import SplitCallsConfig


def test_call_boundary_dataclass():
    """CallBoundary has required fields."""
    boundary = CallBoundary(
        call_id="video_0_30000",
        start_abs=0.0,
        end_abs=30.0,
        boundary_confidence=0.9,
        start_evidence={"type": "start_of_audio"},
        end_evidence={"type": "silence_gap", "gap_s": 3.5}
    )
    assert boundary.boundary_confidence == 0.9


def test_splitter_finds_silence_boundaries():
    """Splitter detects call boundaries at silence gaps."""
    config = SplitCallsConfig(silence_threshold_s=2.5)
    splitter = CallSplitter(config, video_id="test")

    # Two calls with 3s silence gap between them
    # Words are close together within each call (gaps < 2.5s)
    words = [
        # Call 1: 0-5s (words close together)
        {"t0_abs": 0.0, "t1_abs": 0.5, "text": "Hi", "spk": "SPEAKER_00"},
        {"t0_abs": 0.6, "t1_abs": 1.0, "text": "there", "spk": "SPEAKER_00"},
        {"t0_abs": 1.1, "t1_abs": 1.5, "text": "how", "spk": "SPEAKER_00"},
        {"t0_abs": 1.6, "t1_abs": 2.0, "text": "are", "spk": "SPEAKER_00"},
        {"t0_abs": 2.1, "t1_abs": 2.5, "text": "you", "spk": "SPEAKER_00"},
        {"t0_abs": 2.6, "t1_abs": 3.0, "text": "bye", "spk": "SPEAKER_00"},
        # Gap: 3s to 6s (3s silence) triggers split
        # Call 2: 6-10s (words close together)
        {"t0_abs": 6.0, "t1_abs": 6.5, "text": "Hello", "spk": "SPEAKER_00"},
        {"t0_abs": 6.6, "t1_abs": 7.0, "text": "again", "spk": "SPEAKER_00"},
        {"t0_abs": 7.1, "t1_abs": 7.5, "text": "thanks", "spk": "SPEAKER_01"},
    ]

    boundaries = splitter.find_boundaries(words)
    assert len(boundaries) == 2


def test_splitter_finds_greeting_resets():
    """Splitter detects new calls at greeting patterns."""
    config = SplitCallsConfig(
        silence_threshold_s=2.5,
        greeting_tokens=[["hi", "is", "this"], ["hello", "is", "this"]],
        fuzzy_match_threshold=0.7
    )
    splitter = CallSplitter(config, video_id="test")

    # One continuous audio with greeting reset mid-stream (no long silences)
    # All gaps < 2.5s so only greeting detection can trigger split
    words = []
    t = 0.0
    # Call 1: First 35 seconds of continuous talk (random words that won't match greeting)
    random_words = ["okay", "yeah", "sure", "right", "good", "fine", "yes", "no", "well", "so"]
    for i in range(70):  # ~0.5s per word = 35s
        word = random_words[i % len(random_words)]
        words.append({"t0_abs": t, "t1_abs": t + 0.4, "text": word, "spk": "SPEAKER_00"})
        t += 0.5
    # Greeting reset at ~35s (after 30s min_gap_from_start)
    words.append({"t0_abs": t, "t1_abs": t + 0.3, "text": "Hi", "spk": "SPEAKER_00"})
    t += 0.4
    words.append({"t0_abs": t, "t1_abs": t + 0.2, "text": "is", "spk": "SPEAKER_00"})
    t += 0.3
    words.append({"t0_abs": t, "t1_abs": t + 0.3, "text": "this", "spk": "SPEAKER_00"})
    t += 0.4
    words.append({"t0_abs": t, "t1_abs": t + 0.3, "text": "Mary", "spk": "SPEAKER_00"})

    boundaries = splitter.find_boundaries(words)
    assert len(boundaries) == 2
    # Verify at least one boundary was found via greeting reset
    greeting_reset_found = any(b.start_evidence.get("type") == "greeting_reset" for b in boundaries)
    assert greeting_reset_found, "Expected greeting_reset boundary"
