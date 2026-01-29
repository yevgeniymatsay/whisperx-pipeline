# tests/whisperx_pipeline/test_quality_metrics.py
import pytest
from pipeline.quality_metrics import (
    compute_quality_metrics,
    compute_narrator_ratio,
    detect_speaker_flip,
    QualityMetrics,
    QualityReason,
    _compute_overlap_ratio
)


def test_narrator_ratio_high_for_monologue():
    """Narrator ratio is high when one speaker dominates with long turns."""
    # Speaker 0 talks for 50s in one turn, Speaker 1 says 3 words
    words = [
        {"spk": "SPEAKER_00", "t0_abs": 0.0, "t1_abs": 50.0, "text": "long monologue " * 100},
        {"spk": "SPEAKER_01", "t0_abs": 50.5, "t1_abs": 52.0, "text": "okay thanks bye"},
    ]

    ratio = compute_narrator_ratio(words, duration_s=55.0)
    assert ratio > 0.7  # High narrator ratio


def test_narrator_ratio_low_for_dialogue():
    """Narrator ratio is low for back-and-forth conversation."""
    words = []
    for i in range(20):
        spk = "SPEAKER_00" if i % 2 == 0 else "SPEAKER_01"
        words.append({
            "spk": spk,
            "t0_abs": i * 3.0,
            "t1_abs": i * 3.0 + 2.5,
            "text": f"turn {i} content here"
        })

    ratio = compute_narrator_ratio(words, duration_s=60.0)
    assert ratio < 0.3  # Low narrator ratio


def test_detect_speaker_flip_finds_swap():
    """Detect when speakers swap roles mid-call."""
    # First half: SPEAKER_00 leads (agent pattern)
    # Second half: SPEAKER_01 leads (agent pattern)
    words = [
        # First 30s - SPEAKER_00 asks questions, SPEAKER_01 short answers
        {"spk": "SPEAKER_00", "t0_abs": 0, "t1_abs": 5, "text": "Hi is this John calling about your property?"},
        {"spk": "SPEAKER_01", "t0_abs": 5, "t1_abs": 6, "text": "Yes"},
        {"spk": "SPEAKER_00", "t0_abs": 6, "t1_abs": 12, "text": "Great I wanted to ask about the listing"},
        {"spk": "SPEAKER_01", "t0_abs": 12, "t1_abs": 13, "text": "Okay"},
        # Second 30s - roles flip
        {"spk": "SPEAKER_01", "t0_abs": 30, "t1_abs": 38, "text": "So let me tell you about our services and pricing"},
        {"spk": "SPEAKER_00", "t0_abs": 38, "t1_abs": 39, "text": "Uh huh"},
        {"spk": "SPEAKER_01", "t0_abs": 39, "t1_abs": 48, "text": "We offer comprehensive coverage for your needs"},
        {"spk": "SPEAKER_00", "t0_abs": 48, "t1_abs": 49, "text": "I see"},
    ]

    flip_detected, reason = detect_speaker_flip(words)
    assert flip_detected
    assert reason.type == "speaker_flip_suspected"


def test_quality_metrics_struct():
    """QualityMetrics has all required fields."""
    metrics = QualityMetrics(
        overlap_ratio=0.05,
        speaker_switches_per_min=25.0,
        median_turn_s=2.5,
        micro_turn_ratio=0.1,
        narrator_ratio=0.2,
        speaker_flip_suspected=False
    )
    assert metrics.narrator_ratio == 0.2


def test_overlap_ratio_no_overcount():
    """Overlap ratio uses sweep line to avoid overcounting >2 overlapping segments.

    If 3 segments overlap at t=5-7:
    - Pairwise would count: (A,B) + (A,C) + (B,C) = 6s of overlap (WRONG)
    - Sweep line counts: 2s where active_speakers >= 2 (CORRECT)
    """
    # Three segments all overlapping at t=5-7
    segments = [
        {"t0_abs": 0.0, "t1_abs": 7.0, "spk": "A"},   # 0-7
        {"t0_abs": 5.0, "t1_abs": 10.0, "spk": "B"},  # 5-10
        {"t0_abs": 5.0, "t1_abs": 8.0, "spk": "C"},   # 5-8
    ]

    ratio = _compute_overlap_ratio(segments)
    # Total duration: 0-10 = 10s
    # Overlap time: 5-8 = 3s (where 2+ speakers active)
    # Expected ratio: 3/10 = 0.3
    assert 0.25 < ratio < 0.35, f"Expected ~0.3, got {ratio}"
