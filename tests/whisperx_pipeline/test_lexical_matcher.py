# tests/whisperx_pipeline/test_lexical_matcher.py
import pytest
from pipeline.lexical_matcher import (
    FuzzyLexicalMatcher,
    tokenize,
    token_similarity
)


def test_tokenize_normalizes():
    """Tokenize lowercases and strips punctuation."""
    assert tokenize("Hi, is this John?") == ["hi", "is", "this", "john"]
    assert tokenize("Hello! My name is") == ["hello", "my", "name", "is"]


def test_token_similarity_exact():
    """Exact token match has similarity 1.0."""
    assert token_similarity(["hi", "is", "this"], ["hi", "is", "this"]) == 1.0


def test_token_similarity_partial():
    """Partial overlap has proportional similarity."""
    # 2 out of 3 tokens match
    sim = token_similarity(["hi", "is", "this"], ["hi", "is", "that"])
    # Jaccard: 2/4 = 0.5, Sequential: 2/3 = 0.666, Combined: ~0.58
    assert 0.5 < sim < 0.7


def test_fuzzy_matcher_finds_greeting():
    """Fuzzy matcher finds greeting despite ASR variance."""
    matcher = FuzzyLexicalMatcher(
        patterns=[["hi", "is", "this"], ["hello", "is", "this"]],
        threshold=0.7
    )

    # ASR might transcribe "Hi, is this John" as "Hi is this John" or "Hi, this is John"
    words = [
        {"text": "Hi", "t0_abs": 0.0},
        {"text": "is", "t0_abs": 0.2},
        {"text": "this", "t0_abs": 0.4},
        {"text": "John", "t0_abs": 0.6},
    ]

    matches = matcher.find_matches(words)
    assert len(matches) == 1
    assert matches[0]["pattern"] == ["hi", "is", "this"]
    assert matches[0]["t_abs"] == 0.0


def test_fuzzy_matcher_handles_asr_variance():
    """Matcher tolerates ASR errors like 'high' instead of 'hi'."""
    matcher = FuzzyLexicalMatcher(
        patterns=[["hi", "is", "this"]],
        threshold=0.55  # Lower threshold to catch ASR variance
    )

    # ASR transcribed "hi" as "high"
    words = [
        {"text": "High", "t0_abs": 0.0},
        {"text": "is", "t0_abs": 0.2},
        {"text": "this", "t0_abs": 0.4},
    ]

    matches = matcher.find_matches(words)
    # With 2/3 exact + partial fuzzy credit, sim ~0.58
    assert len(matches) >= 1
