# scripts/whisperx_pipeline/lexical_matcher.py
"""Fuzzy lexical matching for call boundary detection."""
import re
from typing import List
from difflib import SequenceMatcher


def tokenize(text: str) -> List[str]:
    """Normalize text to lowercase tokens, strip punctuation."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def token_similarity(tokens_a: List[str], tokens_b: List[str]) -> float:
    """
    Compute similarity between two token sequences.
    Uses both exact match ratio and fuzzy character matching.
    """
    if not tokens_a or not tokens_b:
        return 0.0

    # Exact token overlap
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    overlap = len(set_a & set_b)
    union = len(set_a | set_b)
    jaccard = overlap / union if union > 0 else 0

    # Sequential match (order matters for patterns)
    seq_matches = 0.0
    for i, tok_a in enumerate(tokens_a):
        if i < len(tokens_b):
            tok_b = tokens_b[i]
            if tok_a == tok_b:
                seq_matches += 1
            elif SequenceMatcher(None, tok_a, tok_b).ratio() > 0.8:
                seq_matches += 0.8  # Partial credit for similar tokens

    seq_ratio = seq_matches / max(len(tokens_a), len(tokens_b))

    # Combine both metrics
    return (jaccard + seq_ratio) / 2


class FuzzyLexicalMatcher:
    """Match greeting/reset patterns with fuzzy tolerance for ASR errors."""

    def __init__(self, patterns: List[List[str]], threshold: float = 0.7):
        self.patterns = patterns
        self.threshold = threshold

    def find_matches(self, words: List[dict]) -> List[dict]:
        """
        Find all pattern matches in word sequence.
        Returns list of matches with pattern, timestamp, and confidence.
        """
        matches = []

        for i in range(len(words)):
            for pattern in self.patterns:
                # Extract window of tokens
                window_size = len(pattern) + 1  # Allow one extra for flexibility
                window_words = words[i:i + window_size]

                if len(window_words) < len(pattern):
                    continue

                window_tokens = [tokenize(w.get("text", ""))[0] if tokenize(w.get("text", "")) else ""
                                for w in window_words]

                # Try different alignments within window
                best_sim = 0.0
                for offset in range(min(2, len(window_tokens) - len(pattern) + 1)):
                    candidate = window_tokens[offset:offset + len(pattern)]
                    sim = token_similarity(pattern, candidate)
                    best_sim = max(best_sim, sim)

                if best_sim >= self.threshold:
                    matches.append({
                        "pattern": pattern,
                        "t_abs": words[i]["t0_abs"],
                        "confidence": best_sim,
                        "word_index": i
                    })
                    break  # Only match one pattern per position

        return matches

    def find_greeting_resets(
        self,
        words: List[dict],
        min_gap_from_start_s: float = 30.0
    ) -> List[dict]:
        """
        Find greeting patterns that indicate a new call started.
        Only considers matches after min_gap_from_start_s.
        """
        if not words:
            return []

        start_t = words[0]["t0_abs"]
        all_matches = self.find_matches(words)

        # Filter to matches after the initial greeting window
        resets = [
            m for m in all_matches
            if m["t_abs"] > start_t + min_gap_from_start_s
        ]

        return resets
