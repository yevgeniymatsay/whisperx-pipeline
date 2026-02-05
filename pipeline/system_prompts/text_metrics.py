"""Deterministic transcript metrics used for prompt-style sampling."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Dict


_WORD_RE = re.compile(r"[A-Za-z']+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "he", "her", "hers", "him", "his", "i", "if", "in", "is",
    "it", "its", "me", "my", "no", "not", "of", "on", "or", "our", "ours",
    "she", "so", "that", "the", "their", "theirs", "them", "then", "there",
    "these", "they", "this", "to", "up", "us", "we", "were", "what", "when",
    "where", "who", "why", "with", "you", "your", "yours",
}


def _tokens_from_text(text: str) -> List[str]:
    tokens = [t.lower() for t in _WORD_RE.findall(text or "")]
    return [t for t in tokens if len(t) >= 3 and t not in _STOPWORDS]


def compute_topic_clarity(turns: Iterable[Dict[str, str]], top_k: int = 10) -> float:
    """Compute a cheap topic-clarity proxy from lexical concentration.

    Returns a value in [0, 1]. Higher means the transcript repeatedly returns
    to a smaller set of tokens (more "anchored" topic), while lower means the
    transcript is lexically diffuse (more tangents).
    """
    all_tokens: List[str] = []
    for t in turns:
        all_tokens.extend(_tokens_from_text(t.get("content", "")))

    if not all_tokens:
        return 0.0

    counts = Counter(all_tokens)
    total = sum(counts.values())
    top_sum = sum(c for _, c in counts.most_common(top_k))
    return max(0.0, min(1.0, top_sum / max(1, total)))


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def sentence_count(text: str) -> int:
    # Very simple heuristic that behaves well enough for our constraints.
    parts = re.split(r"[.!?]+", (text or "").strip())
    return sum(1 for p in parts if p.strip())

