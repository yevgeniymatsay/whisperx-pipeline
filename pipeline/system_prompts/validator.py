"""Validation for generated system prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

from .constants import BANNED_SUBSTRINGS
from .text_metrics import word_count, sentence_count


@dataclass
class PromptValidation:
    valid: bool
    violations: List[str]
    stats: Dict[str, Any]


def validate_system_prompt(
    *,
    prompt: str,
    style: str,
    topic: str | None = None,
) -> PromptValidation:
    prompt = (prompt or "").strip()
    violations: List[str] = []

    if not prompt:
        violations.append("empty_prompt")
        return PromptValidation(False, violations, {"word_count": 0, "sentence_count": 0})

    for banned in BANNED_SUBSTRINGS:
        if banned in prompt:
            violations.append(f"contains_banned_substring:{banned}")
            break

    wc = word_count(prompt)
    sc = sentence_count(prompt)

    # Style constraints
    if style == "minimal":
        if sc != 1:
            violations.append(f"minimal_sentence_count:{sc}")

    elif style == "topic_specific":
        if sc < 2 or sc > 3:
            violations.append(f"topic_sentence_count:{sc}")

    elif style == "highly_detailed":
        if sc < 4 or sc > 10:
            violations.append(f"detailed_sentence_count:{sc}")

    else:
        violations.append(f"unknown_style:{style}")

    return PromptValidation(
        valid=len(violations) == 0,
        violations=violations,
        stats={"word_count": wc, "sentence_count": sc},
    )
