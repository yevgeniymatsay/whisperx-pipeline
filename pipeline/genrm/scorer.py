"""Compute scores from GenRM judgments.

GenRM-Principle models output "Final Judgement: Yes/No" after reasoning.
Scoring is binary: Yes = 1.0, No = 0.0, Unknown = 0.5
"""
from typing import Dict, Optional


def judgment_to_score(judgment: Optional[str]) -> float:
    """
    Convert Yes/No judgment to numeric score.

    Args:
        judgment: "Yes", "No", or None

    Returns:
        1.0 for Yes, 0.0 for No, 0.5 for unknown
    """
    if judgment is None:
        return 0.5  # Unknown/unparseable -> neutral
    return 1.0 if judgment.lower() == "yes" else 0.0


def aggregate_scores(
    principle_scores: Dict[str, Dict[str, float]],
    weights: Dict[str, float],
) -> float:
    """
    Compute weighted aggregate score from principle judgments.

    Args:
        principle_scores: {principle_name: {"judgment": "Yes"|"No", "score": float}}
        weights: {principle_name: weight}

    Returns:
        Weighted average of scores (0.0 to 1.0)
    """
    total = 0.0
    weight_sum = 0.0

    for principle, weight in weights.items():
        if principle in principle_scores:
            score = principle_scores[principle].get("score", 0.5)
            total += weight * score
            weight_sum += weight

    if weight_sum == 0:
        return 0.5  # Neutral if no scores

    return total / weight_sum


def compute_pass_count(principle_scores: Dict[str, Dict[str, float]]) -> tuple[int, int]:
    """
    Count how many principles passed (Yes judgment).

    Args:
        principle_scores: {principle_name: {"judgment": "Yes"|"No", "score": float}}

    Returns:
        (passed_count, total_count)
    """
    passed = sum(
        1 for scores in principle_scores.values()
        if scores.get("judgment") == "Yes"
    )
    return passed, len(principle_scores)
