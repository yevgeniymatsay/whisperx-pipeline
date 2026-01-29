"""GenRM judge module for SFT data quality evaluation.

Uses nvidia/Qwen3-Nemotron-32B-GenRM-Principle to score call transcripts
for training data suitability via principle-based evaluation.
"""
from .config import GenRMConfig
from .scorer import judgment_to_score, aggregate_scores
from .router import GenRMRouter, RoutingDecision

__all__ = [
    "GenRMConfig",
    "judgment_to_score",
    "aggregate_scores",
    "GenRMRouter",
    "RoutingDecision",
]
