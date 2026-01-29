"""GenRM configuration settings."""
import os
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class GenRMConfig:
    """Configuration for GenRM judge."""

    # vLLM server settings
    vllm_base_url: str = field(
        default_factory=lambda: os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    )
    model_name: str = "qwen3-nemotron-32b"

    # Routing thresholds (applied to aggregate p_yes)
    accept_threshold: float = 0.80
    review_threshold: float = 0.60
    # Below review_threshold -> reject

    # S3 prefixes for routed calls
    s3_prefix_accepted: str = "genrm/accepted/"
    s3_prefix_review: str = "genrm/review/"
    s3_prefix_rejected: str = "genrm/rejected/"
    s3_prefix_role_fallback: str = "genrm/role_fallback/"

    # vLLM API settings
    # GenRM uses chain-of-thought reasoning before "Final Judgement: Yes/No"
    max_tokens: int = 1024  # Allow space for reasoning + judgment
    top_logprobs: int = 20  # vLLM max is 20
    temperature: float = 0.0  # Deterministic for evaluation

    # Principle weights for aggregate score
    # integrity: Is the transcript accurate and unmanipulated?
    # sft_value: Is this useful for training a cold call agent?
    # competence: Does the agent demonstrate good sales technique?
    principle_weights: Dict[str, float] = field(default_factory=lambda: {
        "integrity": 0.40,
        "sft_value": 0.35,
        "competence": 0.25,
    })

    def validate(self) -> None:
        """Validate configuration."""
        if self.accept_threshold <= self.review_threshold:
            raise ValueError(
                f"accept_threshold ({self.accept_threshold}) must be > "
                f"review_threshold ({self.review_threshold})"
            )

        weights_sum = sum(self.principle_weights.values())
        if abs(weights_sum - 1.0) > 0.001:
            raise ValueError(
                f"principle_weights must sum to 1.0, got {weights_sum}"
            )


# Default instance
DEFAULT_CONFIG = GenRMConfig()
