"""Principle definitions for GenRM evaluation.

These principles define the quality criteria for SFT training data.
Each principle is evaluated separately to get calibrated Yes/No scores.
"""

# Principle prompts for GenRM evaluation
# Each asks a Yes/No question about a specific quality dimension

PRINCIPLES = {
    "integrity": """You are evaluating a cold call transcript for data integrity.

Evaluate whether this transcript represents an AUTHENTIC conversation:
- The dialogue flows naturally without obvious fabrication
- Speaker turns are correctly attributed (agent vs lead)
- No turns appear merged, split incorrectly, or missing
- The transcript appears to be a real recorded conversation

Based on the conversation below, answer ONLY "Yes" or "No":
Does this transcript have high data integrity suitable for training?""",

    "sft_value": """You are evaluating a cold call transcript for supervised fine-tuning value.

Evaluate whether this conversation would be VALUABLE for training a cold call AI agent:
- The agent demonstrates techniques worth learning (objection handling, rapport building)
- The conversation has enough substance (not just "hello/goodbye")
- The lead's responses provide realistic training signal
- This would help an AI learn cold calling patterns

Based on the conversation below, answer ONLY "Yes" or "No":
Is this conversation valuable for training a cold call agent?""",

    "competence": """You are evaluating a cold call transcript for agent competence.

Evaluate whether the AGENT in this call demonstrates professional competence:
- Maintains professional tone throughout
- Handles objections or questions appropriately
- Follows reasonable cold call structure (intro, purpose, engagement)
- Does not say anything inappropriate or unprofessional

Based on the conversation below, answer ONLY "Yes" or "No":
Does the agent demonstrate acceptable professional competence?""",
}


def get_principle_prompt(principle_name: str) -> str:
    """Get the prompt for a specific principle."""
    if principle_name not in PRINCIPLES:
        raise ValueError(
            f"Unknown principle: {principle_name}. "
            f"Available: {list(PRINCIPLES.keys())}"
        )
    return PRINCIPLES[principle_name]


def get_all_principles() -> dict:
    """Get all principle definitions."""
    return PRINCIPLES.copy()
