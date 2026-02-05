"""Constants for synthetic system prompt generation."""

SCHEMA_VERSION = "synthetic_system_prompts/v1"

PROMPT_TEMPLATE_VERSION = "sp-v3"
ANALYSIS_TEMPLATE_VERSION = "sp-analysis-v3"

STYLES = ("minimal", "topic_specific", "highly_detailed")

# Basic safety/quality bans to avoid meta-training artifacts.
BANNED_SUBSTRINGS = (
    "SPEAKER_",
    "In this conversation",
    "in this conversation",
    "This conversation",
    "this conversation",
    "Conversation is about",
    "conversation is about",
    "Topic:",
    "Style:",
    "As an AI",
    "as an AI",
    "language model",
    "fine-tun",
    "SFT",
    "DPO",
    "system prompt",
    "developer prompt",
)
