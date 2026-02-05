"""Synthetic system prompt generation + QA tooling.

This module builds a post-GenRM dataset that annotates real call transcripts with
synthetic system prompts (minimal/topic-specific/highly-detailed) and optional
human preference labels for later preference tuning (e.g., DPO).
"""

