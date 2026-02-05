"""JSON Schemas for Structured Outputs."""

from __future__ import annotations

from typing import Dict, Any


def analysis_response_format() -> Dict[str, Any]:
    """Schema for transcript analysis output."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "system_prompt_analysis",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "topic": {"type": "string"},
                    "persona_facts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "speaker": {"type": "string", "enum": ["assistant", "user"]},
                                "category": {
                                    "type": "string",
                                    "enum": ["job", "location", "preference", "background", "constraint", "other"],
                                },
                                "fact": {"type": "string"},
                                "evidence_turn_ids": {
                                    "type": "array",
                                    "items": {"type": "integer", "minimum": 0},
                                },
                            },
                            "required": ["speaker", "category", "fact", "evidence_turn_ids"],
                        },
                    },
                    "safety_notes": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["topic", "persona_facts", "safety_notes"],
            },
        },
    }


def prompt_response_format() -> Dict[str, Any]:
    """Schema for a single system prompt generation output."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "system_prompt_single",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "system_prompt": {"type": "string"},
                },
                "required": ["system_prompt"],
            },
        },
    }

