"""Generate synthetic system prompts for a conversation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .azure_client import AzureGPTClient
from .constants import (
    ANALYSIS_TEMPLATE_VERSION,
    PROMPT_TEMPLATE_VERSION,
    SCHEMA_VERSION,
    STYLES,
)
from .text_metrics import compute_topic_clarity
from .validator import validate_system_prompt


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_persona_fact_count(persona_facts: List[Dict[str, Any]]) -> int:
    return sum(1 for f in persona_facts if isinstance(f.get("evidence_turn_ids"), list) and f["evidence_turn_ids"])


def compute_persona_strength(persona_fact_count: int) -> float:
    return min(1.0, max(0.0, persona_fact_count / 5.0))


def recommend_level(
    *,
    persona_fact_count: int,
    topic_clarity: float,
    topic_clarity_threshold: float = 0.5,
) -> str:
    if persona_fact_count >= 3:
        return "highly_detailed"
    if topic_clarity < topic_clarity_threshold:
        return "topic_specific"
    return "minimal"


@dataclass
class GenerationResult:
    data: Dict[str, Any]


def generate_prompt_object(
    *,
    turns: List[Dict[str, str]],
    source_bucket: str,
    source_key: str,
    call_id: str,
    client: Optional[AzureGPTClient] = None,
    topic_clarity_threshold: float = 0.5,
) -> GenerationResult:
    """Generate analysis + 3 system prompts and return the S3 object payload."""
    client = client or AzureGPTClient()

    analyzed = client.analyze_transcript(turns)
    topic = (analyzed.get("topic") or "").strip() or "general conversation"
    persona_facts = analyzed.get("persona_facts") or []
    safety_notes = analyzed.get("safety_notes") or []

    topic_clarity = compute_topic_clarity(turns)
    persona_fact_count = compute_persona_fact_count(persona_facts)
    persona_strength = compute_persona_strength(persona_fact_count)
    recommended_level = recommend_level(
        persona_fact_count=persona_fact_count,
        topic_clarity=topic_clarity,
        topic_clarity_threshold=topic_clarity_threshold,
    )

    prompts: Dict[str, str] = {}
    prompts_info: Dict[str, Any] = {}

    for style in STYLES:
        prompt = client.generate_system_prompt(
            turns=turns,
            style=style,
            topic=topic,
            allowed_persona_facts=persona_facts,
            temperature=0.0,
        )
        validation = validate_system_prompt(prompt=prompt, style=style, topic=topic)

        attempts = 1
        repaired = False

        if not validation.valid:
            repaired = True
            repaired_prompt = client.repair_system_prompt(
                turns=turns,
                style=style,
                topic=topic,
                allowed_persona_facts=persona_facts,
                bad_prompt=prompt,
                violations=validation.violations,
            )
            attempts = 2
            validation = validate_system_prompt(prompt=repaired_prompt, style=style, topic=topic)
            prompt = repaired_prompt

        prompts[style] = prompt
        prompts_info[style] = {
            "valid": validation.valid,
            "violations": validation.violations,
            "stats": validation.stats,
            "attempts": attempts,
            "repaired": repaired,
        }

    now = _utc_now_iso()

    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "bucket": source_bucket,
            "key": source_key,
            "call_id": call_id,
        },
        "turns": turns,
        "analysis": {
            "topic": topic,
            "topic_clarity": topic_clarity,
            "persona_facts": persona_facts,
            "persona_fact_count": persona_fact_count,
            "persona_strength": persona_strength,
            "recommended_level": recommended_level,
            "safety_notes": safety_notes,
        },
        "prompts": {
            "minimal": prompts["minimal"],
            "topic_specific": prompts["topic_specific"],
            "highly_detailed": prompts["highly_detailed"],
        },
        "prompts_info": prompts_info,
        "qa": {
            "minimal": {"candidates": [], "preferences": []},
            "topic_specific": {"candidates": [], "preferences": []},
            "highly_detailed": {"candidates": [], "preferences": []},
        },
        "generator_meta": {
            "provider": "azure_openai",
            "deployment": client.config.deployment,
            "api_version": client.config.api_version,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "analysis_template_version": ANALYSIS_TEMPLATE_VERSION,
            "created_at": now,
            "updated_at": now,
        },
    }

    return GenerationResult(data=payload)

