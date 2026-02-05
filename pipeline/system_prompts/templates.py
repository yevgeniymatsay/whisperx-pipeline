"""Prompt templates for analysis and style-conditioned system prompt generation."""

from __future__ import annotations

import json
from typing import Dict, List, Any

from .constants import PROMPT_TEMPLATE_VERSION, ANALYSIS_TEMPLATE_VERSION


def _render_turns_json(turns: List[Dict[str, str]], max_turns: int | None = None) -> str:
    """Render turns as JSON (multi-turn chat format) for model input.

    We keep the JSON structure so downstream prompting doesn't rely on a
    lossy plaintext transcript format.
    """
    normalized: List[Dict[str, str]] = []
    sliced = turns[:max_turns] if max_turns else turns
    for t in sliced:
        role = (t.get("role") or "").strip()
        content = (t.get("content") or "").strip()
        if not role or not content:
            continue
        normalized.append({"role": role, "content": content})
    return json.dumps(normalized, ensure_ascii=False, indent=2)


def analysis_messages(turns: List[Dict[str, str]]) -> List[Dict[str, str]]:
    turns_json = _render_turns_json(turns)
    return [
        {
            "role": "system",
            "content": (
                f"You analyze multi-turn phone-call messages and extract a helpful topic hint plus explicit persona facts.\n"
                f"Return ONLY valid JSON that matches the provided schema.\n"
                f"Rules:\n"
                f"- Do not invent facts.\n"
                f"- Every persona fact must be explicitly supported by the provided turns.\n"
                f"- Provide evidence_turn_ids as the turn indexes (array positions) that support the fact.\n"
                f"- Extract facts about EITHER speaker (assistant or user) when explicitly stated.\n"
                f"- Facts should be short, atomic, and directly quotable from the turns.\n"
                f"- Topic should be a short phrase that describes what the call is about AND what the assistant is trying to do.\n"
                f"- Phrase the topic in abstract terms (e.g., \"follow-up about a possible sale and timing\"), not a generic label like \"general conversation\".\n"
                f"Template version: {ANALYSIS_TEMPLATE_VERSION}\n"
            ),
        },
        {
            "role": "user",
            "content": (
                "Transcript messages (JSON array; index = turn_id):\n"
                f"{turns_json}\n"
            ),
        },
    ]


def style_requirements(style: str) -> str:
    if style == "minimal":
        return (
            "Write a minimal system prompt.\n"
            "- 1 sentence.\n"
            "- Keep it abstract and broadly reusable (do not name the domain/topic; avoid specific names/addresses).\n"
            "- Capture the assistant's overall conversational vibe and approach (mirror the assistant; do not \"improve\" it).\n"
            "- Write like a real human-authored system prompt, not a checklist or meta description.\n"
            "- Do NOT mention being an AI, a model, transcripts, or system prompts.\n"
        )
    if style == "topic_specific":
        return (
            "Write a topic-specific system prompt.\n"
            "- 2 to 3 sentences.\n"
            "- Describe what the call is about in abstract terms (e.g., a potential sale, timing, a follow-up, scheduling a next step), not an industry label.\n"
            "- State a gentle, high-level goal for how the assistant should move the call forward.\n"
            "- Mirror the assistant's tone/energy/assertiveness from the turns (do not soften or intensify).\n"
            "- Write naturally (no headings/labels) and avoid meta phrasing about the conversation.\n"
            "- Do NOT mention being an AI, a model, transcripts, or system prompts.\n"
        )
    if style == "highly_detailed":
        return (
            "Write a highly detailed system prompt.\n"
            "- 4 to 10 sentences.\n"
            "- Mirror the assistant's actual behavior and conversational style from the turns.\n"
            "- You MAY include specific names/addresses/persona/background/preferences ONLY if explicitly supported by the turns.\n"
            "- Include the key conversation arc and tactics the assistant uses (opening, objections, value, next step).\n"
            "- Keep it natural and aligned with the transcript's vibe; write like a human-authored system prompt.\n"
            "- Do NOT mention being an AI, a model, transcripts, or system prompts.\n"
        )
    raise ValueError(f"Unknown style: {style}")


def generation_messages(
    *,
    turns: List[Dict[str, str]],
    style: str,
    topic: str,
    allowed_persona_facts: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    turns_json = _render_turns_json(turns)
    allowed_facts_text = "\n".join(
        f"- ({f.get('speaker')}/{f.get('category')}) {f.get('fact')}"
        for f in allowed_persona_facts
    ) or "(none)"

    return [
        {
            "role": "system",
            "content": (
                "You write system prompts that would make an assistant behave like the assistant speaker in the provided messages.\n"
                "The goal is to reproduce the assistant's behavior and conversational dynamics (not to make it \"better\").\n"
                "Return ONLY valid JSON that matches the provided schema.\n"
                f"Template version: {PROMPT_TEMPLATE_VERSION}\n"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Requested detail level: {style}\n"
                f"Topic hint (use abstract phrasing; do not use headings/labels): {topic}\n"
                "Allowed persona facts (optional; use ONLY if helpful and supported):\n"
                f"{allowed_facts_text}\n\n"
                f"{style_requirements(style)}\n"
                "Additional rules:\n"
                "- Do not copy long exact phrases from the transcript.\n"
                "- Do not include speaker labels (e.g., SPEAKER_00).\n"
                "- Do not output bullet lists; write in natural sentences.\n"
                "- Do not mention training, fine-tuning, SFT, or DPO.\n\n"
                "Transcript messages (JSON array; index = turn_id):\n"
                f"{turns_json}\n"
            ),
        },
    ]


def repair_messages(
    *,
    turns: List[Dict[str, str]],
    style: str,
    topic: str,
    allowed_persona_facts: List[Dict[str, Any]],
    bad_prompt: str,
    violations: List[str],
) -> List[Dict[str, str]]:
    turns_json = _render_turns_json(turns)
    allowed_facts_text = "\n".join(
        f"- ({f.get('speaker')}/{f.get('category')}) {f.get('fact')}"
        for f in allowed_persona_facts
    ) or "(none)"

    violations_text = "\n".join(f"- {v}" for v in violations) or "(none)"

    return [
        {
            "role": "system",
            "content": (
                "You rewrite a system prompt to satisfy constraints while preserving the assistant's behavior from the messages.\n"
                "Return ONLY valid JSON that matches the provided schema.\n"
                f"Template version: {PROMPT_TEMPLATE_VERSION}\n"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Requested detail level: {style}\n"
                f"Topic hint: {topic}\n"
                "Allowed persona facts:\n"
                f"{allowed_facts_text}\n\n"
                "Current prompt (bad):\n"
                f"{bad_prompt}\n\n"
                "Violations to fix:\n"
                f"{violations_text}\n\n"
                f"{style_requirements(style)}\n"
                "Transcript messages (JSON array; index = turn_id):\n"
                f"{turns_json}\n"
            ),
        },
    ]
