"""Export synthetic system prompt datasets to JSONL formats."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _stable_hash(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def preferred_style_for_object(
    obj: Dict[str, Any],
    *,
    topic_clarity_threshold: float = 0.5,
) -> str:
    analysis = obj.get("analysis") or {}
    persona_fact_count = int(analysis.get("persona_fact_count") or 0)
    topic_clarity = float(analysis.get("topic_clarity") or 0.0)

    if persona_fact_count >= 3:
        return "highly_detailed"
    if topic_clarity < topic_clarity_threshold:
        return "topic_specific"
    return "minimal"


def choose_styles_with_detailed_cap(
    objs: List[Dict[str, Any]],
    *,
    topic_clarity_threshold: float = 0.5,
    detailed_cap: float = 0.15,
) -> Dict[str, str]:
    """Return {call_id: chosen_style} with a global cap on highly_detailed."""
    if not 0.0 <= detailed_cap <= 1.0:
        raise ValueError("detailed_cap must be in [0, 1]")

    total = len(objs)
    allow_detailed = int(total * detailed_cap)

    preferred: Dict[str, str] = {}
    detailed_candidates: List[Tuple[str, str]] = []

    for obj in objs:
        call_id = (obj.get("source") or {}).get("call_id") or obj.get("call_id") or ""
        if not call_id:
            raise ValueError("Object missing call_id in source.call_id")

        p = preferred_style_for_object(obj, topic_clarity_threshold=topic_clarity_threshold)
        preferred[call_id] = p
        if p == "highly_detailed":
            detailed_candidates.append((call_id, _stable_hash(call_id)))

    detailed_candidates.sort(key=lambda x: x[1])
    keep_detailed = {cid for cid, _ in detailed_candidates[:allow_detailed]}

    chosen: Dict[str, str] = {}
    for obj in objs:
        call_id = (obj.get("source") or {}).get("call_id") or obj.get("call_id")
        p = preferred[call_id]
        if p == "highly_detailed" and call_id not in keep_detailed:
            chosen[call_id] = "topic_specific"
        else:
            chosen[call_id] = p

    return chosen


def export_sft_jsonl(
    objs: List[Dict[str, Any]],
    *,
    output_path: Path,
    topic_clarity_threshold: float = 0.5,
    detailed_cap: float = 0.15,
    write_manifest: bool = True,
) -> Dict[str, Any]:
    """Write JSONL where each line is {"messages":[...]}."""
    styles = choose_styles_with_detailed_cap(
        objs,
        topic_clarity_threshold=topic_clarity_threshold,
        detailed_cap=detailed_cap,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    manifest: Dict[str, str] = {}

    with output_path.open("w") as f:
        for obj in objs:
            call_id = (obj.get("source") or {}).get("call_id") or obj.get("call_id")
            chosen_style = styles[call_id]
            prompts = obj.get("prompts") or {}
            system_prompt = (prompts.get(chosen_style) or "").strip()
            if not system_prompt:
                # Fallback order
                system_prompt = (prompts.get("minimal") or "").strip()
            turns = obj.get("turns") or []

            record = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    *turns,
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            manifest[call_id] = chosen_style

    meta = {
        "written": count,
        "topic_clarity_threshold": topic_clarity_threshold,
        "detailed_cap": detailed_cap,
    }

    if write_manifest:
        manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
        manifest_path.write_text(json.dumps({"meta": meta, "styles": manifest}, indent=2), encoding="utf-8")

    return meta


def export_dpo_jsonl(
    objs: List[Dict[str, Any]],
    *,
    output_path: Path,
) -> Dict[str, Any]:
    """Export preference pairs as provider-agnostic JSONL.

    Each line:
      {"input": {...}, "chosen": "...", "rejected": "...", "meta": {...}}
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0

    with output_path.open("w") as f:
        for obj in objs:
            source = obj.get("source") or {}
            call_id = source.get("call_id") or obj.get("call_id") or ""
            turns = obj.get("turns") or []
            gen_meta = obj.get("generator_meta") or {}
            template_version = gen_meta.get("prompt_template_version")

            qa = obj.get("qa") or {}
            for style, style_qa in qa.items():
                candidates = style_qa.get("candidates") or []
                by_id = {c.get("id"): c for c in candidates if c.get("id")}

                prefs = style_qa.get("preferences") or []
                for pref in prefs:
                    chosen_id = pref.get("chosen_id")
                    rejected_id = pref.get("rejected_id")
                    if not chosen_id or not rejected_id:
                        continue
                    chosen_c = by_id.get(chosen_id)
                    rejected_c = by_id.get(rejected_id)
                    if not chosen_c or not rejected_c:
                        continue

                    chosen_prompt = (chosen_c.get("prompt") or "").strip()
                    rejected_prompt = (rejected_c.get("prompt") or "").strip()
                    if not chosen_prompt or not rejected_prompt:
                        continue

                    record = {
                        "input": {
                            "prompt_template_version": template_version,
                            "style": style,
                            "transcript_messages": turns,
                        },
                        "chosen": chosen_prompt,
                        "rejected": rejected_prompt,
                        "meta": {
                            "call_id": call_id,
                            "rater": pref.get("rater"),
                            "rated_at": pref.get("rated_at"),
                            "notes": pref.get("notes"),
                        },
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1

    return {"written": written}

