"""Parsing helpers for the System Prompt Workbench."""

from __future__ import annotations

import json
from typing import Any, Dict, List


ALLOWED_MESSAGE_ROLES = {"assistant", "user", "system"}


def _as_str(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def parse_messages_json(text: str) -> List[Dict[str, str]]:
    """Parse user-provided JSON into a normalized messages array.

    Accepts either:
      - a list of {role, content}
      - an object {"messages": [...]}
    """
    raw_text = (text or "").strip()
    if not raw_text:
        raise ValueError("Input JSON is empty")

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        # Common copy/paste failure modes:
        # - `{"role":...}, {"role":...}` (comma-separated objects) without wrapping in `[...]`.
        # - `{"role":...}, ... }], "scalar_score": 0.7, ... }` (missing `{"turns":[` prefix).
        if e.msg == "Extra data":
            candidates = [
                f"[{raw_text}]",
                '{"turns":[' + raw_text,
                '{"messages":[' + raw_text,
                '{"turns":' + raw_text,
                '{"messages":' + raw_text,
            ]
            last: json.JSONDecodeError | None = None
            for candidate in candidates:
                try:
                    data = json.loads(candidate)
                    break
                except json.JSONDecodeError as ee:
                    last = ee
            else:
                raise ValueError(
                    "Invalid JSON: Extra data. "
                    "Paste either a JSON array of {role, content}, "
                    "or an object with {messages:[...]} / {turns:[...]}. "
                    "Tip: wrap comma-separated message objects in `[...]`."
                ) from (last or e)
        else:
            raise ValueError(f"Invalid JSON: {e.msg}") from e

    if isinstance(data, dict):
        if "messages" in data:
            data = data["messages"]
        elif "turns" in data:
            data = data["turns"]

    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of messages, or an object with 'messages' or 'turns'")

    normalized: List[Dict[str, str]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Message at index {i} must be an object")
        role = _as_str(item.get("role"), field=f"messages[{i}].role").strip()
        content = _as_str(item.get("content"), field=f"messages[{i}].content").strip()
        if not role or not content:
            continue
        if role not in ALLOWED_MESSAGE_ROLES:
            continue
        normalized.append({"role": role, "content": content})

    if not normalized:
        raise ValueError("No valid messages found after normalization")

    return normalized


def normalize_transcript_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Normalize messages into transcript turns used by this workbench.

    - Drops any system turns.
    - Requires at least 2 turns and both assistant + user present.
    """
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")

    turns: List[Dict[str, str]] = []
    roles = set()
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            raise ValueError(f"messages[{i}] must be an object")
        role = _as_str(m.get("role"), field=f"messages[{i}].role").strip()
        content = _as_str(m.get("content"), field=f"messages[{i}].content").strip()
        if role == "system":
            continue
        if role not in {"assistant", "user"}:
            continue
        if not content:
            continue
        turns.append({"role": role, "content": content})
        roles.add(role)

    if roles != {"assistant", "user"}:
        raise ValueError("Transcript must include both assistant and user roles")
    if len(turns) < 2:
        raise ValueError("Need at least 2 assistant/user turns")

    return turns


def parse_request_json(text: str) -> Dict[str, Any]:
    """Parse and validate a raw request JSON blob from the Request Lab editor."""
    raw_text = (text or "").strip()
    if not raw_text:
        raise ValueError("Request JSON is empty")

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid request JSON: {e.msg}") from e

    if not isinstance(data, dict):
        raise ValueError("Request JSON must be an object")

    missing = [k for k in ("schema_type", "messages", "temperature", "max_tokens") if k not in data]
    if missing:
        raise ValueError(f"Request JSON missing keys: {', '.join(missing)}")

    schema_type = _as_str(data.get("schema_type"), field="schema_type").strip()
    if schema_type not in {"analysis", "prompt"}:
        raise ValueError("schema_type must be 'analysis' or 'prompt'")

    messages = data.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages must be an array")
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            raise ValueError(f"messages[{i}] must be an object")
        role = _as_str(m.get("role"), field=f"messages[{i}].role").strip()
        content = _as_str(m.get("content"), field=f"messages[{i}].content").strip()
        if not role or not content:
            raise ValueError(f"messages[{i}] role/content cannot be empty")

    temperature = data.get("temperature")
    if not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be a number")

    max_tokens = data.get("max_tokens")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")

    return {
        "schema_type": schema_type,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
