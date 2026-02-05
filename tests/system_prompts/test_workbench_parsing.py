# tests/system_prompts/test_workbench_parsing.py

import pytest

from pipeline.system_prompts.workbench.parsing import (
    parse_messages_json,
    normalize_transcript_messages,
    parse_request_json,
)


def test_parse_messages_json_accepts_list_and_normalizes():
    text = """
    [
      {"role": "assistant", "content": " hi "},
      {"role": "user", "content": " ok "},
      {"role": "other", "content": "drop me"},
      {"role": "system", "content": " keep system "}
    ]
    """
    messages = parse_messages_json(text)
    assert messages == [
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "ok"},
        {"role": "system", "content": "keep system"},
    ]


def test_parse_messages_json_accepts_object_with_messages():
    text = """
    {
      "messages": [
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Hi"}
      ]
    }
    """
    messages = parse_messages_json(text)
    assert len(messages) == 2
    assert messages[0]["role"] == "assistant"

def test_parse_messages_json_accepts_object_with_turns():
    text = """
    {
      "turns": [
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Hi"}
      ],
      "scalar_score": 0.7
    }
    """
    messages = parse_messages_json(text)
    assert len(messages) == 2
    assert messages[1]["role"] == "user"


def test_normalize_transcript_messages_drops_system_and_requires_both_roles():
    messages = [
        {"role": "system", "content": "x"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Hi"},
    ]
    turns = normalize_transcript_messages(messages)
    assert turns == [
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Hi"},
    ]

    with pytest.raises(ValueError, match="both assistant and user"):
        normalize_transcript_messages([{"role": "assistant", "content": "Only one side"}])


def test_parse_request_json_validates_shape():
    text = """
    {
      "schema_type": "analysis",
      "messages": [{"role":"system","content":"x"},{"role":"user","content":"y"}],
      "temperature": 0,
      "max_tokens": 123
    }
    """
    req = parse_request_json(text)
    assert req["schema_type"] == "analysis"
    assert req["temperature"] == 0.0
    assert req["max_tokens"] == 123

    with pytest.raises(ValueError, match="missing keys"):
        parse_request_json('{"schema_type":"analysis"}')
