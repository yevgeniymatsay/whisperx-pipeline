# tests/system_prompts/test_export.py

import json

from pipeline.system_prompts.export import export_dpo_jsonl, export_sft_jsonl


def test_export_sft_jsonl_writes_messages(tmp_path):
    objs = [
        {
            "source": {"call_id": "c1"},
            "analysis": {"persona_fact_count": 0, "topic_clarity": 0.9},
            "prompts": {
                "minimal": "You have a natural phone conversation.",
                "topic_specific": "You have a natural phone conversation about scheduling a follow-up.",
                "highly_detailed": "You have a natural phone conversation. Keep it friendly and concise.",
            },
            "turns": [
                {"role": "assistant", "content": "Hi, is now a bad time?"},
                {"role": "user", "content": "Uh, who is this?"},
            ],
        }
    ]

    out = tmp_path / "out.jsonl"
    meta = export_sft_jsonl(objs, output_path=out, detailed_cap=0.0)
    assert meta["written"] == 1

    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["messages"][0]["role"] == "system"
    assert record["messages"][1]["role"] == "assistant"


def test_export_dpo_jsonl_writes_pairs(tmp_path):
    obj = {
        "source": {"call_id": "c1"},
        "generator_meta": {"prompt_template_version": "sp-v1"},
        "turns": [
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Hi"},
        ],
        "qa": {
            "minimal": {
                "candidates": [
                    {"id": "a", "prompt": "You have a natural phone conversation."},
                    {"id": "b", "prompt": "As an AI language model, you chat."},
                ],
                "preferences": [
                    {"chosen_id": "a", "rejected_id": "b", "rater": "test", "rated_at": "now"},
                ],
            }
        },
    }

    out = tmp_path / "pairs.jsonl"
    meta = export_dpo_jsonl([obj], output_path=out)
    assert meta["written"] == 1

    record = json.loads(out.read_text().strip())
    assert record["input"]["style"] == "minimal"
    assert record["chosen"] == "You have a natural phone conversation."

