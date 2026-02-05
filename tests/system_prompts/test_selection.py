# tests/system_prompts/test_selection.py

from pipeline.system_prompts.export import choose_styles_with_detailed_cap


def test_detailed_cap_limits_highly_detailed_count():
    objs = []
    for i in range(10):
        persona_fact_count = 3 if i < 4 else 0
        objs.append(
            {
                "source": {"call_id": f"call_{i}"},
                "analysis": {
                    "persona_fact_count": persona_fact_count,
                    "topic_clarity": 0.9,
                },
            }
        )

    chosen = choose_styles_with_detailed_cap(objs, detailed_cap=0.2)
    detailed = [cid for cid, style in chosen.items() if style == "highly_detailed"]
    assert len(detailed) == 2

    downgraded = [cid for cid, style in chosen.items() if style == "topic_specific"]
    assert len(downgraded) == 2

