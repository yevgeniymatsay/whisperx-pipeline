# tests/system_prompts/test_text_metrics.py

from pipeline.system_prompts.text_metrics import compute_topic_clarity


def test_topic_clarity_higher_for_repetitive_topic():
    turns_topic = [
        {"role": "assistant", "content": "Mortgage rates mortgage rates mortgage rates."},
        {"role": "user", "content": "Yeah mortgage rates keep changing."},
    ]
    turns_diffuse = [
        {"role": "assistant", "content": "Hey how are you doing today?"},
        {"role": "user", "content": "Good. Weather is nice. I went shopping and then cooked dinner."},
        {"role": "assistant", "content": "Cool. Also did you watch the game? I might travel next week."},
    ]

    c1 = compute_topic_clarity(turns_topic)
    c2 = compute_topic_clarity(turns_diffuse)
    assert c1 > c2


def test_topic_clarity_empty_is_zero():
    assert compute_topic_clarity([]) == 0.0

