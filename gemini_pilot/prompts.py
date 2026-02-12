from __future__ import annotations

CALL_TIMESTAMP_PROMPT_V1 = """Analyze this audio and identify only real phone-call conversation segments.

A real call segment means active back-and-forth call conversation. Exclude:
- intros/outros, ads, music, noise-only sections, silence, hold music, voicemail prompts,
- monologues or narration that are not actual call conversation.

Output rules (strict):
1) If no real call segments exist, return exactly:
NO_CALL_SEGMENTS

2) Otherwise return one segment per line in this exact format:
CALL_SEGMENT HH:MM:SS --> HH:MM:SS

3) Times must be ascending and non-overlapping.
4) Return only those lines. No explanations, bullets, JSON, or extra text.
"""

