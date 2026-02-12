from __future__ import annotations

CALL_TIMESTAMP_PROMPT_V1 = """Analyze this audio and identify only real phone-call conversation segments.

A real call segment means active back-and-forth call conversation. Exclude:
- intros/outros, ads, music, noise-only sections, silence, hold music, voicemail prompts,
- monologues or narration that are not actual call conversation.

Output rules (strict):
1) Return JSON that matches the provided response schema exactly (no extra keys, no extra text).
2) If no real call segments exist, return {"segments": []}.
3) Otherwise, return segments as a list of {start, end}.
4) Timestamps must be in HH:MM:SS (zero-padded).
5) Segments must be ascending and non-overlapping.
6) Prefer fewer segments: only split when there is a clear stretch of excluded content between call parts.
"""
