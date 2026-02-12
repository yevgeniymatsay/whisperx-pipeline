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

CALL_TIMESTAMP_PROMPT_V2 = """\
You are a Conversation Analyst specializing in Telephony Segmentation.
Your ONLY goal is to timestamp every phone call in this audio, from the
START of each call to the END of each call.

All audio is in English.

CORE DEFINITION:
A "phone call segment" is a verbal exchange over the phone between the
caller (host/salesperson/agent) and the person on the other end of the line
(lead/customer/person receiving the call).

START TIME — mark the first verbal utterance of the phone call:
  ✓ INCLUDE: First greeting ("Hello?", "Hi", etc.)
  ✗ EXCLUDE: Phone ringing, dial tones, digital connecting sounds.
  ✗ EXCLUDE: Host talking to camera BEFORE the call connects ("Let's dial…").

END TIME — mark the final verbal utterance of the phone call:
  ✓ INCLUDE: Final farewell ("Bye", "Thank you", "Not interested", etc.)
  ✗ EXCLUDE: Hangup click sound.
  ✗ EXCLUDE: Post-call commentary, music, or host talking to camera.

WHAT TO CAPTURE:
- Every phone call regardless of outcome: successful sales, wrong numbers,
  hostile rejections, short hang-ups, voicemail messages.
- When in doubt, include the segment — it is easier to filter downstream
  than to recover missed calls.

WHAT TO SKIP (not part of any call):
- Host narration or monologue to camera (not on a phone call).
- Intro/outro sequences, ads, background music with no call.
- Pure silence or dead air with no conversation.
"""
