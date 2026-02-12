"""Response schema for Gemini call-timestamp extraction (V2).

Uses native ``google.genai.types.Schema`` so the API enforces structure
server-side — no post-hoc JSON validation needed.
"""
from __future__ import annotations

from google.genai import types

SCHEMA_VERSION = "v2"

CALL_SEGMENT_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "segments": types.Schema(
            type=types.Type.ARRAY,
            description="List of phone call segments found in the audio.",
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "start": types.Schema(
                        type=types.Type.STRING,
                        description="Start timestamp (MM:SS) of the first word spoken in the call.",
                    ),
                    "end": types.Schema(
                        type=types.Type.STRING,
                        description="End timestamp (MM:SS) of the last word spoken in the call.",
                    ),
                    "confidence": types.Schema(
                        type=types.Type.STRING,
                        enum=["High", "Medium", "Low"],
                        description="Confidence that this is a real phone conversation.",
                    ),
                },
                required=["start", "end", "confidence"],
            ),
        )
    },
    required=["segments"],
)
