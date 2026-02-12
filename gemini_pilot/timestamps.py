from __future__ import annotations

import re
from dataclasses import dataclass

_MAX_AUDIO_SECONDS = int(9.5 * 3600)
_SEGMENT_LINE_RE = re.compile(
    r"^(?:[-*]\s*|\d+\s*[).:-]\s*)?CALL_SEGMENT\s+"
    r"(?P<start>\d{1,3}:\d{2}(?::\d{2})?)\s*-->\s*(?P<end>\d{1,3}:\d{2}(?::\d{2})?)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TimestampSegment:
    start_s: int
    end_s: int
    start_ts: str
    end_ts: str
    raw_line: str


@dataclass(frozen=True)
class TimestampParseResult:
    segments: list[TimestampSegment]
    warnings: list[str]
    ambiguous: bool
    no_call_segments: bool


def timestamp_to_seconds(token: str) -> int:
    raw = token.strip()
    parts = raw.split(":")
    if len(parts) == 2:
        mm_s, ss_s = parts
        if not (mm_s.isdigit() and ss_s.isdigit()):
            raise ValueError(f"Invalid timestamp token: {token!r}")
        mm = int(mm_s)
        ss = int(ss_s)
        if not (0 <= ss < 60):
            raise ValueError(f"Invalid MM:SS seconds value: {token!r}")
        total = mm * 60 + ss
    elif len(parts) == 3:
        hh_s, mm_s, ss_s = parts
        if not (hh_s.isdigit() and mm_s.isdigit() and ss_s.isdigit()):
            raise ValueError(f"Invalid timestamp token: {token!r}")
        hh = int(hh_s)
        mm = int(mm_s)
        ss = int(ss_s)
        if not (0 <= mm < 60 and 0 <= ss < 60):
            raise ValueError(f"Invalid HH:MM:SS minute/second value: {token!r}")
        total = hh * 3600 + mm * 60 + ss
    else:
        raise ValueError(f"Unsupported timestamp format: {token!r}")

    if total > _MAX_AUDIO_SECONDS:
        raise ValueError(f"Timestamp exceeds max supported audio duration (~9.5h): {token!r}")
    return total


def parse_timestamp_response(text: str) -> TimestampParseResult:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    warnings: list[str] = []
    parsed_segments: list[TimestampSegment] = []
    no_call_markers = 0

    if not lines:
        warnings.append("Response was empty.")
        return TimestampParseResult(segments=[], warnings=warnings, ambiguous=True, no_call_segments=False)

    for idx, line in enumerate(lines, start=1):
        if line.upper() == "NO_CALL_SEGMENTS":
            no_call_markers += 1
            continue

        m = _SEGMENT_LINE_RE.match(line)
        if not m:
            warnings.append(f"line {idx}: unrecognized format: {line}")
            continue
        start_ts = m.group("start")
        end_ts = m.group("end")
        try:
            start_s = timestamp_to_seconds(start_ts)
            end_s = timestamp_to_seconds(end_ts)
        except ValueError as exc:
            warnings.append(f"line {idx}: {exc}")
            continue
        if end_s <= start_s:
            warnings.append(f"line {idx}: end timestamp must be greater than start timestamp.")
            continue

        parsed_segments.append(
            TimestampSegment(
                start_s=start_s,
                end_s=end_s,
                start_ts=start_ts,
                end_ts=end_ts,
                raw_line=line,
            )
        )

    if no_call_markers > 1:
        warnings.append("NO_CALL_SEGMENTS was repeated.")
    if no_call_markers > 0 and parsed_segments:
        warnings.append("NO_CALL_SEGMENTS was mixed with CALL_SEGMENT lines.")

    sorted_segments = sorted(parsed_segments, key=lambda s: (s.start_s, s.end_s))
    if sorted_segments != parsed_segments and parsed_segments:
        warnings.append("Segments were not returned in ascending order.")

    seen_ranges: set[tuple[int, int]] = set()
    prev_end: int | None = None
    for i, seg in enumerate(sorted_segments, start=1):
        key = (seg.start_s, seg.end_s)
        if key in seen_ranges:
            warnings.append(f"segment {i}: duplicate segment range {seg.start_ts} --> {seg.end_ts}.")
        seen_ranges.add(key)
        if prev_end is not None and seg.start_s < prev_end:
            warnings.append(f"segment {i}: overlaps previous segment.")
        prev_end = max(prev_end or 0, seg.end_s)

    if not sorted_segments and no_call_markers == 0 and not warnings:
        warnings.append("No parseable CALL_SEGMENT lines found.")

    ambiguous = bool(warnings)
    segments = [] if ambiguous else sorted_segments
    return TimestampParseResult(
        segments=segments,
        warnings=warnings,
        ambiguous=ambiguous,
        no_call_segments=(no_call_markers > 0 and not sorted_segments),
    )

