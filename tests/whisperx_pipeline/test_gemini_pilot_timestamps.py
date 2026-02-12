from __future__ import annotations

import json

from gemini_pilot.runner import run_cli
from gemini_pilot.timestamps import parse_timestamp_response, timestamp_to_seconds


def test_parse_valid_mmss_multiple_segments() -> None:
    text = "\n".join(
        [
            "CALL_SEGMENT 00:12 --> 00:45",
            "CALL_SEGMENT 01:10 --> 02:00",
        ]
    )
    result = parse_timestamp_response(text)
    assert result.ambiguous is False
    assert result.warnings == []
    assert [(s.start_s, s.end_s) for s in result.segments] == [(12, 45), (70, 120)]


def test_parse_valid_hhmmss_segment() -> None:
    text = "CALL_SEGMENT 00:01:00 --> 00:02:30"
    result = parse_timestamp_response(text)
    assert result.ambiguous is False
    assert len(result.segments) == 1
    assert result.segments[0].start_s == 60
    assert result.segments[0].end_s == 150


def test_parse_rejects_reversed_and_invalid_timestamps() -> None:
    text = "\n".join(
        [
            "CALL_SEGMENT 00:30 --> 00:10",
            "CALL_SEGMENT 00:99 --> 01:02",
        ]
    )
    result = parse_timestamp_response(text)
    assert result.ambiguous is True
    assert result.segments == []
    assert len(result.warnings) >= 2


def test_parse_malformed_lines_are_ambiguous() -> None:
    text = "\n".join(
        [
            "Here are the segments you asked for:",
            "CALL_SEGMENT 00:10 --> 00:20",
        ]
    )
    result = parse_timestamp_response(text)
    assert result.ambiguous is True
    assert result.segments == []
    assert any("unrecognized format" in w for w in result.warnings)


def test_overlapping_segments_are_dropped() -> None:
    text = "\n".join(
        [
            "CALL_SEGMENT 00:10 --> 00:30",
            "CALL_SEGMENT 00:25 --> 00:40",
        ]
    )
    result = parse_timestamp_response(text)
    assert result.ambiguous is True
    assert result.segments == []
    assert any("overlaps previous segment" in w for w in result.warnings)


def test_timestamp_to_seconds_rejects_overflow() -> None:
    try:
        timestamp_to_seconds("10:00:00")
    except ValueError as exc:
        assert "max supported audio duration" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_cli_dry_run_with_local_and_s3_inputs(tmp_path) -> None:
    local_mp3 = tmp_path / "sample.mp3"
    local_mp3.write_bytes(b"fake-audio")
    out_dir = tmp_path / "out"
    rc = run_cli(
        [
            "--audio-path",
            str(local_mp3),
            "--audio-s3-uri",
            "s3://example-bucket/path/to/input.mp3",
            "--out-dir",
            str(out_dir),
            "--dry-run",
        ]
    )
    assert rc == 0
    manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((out_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert manifest["dry_run"] is True
    assert summary["dry_run"] is True
    assert summary["planned_requests"] == 4  # 2 inputs x 2 default models


def test_cli_missing_api_key_returns_clean_error(tmp_path, monkeypatch, caplog) -> None:
    local_mp3 = tmp_path / "sample.mp3"
    local_mp3.write_bytes(b"fake-audio")

    # Avoid loading repo .env values while testing missing-key behavior.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    caplog.set_level("ERROR")
    rc = run_cli(["--audio-path", str(local_mp3), "--out-dir", str(tmp_path / "out")])
    assert rc == 2
    assert "Missing Gemini API key" in caplog.text
