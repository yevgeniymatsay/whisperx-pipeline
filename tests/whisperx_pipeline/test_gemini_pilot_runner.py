from __future__ import annotations

import json

import pytest

from gemini_pilot.runner import run_cli


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


def test_cli_help_does_not_include_parse_toggle(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        run_cli(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--parse-timestamps" not in out


def test_cli_missing_api_key_returns_clean_error(tmp_path, monkeypatch, caplog) -> None:
    local_mp3 = tmp_path / "sample.mp3"
    local_mp3.write_bytes(b"fake-audio")

    # Avoid loading repo .env values while testing missing-key behavior.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_PILOT_DISABLE_DOTENV", "1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    caplog.set_level("ERROR")
    rc = run_cli(["--audio-path", str(local_mp3), "--out-dir", str(tmp_path / "out")])
    assert rc == 2
    assert "Missing Gemini API key" in caplog.text
