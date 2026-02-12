from __future__ import annotations

import json
import runpy
from pathlib import Path


def _load_eval_module() -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "gemini_pilot" / "scripts" / "eval_gemini_call_detector.py"
    return runpy.run_path(str(path))


def test_extract_video_id_from_source_id() -> None:
    mod = _load_eval_module()
    extract = mod["_extract_video_id"]

    assert extract("local_DEt3IRqqUVs_deadbeef") == "DEt3IRqqUVs"
    assert extract("s3_DEt3IRqqUVs_deadbeef") == "DEt3IRqqUVs"

    # Sanitized filename stems replace " - " with "_-_", so video_id lives after the last "_-_".
    assert extract("local_10_Live_Cold_Calls_-_DEt3IRqqUVs_deadbeef") == "DEt3IRqqUVs"

    assert extract("local_bad") is None


def test_load_predictions_reads_response_text_from_result_json(tmp_path) -> None:
    mod = _load_eval_module()
    load_predictions = mod["load_predictions"]

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    source_id = "local_10_Live_Cold_Calls_-_DEt3IRqqUVs_deadbeef"
    model = "gemini-2.5-flash"

    result_json_path = run_dir / "outputs" / source_id / f"{model}.result.json"

    # Minimal run_summary.json pointing at one ok result.
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "status": "ok",
                        "source_id": source_id,
                        "model": model,
                        "result_json": str(result_json_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result_json_path.parent.mkdir(parents=True, exist_ok=True)
    result_payload = {
        "status": "ok",
        "source_id": source_id,
        "model": model,
        "response_text": json.dumps(
            {
                "segments": [
                    {"start": "00:10", "end": "00:20"},
                    {"start": "01:00", "end": "01:30"},
                ]
            }
        ),
    }
    result_json_path.write_text(json.dumps(result_payload), encoding="utf-8")

    preds = load_predictions(run_dir)
    assert "DEt3IRqqUVs" in preds
    assert model in preds["DEt3IRqqUVs"]
    segs = preds["DEt3IRqqUVs"][model]
    assert len(segs) == 2
    assert segs[0].start == 10
    assert segs[0].end == 20


def test_load_predictions_falls_back_to_legacy_raw_txt(tmp_path) -> None:
    mod = _load_eval_module()
    load_predictions = mod["load_predictions"]

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    source_id = "local_10_Live_Cold_Calls_-_DEt3IRqqUVs_deadbeef"
    model = "gemini-2.5-flash"

    (run_dir / "run_summary.json").write_text(
        json.dumps({"results": [{"status": "ok", "source_id": source_id, "model": model}]}),
        encoding="utf-8",
    )

    out_dir = run_dir / "outputs" / source_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{model}.raw.txt").write_text(
        json.dumps(
            {
                "segments": [
                    {"start": "00:10", "end": "00:20"},
                    {"start": "01:00", "end": "01:30"},
                ]
            }
        ),
        encoding="utf-8",
    )

    preds = load_predictions(run_dir)
    assert "DEt3IRqqUVs" in preds
    assert model in preds["DEt3IRqqUVs"]
    segs = preds["DEt3IRqqUVs"][model]
    assert len(segs) == 2
    assert segs[0].start == 10
    assert segs[0].end == 20
