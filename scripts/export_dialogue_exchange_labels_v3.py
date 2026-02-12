#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.call_extractor_wavlm.dialogue_exchange import Turn, dialogue_exchange_intervals  # noqa: E402
from pipeline.call_extractor_wavlm.io import get_s3  # noqa: E402
from pipeline.call_extractor_wavlm.production_metrics import merge_intervals  # noqa: E402
from pipeline.config import AWS_REGION, S3_BUCKET  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iso_utc_now() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).isoformat()


def _load_qa_mapping(
    qa_report_json: Path,
    *,
    embedder: str,
) -> dict[str, dict[str, Any]]:
    obj = _read_json(qa_report_json)
    out: dict[str, dict[str, Any]] = {}
    for clip in obj.get("clips") or []:
        clip_id = str(clip.get("clip_id") or "")
        if not clip_id:
            continue
        per_model = (clip.get("per_model") or {}).get(embedder) or {}
        out[clip_id] = per_model
    return out


def _load_diarized_turns(
    diarized_json_path: Path,
    *,
    speaker_mapping: dict[str, str],
) -> list[Turn]:
    data = _read_json(diarized_json_path)
    turns: list[Turn] = []
    for seg in data.get("segments") or []:
        raw_spk = str(seg.get("speaker") or "").strip()
        if not raw_spk:
            continue
        spk = str(speaker_mapping.get(raw_spk, "S0"))
        if spk not in ("S0", "S1"):
            spk = "S0"
        try:
            start = float(seg.get("start"))
            end = float(seg.get("end"))
        except Exception:
            continue
        if end <= start:
            continue
        turns.append(Turn(start_s=float(start), end_s=float(end), speaker=spk))
    turns.sort(key=lambda t: (t.start_s, t.end_s, t.speaker))
    return turns


def main() -> int:
    parser = argparse.ArgumentParser(description="Export v3 dialogue-exchange labels from diarized call clips.")
    parser.add_argument("--pilot-dir", type=Path, required=True, help="Artifacts dir with metadata.jsonl and diarized/*.json")
    parser.add_argument(
        "--qa-report-json",
        type=Path,
        default=None,
        help="QA report JSON with speaker canonicalization mappings (default: reports/diarization_pilot/<pilot_name>/qa_report.json)",
    )
    parser.add_argument("--embedder", type=str, default="ecapa", help="Which embedder mapping to use from qa_report.json (default: ecapa)")
    parser.add_argument("--tiny-speaker-s", type=float, default=3.0, help="Treat as K=1 if min canonical speaker total < this many seconds")
    parser.add_argument("--dt-s", type=float, default=0.10, help="Time step for exchange mask (default: 0.10)")
    parser.add_argument("--gap-pctl", type=float, default=90.0, help="Percentile for T_bridge (default: 90)")
    parser.add_argument("--t-bridge-min-s", type=float, default=2.0, help="Minimum T_bridge (default: 2)")
    parser.add_argument("--t-bridge-max-s", type=float, default=15.0, help="Maximum T_bridge (default: 15)")
    parser.add_argument(
        "--video-ids-file",
        type=Path,
        default=None,
        help="Optional file listing all video IDs to emit labels for (missing ones become empty labels).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Local output dir (default: <pilot-dir>/v3_dialogue_exchange_labels/)",
    )
    parser.add_argument(
        "--upload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upload label JSONs to S3 (default: false)",
    )
    parser.add_argument(
        "--s3-prefix-out",
        type=str,
        default="labeling/corrected_boundaries/v3_dialogue_exchange/",
        help="S3 prefix to write labels (default: labeling/corrected_boundaries/v3_dialogue_exchange/)",
    )
    args = parser.parse_args()

    pilot_dir = Path(args.pilot_dir)
    meta_path = pilot_dir / "metadata.jsonl"
    diarized_dir = pilot_dir / "diarized"
    if not meta_path.is_file():
        raise SystemExit(f"Missing metadata.jsonl: {meta_path}")
    if not diarized_dir.is_dir():
        raise SystemExit(f"Missing diarized/: {diarized_dir}")

    pilot_name = pilot_dir.name
    qa_path = Path(args.qa_report_json) if args.qa_report_json else (Path("reports") / "diarization_pilot" / pilot_name / "qa_report.json")
    if not qa_path.is_file():
        raise SystemExit(f"Missing QA report JSON: {qa_path} (run diarization_pilot_canonicalize_and_report.py first)")

    out_dir = Path(args.out_dir) if args.out_dir else (pilot_dir / "v3_dialogue_exchange_labels")
    out_dir.mkdir(parents=True, exist_ok=True)

    qa_per_clip = _load_qa_mapping(qa_path, embedder=str(args.embedder))
    rows = _read_jsonl(meta_path)
    if not rows:
        raise SystemExit(f"No rows in metadata: {meta_path}")

    # Optional: emit empty labels for all video IDs in a split file.
    all_video_ids: Optional[list[str]] = None
    if args.video_ids_file is not None:
        raw = Path(args.video_ids_file).read_text(encoding="utf-8").splitlines()
        all_video_ids = [ln.strip() for ln in raw if ln.strip() and not ln.strip().startswith("#")]

    by_vid: dict[str, list[tuple[float, float]]] = {}
    per_clip_debug: list[dict[str, Any]] = []
    missing_diarized: list[str] = []
    skipped_k1: list[str] = []

    for row in rows:
        clip_id = str(row["clip_id"])
        video_id = str(row["video_id"])
        offset_s = float(row["offset_s"])
        clip_duration_s = float(row["clip_duration_s"])

        per = qa_per_clip.get(clip_id) or {}
        mapping = per.get("canonical_mapping") or {}
        k = int(per.get("canonical_speaker_count") or 0)
        totals = per.get("canonical_speaker_total_s") or {}
        min_total = None
        if isinstance(totals, dict) and totals:
            try:
                min_total = float(min(float(v) for v in totals.values()))
            except Exception:
                min_total = None
        tiny_second = bool(per.get("tiny_second_speaker"))

        # Allow K=1, and also collapse "tiny second speaker" clips to K=1 (no exchange).
        if k < 2 or tiny_second or (min_total is not None and min_total < float(args.tiny_speaker_s)):
            skipped_k1.append(clip_id)
            continue

        diarized_path = diarized_dir / f"{clip_id}.diarized_json.json"
        if not diarized_path.is_file():
            missing_diarized.append(clip_id)
            continue

        turns = _load_diarized_turns(diarized_path, speaker_mapping=dict(mapping))
        t_bridge_s, rel_intervals = dialogue_exchange_intervals(
            turns,
            clip_duration_s=float(clip_duration_s),
            dt_s=float(args.dt_s),
            gap_pctl=float(args.gap_pctl),
            t_bridge_min_s=float(args.t_bridge_min_s),
            t_bridge_max_s=float(args.t_bridge_max_s),
        )
        abs_intervals = [(offset_s + a, offset_s + b) for a, b in rel_intervals]
        by_vid.setdefault(video_id, []).extend(abs_intervals)

        per_clip_debug.append({
            "clip_id": clip_id,
            "video_id": video_id,
            "offset_s": float(offset_s),
            "clip_duration_s": float(clip_duration_s),
            "t_bridge_s": float(t_bridge_s),
            "interval_count": int(len(rel_intervals)),
        })

    # Merge per-video intervals and write label files.
    written = 0
    updated_at = _iso_utc_now()
    videos = sorted(set(all_video_ids or list(by_vid.keys())))
    if all_video_ids is not None:
        videos = list(dict.fromkeys(all_video_ids))  # de-dup preserve order

    s3 = get_s3(region=AWS_REGION) if bool(args.upload) else None
    for vid in videos:
        intervals = by_vid.get(vid) or []
        merged = merge_intervals(intervals, join_tolerance_s=0.0)
        payload = {
            "video_id": vid,
            "updated_at": updated_at,
            "boundaries": [{"start_s": float(a), "end_s": float(b)} for a, b in merged],
        }
        out_path = out_dir / f"{vid}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written += 1
        if s3 is not None:
            key = f"{str(args.s3_prefix_out).rstrip('/')}/{vid}.json"
            s3.put_object(Bucket=S3_BUCKET, Key=key, Body=(json.dumps(payload) + "\n").encode("utf-8"))

    debug_path = out_dir / "export_debug.json"
    debug_obj = {
        "pilot_dir": str(pilot_dir),
        "qa_report_json": str(qa_path),
        "embedder": str(args.embedder),
        "tiny_speaker_s": float(args.tiny_speaker_s),
        "dt_s": float(args.dt_s),
        "gap_pctl": float(args.gap_pctl),
        "t_bridge_min_s": float(args.t_bridge_min_s),
        "t_bridge_max_s": float(args.t_bridge_max_s),
        "s3_bucket": S3_BUCKET,
        "s3_prefix_out": str(args.s3_prefix_out),
        "videos_written": int(written),
        "missing_diarized_clips": missing_diarized,
        "skipped_k1_clips": skipped_k1,
        "per_clip": per_clip_debug[:50],  # keep it small
    }
    debug_path.write_text(json.dumps(debug_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    logger.info(f"Wrote {written} label files -> {out_dir}")
    if args.upload:
        logger.info(f"Uploaded to s3://{S3_BUCKET}/{str(args.s3_prefix_out).rstrip('/')}/")
    if missing_diarized:
        logger.warning(f"Missing diarized JSON for {len(missing_diarized)} clips (see export_debug.json)")
    if skipped_k1:
        logger.info(f"Skipped {len(skipped_k1)} clips due to K=1 or tiny second speaker")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

