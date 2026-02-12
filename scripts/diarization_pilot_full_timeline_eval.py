#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Interval:
    start_s: float
    end_s: float

    def clamp(self, lo: float, hi: float) -> "Interval":
        start = float(max(lo, min(self.start_s, hi)))
        end = float(max(lo, min(self.end_s, hi)))
        if end < start:
            end = start
        return Interval(start_s=start, end_s=end)

    @property
    def dur_s(self) -> float:
        return float(max(0.0, self.end_s - self.start_s))


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


def _merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    items = [iv for iv in intervals if iv.dur_s > 0.0]
    if not items:
        return []
    items.sort(key=lambda x: (x.start_s, x.end_s))
    out: list[Interval] = [items[0]]
    for iv in items[1:]:
        last = out[-1]
        if iv.start_s <= last.end_s:
            out[-1] = Interval(start_s=last.start_s, end_s=max(last.end_s, iv.end_s))
        else:
            out.append(iv)
    return out


def _overlap_seconds(a: list[Interval], b: list[Interval]) -> float:
    """Compute total intersection duration for two unioned interval lists."""
    if not a or not b:
        return 0.0
    i = 0
    j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        ia = a[i]
        ib = b[j]
        lo = max(ia.start_s, ib.start_s)
        hi = min(ia.end_s, ib.end_s)
        if hi > lo:
            total += float(hi - lo)
        if ia.end_s <= ib.end_s:
            i += 1
        else:
            j += 1
    return float(total)


def _union_seconds(intervals: list[Interval]) -> float:
    return float(sum(iv.dur_s for iv in intervals))


def _percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, float(q)))


def _safe_div(num: float, den: float) -> float:
    if den <= 0.0:
        return 0.0
    return float(num) / float(den)


def _speech_bounds_from_diarized_json(data: dict[str, Any]) -> Optional[Tuple[float, float, float, int]]:
    """Return (speech_start_s, speech_end_s, speech_union_s, unique_speakers)."""
    segs = data.get("segments") or []
    starts: list[float] = []
    ends: list[float] = []
    speakers: set[str] = set()
    speech_intervals: list[Interval] = []
    for seg in segs:
        try:
            s = float(seg.get("start"))
            e = float(seg.get("end"))
        except Exception:
            continue
        if e <= s:
            continue
        starts.append(s)
        ends.append(e)
        spk = seg.get("speaker")
        if isinstance(spk, str) and spk:
            speakers.add(spk)
        speech_intervals.append(Interval(start_s=s, end_s=e))
    if not starts or not ends:
        return None
    speech_start_s = float(min(starts))
    speech_end_s = float(max(ends))
    speech_union = _merge_intervals(speech_intervals)
    speech_union_s = _union_seconds(speech_union)
    return speech_start_s, speech_end_s, float(speech_union_s), int(len(speakers))


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 5: map clip diarization back to full timeline and score coverage/recall.")
    parser.add_argument("--pilot-dir", type=Path, required=True, help="Pilot dir (artifacts/diarization_pilot/pilot_...)")
    parser.add_argument(
        "--qa-report-json",
        type=Path,
        default=None,
        help="Optional QA report JSON (default: reports/diarization_pilot/<pilot_id>/qa_report.json if present)",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.8,
        help="Coverage threshold for coverage>=X rate (default: 0.8)",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Output JSON path (default: reports/diarization_pilot/<pilot_id>/full_timeline_eval.json)",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Output Markdown path (default: reports/diarization_pilot/<pilot_id>/full_timeline_eval.md)",
    )
    args = parser.parse_args()

    pilot_dir = Path(args.pilot_dir)
    pilot_id = pilot_dir.name
    meta_path = pilot_dir / "metadata.jsonl"
    diarized_dir = pilot_dir / "diarized"
    if not meta_path.is_file():
        raise SystemExit(f"Missing metadata.jsonl: {meta_path}")
    if not diarized_dir.is_dir():
        raise SystemExit(f"Missing diarized/: {diarized_dir}")

    report_dir = Path("reports") / "diarization_pilot" / pilot_id
    out_json = Path(args.out_json) if args.out_json else report_dir / "full_timeline_eval.json"
    out_md = Path(args.out_md) if args.out_md else report_dir / "full_timeline_eval.md"

    qa_path = Path(args.qa_report_json) if args.qa_report_json else (report_dir / "qa_report.json")
    qa_obj: Optional[dict[str, Any]] = None
    if qa_path.is_file():
        try:
            qa_obj = _read_json(qa_path)
        except Exception:
            qa_obj = None

    rows = _read_jsonl(meta_path)
    if not rows:
        raise SystemExit(f"No rows in metadata: {meta_path}")

    per_clip: list[dict[str, Any]] = []
    missing_diarized: list[str] = []
    no_segments: list[str] = []
    coverages: list[float] = []
    coverages_dialogue: dict[str, list[float]] = {}
    coverages_k1: dict[str, list[float]] = {}

    # Used for union recall/purity in full timeline (per video).
    gt_by_vid: dict[str, list[Interval]] = {}
    pred_by_vid: dict[str, list[Interval]] = {}

    # Optional per-clip QA lookup (embedder-specific flags).
    qa_per_clip: dict[str, dict[str, Any]] = {}
    qa_embedders: list[str] = []
    if qa_obj:
        qa_embedders = list((qa_obj.get("summary") or {}).get("embedders") or [])
        for emb in qa_embedders:
            coverages_dialogue[emb] = []
            coverages_k1[emb] = []
        for clip in qa_obj.get("clips") or []:
            clip_id = str(clip.get("clip_id") or "")
            if clip_id:
                qa_per_clip[clip_id] = clip.get("per_model") or {}

    for row in rows:
        clip_id = str(row["clip_id"])
        video_id = str(row["video_id"])
        call_index = int(row.get("call_index", -1))
        offset_s = float(row["offset_s"])
        clip_duration_s = float(row["clip_duration_s"])
        gt = Interval(start_s=offset_s, end_s=offset_s + clip_duration_s)

        diarized_path = diarized_dir / f"{clip_id}.diarized_json.json"
        if not diarized_path.is_file():
            missing_diarized.append(clip_id)
            continue
        data = _read_json(diarized_path)
        bounds = _speech_bounds_from_diarized_json(data)
        if bounds is None:
            no_segments.append(clip_id)
            continue
        speech_start_s, speech_end_s, speech_union_s, unique_speakers = bounds

        # Clamp to clip duration to avoid weird rounding drift.
        speech_iv_rel = Interval(start_s=speech_start_s, end_s=speech_end_s).clamp(0.0, float(clip_duration_s))
        pred = Interval(start_s=offset_s + speech_iv_rel.start_s, end_s=offset_s + speech_iv_rel.end_s)

        # Coverage vs manual call interval (GT).
        intersection = Interval(start_s=max(gt.start_s, pred.start_s), end_s=min(gt.end_s, pred.end_s))
        coverage = _safe_div(intersection.dur_s, gt.dur_s)
        coverages.append(float(coverage))

        gt_by_vid.setdefault(video_id, []).append(gt)
        pred_by_vid.setdefault(video_id, []).append(pred)

        # Optional embedder-specific subsets.
        if qa_obj and clip_id in qa_per_clip:
            per_model = qa_per_clip[clip_id]
            for emb in qa_embedders:
                res = per_model.get(emb) or {}
                k = int(res.get("canonical_speaker_count") or 0)
                tiny = bool(res.get("tiny_second_speaker"))
                if k == 1:
                    coverages_k1[emb].append(float(coverage))
                if k == 2 and not tiny:
                    coverages_dialogue[emb].append(float(coverage))

        per_clip.append(
            {
                "clip_id": clip_id,
                "video_id": video_id,
                "call_index": call_index,
                "gt_start_s": float(gt.start_s),
                "gt_end_s": float(gt.end_s),
                "pred_start_s": float(pred.start_s),
                "pred_end_s": float(pred.end_s),
                "trim_start_s": float(speech_iv_rel.start_s),
                "trim_end_s": float(max(0.0, clip_duration_s - speech_iv_rel.end_s)),
                "coverage": float(coverage),
                "speech_union_fraction": _safe_div(float(speech_union_s), float(max(1e-9, clip_duration_s))),
                "unique_speakers_raw": int(unique_speakers),
            }
        )

    # Per-video union recall/purity.
    per_video: list[dict[str, Any]] = []
    for vid in sorted(gt_by_vid.keys() | pred_by_vid.keys()):
        gt_u = _merge_intervals(gt_by_vid.get(vid, []))
        pred_u = _merge_intervals(pred_by_vid.get(vid, []))
        overlap_s = _overlap_seconds(gt_u, pred_u)
        gt_s = _union_seconds(gt_u)
        pred_s = _union_seconds(pred_u)
        per_video.append(
            {
                "video_id": vid,
                "gt_union_s": float(gt_s),
                "pred_union_s": float(pred_s),
                "overlap_s": float(overlap_s),
                "union_recall": _safe_div(overlap_s, gt_s),
                "union_purity": _safe_div(overlap_s, pred_s),
                "calls": int(len(gt_by_vid.get(vid, []))),
            }
        )

    gt_all_u = _merge_intervals([iv for xs in gt_by_vid.values() for iv in xs])
    pred_all_u = _merge_intervals([iv for xs in pred_by_vid.values() for iv in xs])
    overlap_all_s = _overlap_seconds(gt_all_u, pred_all_u)
    gt_all_s = _union_seconds(gt_all_u)
    pred_all_s = _union_seconds(pred_all_u)

    min_cov = float(args.min_coverage)
    coverage_ge = int(sum(1 for c in coverages if float(c) >= min_cov))
    coverage_rate = _safe_div(float(coverage_ge), float(len(coverages) or 1))

    summary: dict[str, Any] = {
        "pilot_dir": str(pilot_dir),
        "pilot_id": pilot_id,
        "clip_count_total": int(len(rows)),
        "clip_count_scored": int(len(coverages)),
        "missing_diarized_count": int(len(missing_diarized)),
        "no_segments_count": int(len(no_segments)),
        "min_coverage": float(min_cov),
        "coverage_ge_min_count": int(coverage_ge),
        "coverage_ge_min_rate": float(coverage_rate),
        "coverage_p10": _percentile(coverages, 10),
        "coverage_p50": _percentile(coverages, 50),
        "coverage_p90": _percentile(coverages, 90),
        "union_recall_all": _safe_div(overlap_all_s, gt_all_s),
        "union_purity_all": _safe_div(overlap_all_s, pred_all_s),
        "gt_union_s_all": float(gt_all_s),
        "pred_union_s_all": float(pred_all_s),
        "overlap_s_all": float(overlap_all_s),
    }

    embedder_summary: dict[str, Any] = {}
    if qa_embedders:
        for emb in qa_embedders:
            vals_d = coverages_dialogue.get(emb, [])
            vals_k1 = coverages_k1.get(emb, [])
            embedder_summary[emb] = {
                "dialogue_count": int(len(vals_d)),
                "dialogue_coverage_ge_min_rate": _safe_div(float(sum(1 for c in vals_d if c >= min_cov)), float(len(vals_d) or 1)),
                "dialogue_coverage_p50": _percentile(vals_d, 50),
                "k1_count": int(len(vals_k1)),
                "k1_coverage_p50": _percentile(vals_k1, 50),
            }

    out_obj = {
        "summary": summary,
        "embedder_summary": embedder_summary,
        "missing_diarized": missing_diarized,
        "no_segments": no_segments,
        "per_video": per_video,
        "per_clip": sorted(per_clip, key=lambda r: (r["video_id"], r["call_index"], r["gt_start_s"])),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Write a compact Markdown report.
    worst = sorted(per_clip, key=lambda r: float(r["coverage"]))[:10]
    lines: list[str] = []
    lines.append(f"# Diarization Pilot Full-Timeline Eval: {pilot_id}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Pilot dir: `{pilot_dir}`")
    lines.append(f"- Clips scored: {summary['clip_count_scored']}/{summary['clip_count_total']}")
    lines.append(f"- Missing diarized outputs: {summary['missing_diarized_count']}")
    lines.append(f"- No segments (empty diarization): {summary['no_segments_count']}")
    lines.append(f"- Coverage>= {min_cov:.2f}: {summary['coverage_ge_min_count']}/{summary['clip_count_scored']} ({summary['coverage_ge_min_rate']:.3f})")
    lines.append(
        f"- Coverage p10/p50/p90: {summary['coverage_p10']:.3f} / {summary['coverage_p50']:.3f} / {summary['coverage_p90']:.3f}"
    )
    lines.append(f"- Union recall (all): {summary['union_recall_all']:.3f}")
    lines.append(f"- Union purity (all): {summary['union_purity_all']:.3f}")

    if qa_embedders:
        lines.append("")
        lines.append("## Coverage By Subset (From QA Report)")
        lines.append("")
        for emb in qa_embedders:
            e = embedder_summary.get(emb, {})
            lines.append(
                f"- `{emb}`: dialogue_count={e.get('dialogue_count')} "
                f"dialogue_cov>=min={float(e.get('dialogue_coverage_ge_min_rate') or 0.0):.3f} "
                f"dialogue_cov_p50={e.get('dialogue_coverage_p50')} "
                f"k1_count={e.get('k1_count')} "
                f"k1_cov_p50={e.get('k1_coverage_p50')}"
            )

    lines.append("")
    lines.append("## Worst Coverage Clips (Top 10)")
    lines.append("")
    lines.append("| clip_id | video_id | coverage | trim_start_s | trim_end_s | unique_speakers_raw |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in worst:
        lines.append(
            f"| {r['clip_id']} | {r['video_id']} | {r['coverage']:.3f} | {r['trim_start_s']:.2f} |"
            f" {r['trim_end_s']:.2f} | {r['unique_speakers_raw']} |"
        )

    if missing_diarized:
        lines.append("")
        lines.append("## Missing Diarized Outputs")
        lines.append("")
        for cid in missing_diarized:
            lines.append(f"- {cid}")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Wrote {out_json}")
    logger.info(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
