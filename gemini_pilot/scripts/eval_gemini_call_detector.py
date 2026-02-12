#!/usr/bin/env python3
"""Evaluate Gemini call-detector predictions against human-labeled ground truth.

This script reads Gemini pilot run artifacts (`*.result.json` with
`response_text`) and produces two eval artifacts:
- gemini_pilot/eval/<run_id>_eval.json
- gemini_pilot/eval/<run_id>_eval.md

Usage:
    python gemini_pilot/scripts/eval_gemini_call_detector.py \
        --run-dir artifacts/gemini_pilot/run_XXXXXXXX \
        --gt-s3-prefix s3://rezora-whisperx-us-east-1-864981718771/labeling/corrected_boundaries/v1/ \
        --iou-threshold 0.3
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console(width=120)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class MatchResult:
    label: str  # MATCH, MISSED, EXTRA, BOUNDARY_OFF
    gt: Segment | None = None
    pred: Segment | None = None
    iou: float = 0.0
    start_offset: float = 0.0  # pred.start - gt.start (positive = pred is later)
    end_offset: float = 0.0  # pred.end - gt.end


@dataclass
class VideoEval:
    video_id: str
    model: str
    gt_segments: list[Segment]
    pred_segments: list[Segment]
    results: list[MatchResult] = field(default_factory=list)
    matches: int = 0
    missed: int = 0
    extra: int = 0
    boundary_off: int = 0


@dataclass
class ModelSummary:
    model: str
    total_gt: int = 0
    total_pred: int = 0
    total_matches: int = 0
    total_missed: int = 0
    total_extra: int = 0
    total_boundary_off: int = 0

    @property
    def precision(self) -> float:
        if self.total_matches + self.total_extra == 0:
            return 0.0
        return self.total_matches / (self.total_matches + self.total_extra)

    @property
    def recall(self) -> float:
        if self.total_matches + self.total_missed == 0:
            return 0.0
        return self.total_matches / (self.total_matches + self.total_missed)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------


def _detect_and_parse_timestamps(segments_raw: list[dict]) -> list[Segment]:
    """Auto-detect timestamp format and parse all segments.

    Formats seen from Gemini models:
      - MM:SS:00  (third field always 00 → treat first two as MM:SS)
      - HH:MM:SS  (true hours:minutes:seconds)
      - MM:SS     (two-field, minutes:seconds)
    """
    all_ts: list[str] = []
    for seg in segments_raw:
        all_ts.append(seg.get("start", ""))
        all_ts.append(seg.get("end", ""))

    third_fields: list[str] = []
    two_field_count = 0
    three_field_count = 0
    for ts in all_ts:
        parts = ts.strip().split(":")
        if len(parts) == 3:
            three_field_count += 1
            third_fields.append(parts[2])
        elif len(parts) == 2:
            two_field_count += 1

    if three_field_count > 0 and two_field_count == 0:
        all_third_zero = all(f.strip() == "00" for f in third_fields)
        fmt = "MMSS00" if all_third_zero else "HHMMSS"
    else:
        fmt = "MMSS"

    out: list[Segment] = []
    for seg in segments_raw:
        start_s = _ts_to_seconds(seg.get("start", ""), fmt)
        end_s = _ts_to_seconds(seg.get("end", ""), fmt)
        if start_s is not None and end_s is not None and end_s > start_s:
            out.append(Segment(start=start_s, end=end_s))
    return out


def _ts_to_seconds(ts: str, fmt: str) -> float | None:
    parts = ts.strip().split(":")
    try:
        nums = [int(p) for p in parts]
    except (ValueError, TypeError):
        return None

    if fmt == "MMSS00" and len(nums) == 3:
        return nums[0] * 60 + nums[1]
    if fmt == "HHMMSS" and len(nums) == 3:
        return nums[0] * 3600 + nums[1] * 60 + nums[2]
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    if len(nums) == 3:
        # Fallback: treat as HH:MM:SS
        return nums[0] * 3600 + nums[1] * 60 + nums[2]
    return None


# ---------------------------------------------------------------------------
# IoU matching
# ---------------------------------------------------------------------------


def _iou(a: Segment, b: Segment) -> float:
    overlap_start = max(a.start, b.start)
    overlap_end = min(a.end, b.end)
    overlap = max(0.0, overlap_end - overlap_start)
    union = (a.end - a.start) + (b.end - b.start) - overlap
    if union <= 0:
        return 0.0
    return overlap / union


def match_segments(
    gt: list[Segment],
    pred: list[Segment],
    iou_threshold: float,
    boundary_off_threshold: float = 10.0,
) -> list[MatchResult]:
    candidates: list[tuple[float, int, int]] = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            score = _iou(g, p)
            if score >= iou_threshold:
                candidates.append((score, gi, pi))

    candidates.sort(key=lambda x: x[0], reverse=True)

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    results: list[MatchResult] = []

    for score, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)

        g, p = gt[gi], pred[pi]
        start_off = p.start - g.start
        end_off = p.end - g.end

        is_boundary_off = abs(start_off) > boundary_off_threshold or abs(end_off) > boundary_off_threshold
        label = "BOUNDARY_OFF" if is_boundary_off else "MATCH"

        results.append(
            MatchResult(
                label=label,
                gt=g,
                pred=p,
                iou=score,
                start_offset=start_off,
                end_offset=end_off,
            )
        )

    for gi, g in enumerate(gt):
        if gi not in matched_gt:
            results.append(MatchResult(label="MISSED", gt=g))

    for pi, p in enumerate(pred):
        if pi not in matched_pred:
            results.append(MatchResult(label="EXTRA", pred=p))

    def _sort_key(r: MatchResult) -> float:
        if r.gt:
            return r.gt.start
        if r.pred:
            return r.pred.start
        return 0.0

    results.sort(key=_sort_key)
    return results


# ---------------------------------------------------------------------------
# Ground truth loading (S3)
# ---------------------------------------------------------------------------


def _ensure_trailing_slash(s3_prefix: str) -> str:
    return s3_prefix if s3_prefix.endswith("/") else (s3_prefix + "/")


def _download_gt_files(s3_prefix: str, video_ids: list[str], local_dir: Path) -> list[Path]:
    local_dir.mkdir(parents=True, exist_ok=True)
    s3_prefix = _ensure_trailing_slash(s3_prefix)
    out: list[Path] = []
    total = len(video_ids)
    for i, vid in enumerate(video_ids, 1):
        src = f"{s3_prefix}{vid}.json"
        dst = local_dir / f"{vid}.json"
        console.print(f"[bold]Downloading GT[/bold] {i}/{total}: {vid}")
        subprocess.run(["aws", "s3", "cp", src, str(dst)], check=True)
        out.append(dst)
    return out


def load_ground_truth(
    s3_prefix: str,
    *,
    video_ids: list[str],
    cache_dir: Path | None = None,
) -> dict[str, list[Segment]]:
    if cache_dir is None:
        cache_dir = Path(tempfile.mkdtemp(prefix="gemini_eval_gt_"))

    files = _download_gt_files(s3_prefix, sorted(set(video_ids)), cache_dir)
    gt: dict[str, list[Segment]] = {}

    for f in files:
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            logger.warning("Skipping malformed GT file: %s", f)
            continue

        video_id = data.get("video_id", f.stem)
        boundaries = data.get("boundaries", [])
        segments = []
        for b in boundaries:
            start = b.get("start_s")
            end = b.get("end_s")
            if start is not None and end is not None and end > start:
                segments.append(Segment(start=float(start), end=float(end)))

        gt[video_id] = sorted(segments, key=lambda s: s.start)

    return gt


# ---------------------------------------------------------------------------
# Prediction loading (run artifacts)
# ---------------------------------------------------------------------------


_VIDEO_ID_FROM_STEM = re.compile(r"^(?:local|s3)_(.+?)_[0-9a-f]{6,12}$")


def _extract_video_id(source_id: str) -> str | None:
    m = _VIDEO_ID_FROM_STEM.match(source_id)
    if not m:
        return None
    stem = m.group(1)
    if "_-_" in stem:
        return stem.rsplit("_-_", 1)[-1]
    return stem


def _recover_truncated_segments(path: Path) -> dict | None:
    text = path.read_text(errors="replace")
    pattern = re.compile(r'\{\s*"start"\s*:\s*"([^"]+)"\s*,\s*"end"\s*:\s*"([^"]+)"\s*\}')
    matches = pattern.findall(text)
    if not matches:
        return None
    segments = [{"start": m[0], "end": m[1]} for m in matches]
    return {"segments": segments}


def _resolve_result_json_path(run_dir: Path, result: dict, source_id: str, model: str) -> Path | None:
    candidates: list[Path] = []
    from_summary = result.get("result_json")
    if isinstance(from_summary, str) and from_summary:
        p = Path(from_summary)
        candidates.append(p)
        if not p.is_absolute():
            candidates.append(run_dir / p)
    candidates.append(run_dir / "outputs" / source_id / f"{model}.result.json")

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def load_predictions(run_dir: Path) -> dict[str, dict[str, list[Segment]]]:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"run_summary.json not found in {run_dir}")

    summary = json.loads(summary_path.read_text())
    preds: dict[str, dict[str, list[Segment]]] = {}

    for result in summary.get("results", []):
        if result.get("status") != "ok":
            continue

        source_id = result["source_id"]
        model = result["model"]
        video_id = _extract_video_id(source_id)
        if not video_id:
            logger.warning("Cannot extract video_id from source_id=%s", source_id)
            continue

        raw_data: dict | None = None
        result_json_path = _resolve_result_json_path(run_dir, result, source_id, model)
        if result_json_path is not None:
            try:
                result_payload = json.loads(result_json_path.read_text())
            except json.JSONDecodeError:
                logger.warning("Malformed result JSON in %s", result_json_path)
                result_payload = None
            if result_payload is not None:
                response_text = result_payload.get("response_text")
                if isinstance(response_text, str):
                    try:
                        raw_data = json.loads(response_text)
                    except json.JSONDecodeError:
                        logger.warning("Malformed response_text JSON in %s", result_json_path)
                elif "response_text" in result_payload:
                    logger.warning("Non-string response_text in %s", result_json_path)

        # Backward compatibility for older runs that only wrote .raw.txt.
        if raw_data is None:
            raw_path = run_dir / "outputs" / source_id / f"{model}.raw.txt"
            if not raw_path.exists():
                logger.warning("Missing result_json response_text and legacy raw file for %s/%s", source_id, model)
                continue

            try:
                raw_data = json.loads(raw_path.read_text())
            except json.JSONDecodeError:
                raw_data = _recover_truncated_segments(raw_path)
                if raw_data is None:
                    logger.warning("Malformed JSON (unrecoverable) in %s", raw_path)
                    continue
                logger.info(
                    "Recovered %d segments from truncated output in %s",
                    len(raw_data.get("segments", [])),
                    raw_path,
                )

        segments_raw = raw_data.get("segments", [])
        segments = _detect_and_parse_timestamps(segments_raw)

        if video_id not in preds:
            preds[video_id] = {}
        preds[video_id][model] = segments

    return preds


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _fmt_range(seg: Segment) -> str:
    return f"{_fmt_time(seg.start)}-{_fmt_time(seg.end)} ({int(seg.duration)}s)"


def _fmt_offset(val: float) -> str:
    sign = "+" if val >= 0 else ""
    return f"{sign}{int(val)}s"


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_video_eval(ve: VideoEval) -> None:
    console.print(f"\n[bold]═══ {ve.video_id} — {ve.model} ═══[/bold]")
    console.print(f"GT: {len(ve.gt_segments)} segments │ Predicted: {len(ve.pred_segments)} segments")
    console.print()

    max_extra_shown = 10
    extra_shown = 0
    extra_total = sum(1 for r in ve.results if r.label == "EXTRA")

    for r in ve.results:
        if r.label in ("MATCH", "BOUNDARY_OFF"):
            assert r.gt and r.pred
            line = f"  [green]{r.label:14s}[/green]" if r.label == "MATCH" else f"  [yellow]{r.label:14s}[/yellow]"
            line += f" GT {_fmt_range(r.gt)}  ↔  Pred {_fmt_range(r.pred)}"
            line += f"  IoU={r.iou:.2f}"
            if abs(r.start_offset) >= 1:
                line += f"  start {_fmt_offset(r.start_offset)}"
            if abs(r.end_offset) >= 1:
                line += f"  end {_fmt_offset(r.end_offset)}"
            if r.label == "BOUNDARY_OFF":
                line += "  [yellow]⚠ BOUNDARY_OFF[/yellow]"
            console.print(line)
        elif r.label == "MISSED":
            assert r.gt
            console.print(f"  [red]{'MISSED':14s}[/red] GT {_fmt_range(r.gt)}")
        elif r.label == "EXTRA":
            assert r.pred
            extra_shown += 1
            if extra_shown <= max_extra_shown:
                console.print(f"  [magenta]{'EXTRA':14s}[/magenta] {'':28s}Pred {_fmt_range(r.pred)}")
            elif extra_shown == max_extra_shown + 1:
                console.print(f"  [magenta]  ... and {extra_total - max_extra_shown} more EXTRA segments[/magenta]")


def print_summary_table(summaries: list[ModelSummary]) -> None:
    table = Table(title="Model Summary", show_lines=True)
    table.add_column("Model", style="bold", min_width=22, no_wrap=True)
    table.add_column("GT", justify="right")
    table.add_column("Pred", justify="right")
    table.add_column("Match", justify="right", style="green")
    table.add_column("Miss", justify="right", style="red")
    table.add_column("Extra", justify="right", style="magenta")
    table.add_column("BndOff", justify="right", style="yellow")
    table.add_column("Prec", justify="right")
    table.add_column("Rec", justify="right")
    table.add_column("F1", justify="right", style="bold cyan")

    for s in sorted(summaries, key=lambda x: x.f1, reverse=True):
        table.add_row(
            s.model,
            str(s.total_gt),
            str(s.total_pred),
            str(s.total_matches),
            str(s.total_missed),
            str(s.total_extra),
            str(s.total_boundary_off),
            f"{s.precision:.2f}",
            f"{s.recall:.2f}",
            f"{s.f1:.2f}",
        )

    console.print()
    console.print(table)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _build_json_report(video_evals: list[VideoEval], summaries: list[ModelSummary], args: argparse.Namespace) -> dict:
    def _seg_dict(s: Segment | None) -> dict | None:
        if s is None:
            return None
        return {"start": s.start, "end": s.end, "duration": s.duration}

    video_details = []
    for ve in video_evals:
        results_list = []
        for r in ve.results:
            results_list.append(
                {
                    "label": r.label,
                    "gt": _seg_dict(r.gt),
                    "pred": _seg_dict(r.pred),
                    "iou": round(r.iou, 4),
                    "start_offset": round(r.start_offset, 2),
                    "end_offset": round(r.end_offset, 2),
                }
            )
        video_details.append(
            {
                "video_id": ve.video_id,
                "model": ve.model,
                "gt_count": len(ve.gt_segments),
                "pred_count": len(ve.pred_segments),
                "matches": ve.matches,
                "missed": ve.missed,
                "extra": ve.extra,
                "boundary_off": ve.boundary_off,
                "results": results_list,
            }
        )

    model_summaries = []
    for s in sorted(summaries, key=lambda x: x.f1, reverse=True):
        model_summaries.append(
            {
                "model": s.model,
                "total_gt": s.total_gt,
                "total_pred": s.total_pred,
                "total_matches": s.total_matches,
                "total_missed": s.total_missed,
                "total_extra": s.total_extra,
                "total_boundary_off": s.total_boundary_off,
                "precision": round(s.precision, 4),
                "recall": round(s.recall, 4),
                "f1": round(s.f1, 4),
            }
        )

    return {
        "run_dir": str(args.run_dir),
        "gt_s3_prefix": args.gt_s3_prefix,
        "iou_threshold": args.iou_threshold,
        "boundary_off_threshold": args.boundary_off_threshold,
        "video_evals": video_details,
        "model_summaries": model_summaries,
    }


_LABEL_ICON = {
    "MATCH": "\u2705",  # ✅
    "BOUNDARY_OFF": "\u26a0\ufe0f",  # ⚠️
    "MISSED": "\u274c",  # ❌
    "EXTRA": "\U0001f47b",  # 👻
}


def _build_markdown_report(
    video_evals: list[VideoEval],
    summaries: list[ModelSummary],
    args: argparse.Namespace,
    *,
    prompt_version: str = "unknown",
    schema_version: str = "unknown",
) -> str:
    lines: list[str] = []
    w = lines.append

    run_id = args.run_dir.name
    w("# Gemini Call Detector — Eval Report")
    w("")
    w(f"**Run:** `{run_id}`  ")
    w(f"**Prompt:** `{prompt_version}` · **Schema:** `{schema_version}`  ")
    w(f"**IoU threshold:** {args.iou_threshold}  ")
    w(f"**Boundary-off threshold:** {args.boundary_off_threshold}s  ")
    w(f"**GT source:** `{args.gt_s3_prefix}`")
    w("")

    w("## Model Summary")
    w("")
    sorted_summaries = sorted(summaries, key=lambda x: x.f1, reverse=True)
    w("| Model | GT | Pred | Match | Miss | Extra | BndOff | Prec | Rec | F1 |")
    w("|:------|---:|-----:|------:|-----:|------:|-------:|-----:|----:|---:|")
    for s in sorted_summaries:
        f1_str = f"**{s.f1:.2f}**"
        w(
            f"| {s.model} | {s.total_gt} | {s.total_pred} "
            f"| {s.total_matches} | {s.total_missed} | {s.total_extra} "
            f"| {s.total_boundary_off} | {s.precision:.2f} | {s.recall:.2f} "
            f"| {f1_str} |"
        )
    w("")

    w("### Legend")
    w("")
    w("| Icon | Label | Meaning |")
    w("|:----:|:------|:--------|")
    w(f"| {_LABEL_ICON['MATCH']} | MATCH | GT segment matched by prediction (IoU ≥ {args.iou_threshold}) |")
    w(
        f"| {_LABEL_ICON['BOUNDARY_OFF']} | BOUNDARY_OFF | Matched, but start or end off by >{int(args.boundary_off_threshold)}s |"
    )
    w(f"| {_LABEL_ICON['MISSED']} | MISSED | GT segment with no matching prediction (false negative) |")
    w(f"| {_LABEL_ICON['EXTRA']} | EXTRA | Predicted segment with no matching GT (false positive) |")
    w("")

    from collections import OrderedDict

    by_video: OrderedDict[str, list[VideoEval]] = OrderedDict()
    for ve in video_evals:
        by_video.setdefault(ve.video_id, []).append(ve)

    for vid, evals in by_video.items():
        w("---")
        w(f"## {vid}")
        w("")

        for ve in evals:
            match_pct = (ve.matches / len(ve.gt_segments) * 100) if ve.gt_segments else 0
            w(f"### {ve.model}")
            w(
                f"GT: **{len(ve.gt_segments)}** segments · "
                f"Pred: **{len(ve.pred_segments)}** segments · "
                f"Matched: **{ve.matches}** ({match_pct:.0f}%) · "
                f"Missed: **{ve.missed}** · "
                f"Extra: **{ve.extra}**"
            )
            w("")

            w("| | GT | Pred | IoU | Offset |")
            w("|:--:|:---|:-----|:---:|:-------|")

            extra_count = 0
            max_extra_rows = 10
            for r in ve.results:
                icon = _LABEL_ICON.get(r.label, "")
                if r.label in ("MATCH", "BOUNDARY_OFF"):
                    assert r.gt and r.pred
                    gt_str = f"`{_fmt_range(r.gt)}`"
                    pred_str = f"`{_fmt_range(r.pred)}`"
                    iou_str = f"{r.iou:.2f}"

                    offsets = []
                    if abs(r.start_offset) >= 1:
                        offsets.append(f"start {_fmt_offset(r.start_offset)}")
                    if abs(r.end_offset) >= 1:
                        offsets.append(f"end {_fmt_offset(r.end_offset)}")
                    off_str = ", ".join(offsets) if offsets else "—"
                    w(f"| {icon} | {gt_str} | {pred_str} | {iou_str} | {off_str} |")
                elif r.label == "MISSED":
                    assert r.gt
                    gt_str = f"`{_fmt_range(r.gt)}`"
                    w(f"| {icon} | {gt_str} | — | — | — |")
                elif r.label == "EXTRA":
                    assert r.pred
                    extra_count += 1
                    if extra_count <= max_extra_rows:
                        pred_str = f"`{_fmt_range(r.pred)}`"
                        w(f"| {icon} | — | {pred_str} | — | — |")
                    elif extra_count == max_extra_rows + 1:
                        remaining = sum(1 for x in ve.results if x.label == "EXTRA") - max_extra_rows
                        w(f"| {icon} | — | *… +{remaining} more* | — | — |")

            w("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Gemini call-detector predictions against ground truth.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to gemini_pilot run directory (contains run_summary.json).",
    )
    parser.add_argument(
        "--gt-s3-prefix",
        type=str,
        default="s3://rezora-whisperx-us-east-1-864981718771/labeling/corrected_boundaries/v1/",
        help="S3 prefix for ground truth boundary JSONs.",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="Minimum IoU to consider a match.")
    parser.add_argument(
        "--boundary-off-threshold",
        type=float,
        default=10.0,
        help="Seconds offset to flag BOUNDARY_OFF (default: 10).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Path to write JSON report. Default: gemini_pilot/eval/<run_id>_eval.json",
    )
    parser.add_argument(
        "--gt-cache-dir",
        type=Path,
        default=None,
        help="Local cache for GT downloads. Default: temp dir.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--print-details",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print per-video×model diagnostic timelines (default: false).",
    )
    parser.add_argument(
        "--print-top-k",
        type=int,
        default=None,
        help="When --print-details, print only the worst K video×model combos (default: all).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    console.print("[bold]Loading predictions...[/bold]")
    preds = load_predictions(args.run_dir)
    n_pred_models = sum(len(models) for models in preds.values())
    console.print(f"  Loaded predictions for {len(preds)} videos, {n_pred_models} video×model combos")
    if not preds:
        console.print("[red]No predictions found in the run directory.[/red]")
        return 1

    pred_video_ids = sorted(preds.keys())

    console.print("[bold]Loading ground truth...[/bold]")
    gt = load_ground_truth(args.gt_s3_prefix, video_ids=pred_video_ids, cache_dir=args.gt_cache_dir)
    console.print(f"  Loaded GT for {len(gt)} videos ({sum(len(v) for v in gt.values())} segments)")

    manifest_path = args.run_dir / "run_manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    prompt_version = manifest.get("prompt", {}).get("version", "unknown")
    schema_version = manifest.get("schema_version", "unknown")
    console.print(f"  Prompt version: {prompt_version} │ Schema version: {schema_version}")

    common_ids = sorted(set(gt.keys()) & set(preds.keys()))
    if not common_ids:
        console.print("[red]No overlapping video IDs between GT and predictions![/red]")
        console.print(f"  GT videos: {sorted(gt.keys())}")
        console.print(f"  Pred videos: {sorted(preds.keys())}")
        return 1

    console.print(f"  Evaluating {len(common_ids)} video(s): {', '.join(common_ids)}")

    all_evals: list[VideoEval] = []
    model_stats: dict[str, ModelSummary] = {}

    for vid in common_ids:
        gt_segs = gt[vid]
        for model, pred_segs in sorted(preds[vid].items()):
            results = match_segments(
                gt_segs,
                pred_segs,
                iou_threshold=args.iou_threshold,
                boundary_off_threshold=args.boundary_off_threshold,
            )

            ve = VideoEval(video_id=vid, model=model, gt_segments=gt_segs, pred_segments=pred_segs, results=results)
            for r in results:
                if r.label == "MATCH":
                    ve.matches += 1
                elif r.label == "BOUNDARY_OFF":
                    ve.boundary_off += 1
                    ve.matches += 1
                elif r.label == "MISSED":
                    ve.missed += 1
                elif r.label == "EXTRA":
                    ve.extra += 1

            all_evals.append(ve)

            if model not in model_stats:
                model_stats[model] = ModelSummary(model=model)
            ms = model_stats[model]
            ms.total_gt += len(gt_segs)
            ms.total_pred += len(pred_segs)
            ms.total_matches += ve.matches
            ms.total_missed += ve.missed
            ms.total_extra += ve.extra
            ms.total_boundary_off += ve.boundary_off

    summaries = list(model_stats.values())
    print_summary_table(summaries)

    if args.print_details:
        to_print = all_evals
        k = args.print_top_k
        if isinstance(k, int) and k > 0:

            def _severity(ve: VideoEval) -> tuple[int, int, int, int]:
                return (ve.missed, ve.extra, ve.boundary_off, len(ve.gt_segments))

            to_print = sorted(all_evals, key=_severity, reverse=True)[:k]
        for ve in to_print:
            print_video_eval(ve)

    run_id = args.run_dir.name
    out_json = args.out_json
    if out_json is None:
        out_dir = Path("gemini_pilot") / "eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / f"{run_id}_eval.json"

    report = _build_json_report(all_evals, summaries, args)
    report["prompt_version"] = prompt_version
    report["schema_version"] = schema_version
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    console.print(f"\n[bold]JSON report saved → {out_json}[/bold]")

    out_md = out_json.with_suffix(".md")
    md_content = _build_markdown_report(all_evals, summaries, args, prompt_version=prompt_version, schema_version=schema_version)
    out_md.write_text(md_content + "\n")
    console.print(f"[bold]Markdown report saved → {out_md}[/bold]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
