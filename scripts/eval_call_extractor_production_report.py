#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import AWS_REGION, S3_BUCKET
from pipeline.call_extractor_wavlm.audio_cache import ensure_flac_cached
from pipeline.call_extractor_wavlm.io import (
    build_audio_index,
    cache_key_for_s3_prefix,
    ffprobe_duration_s,
    git_short_sha,
    s3_download_if_missing,
    s3_read_json,
)
from pipeline.call_extractor_wavlm.labels import parse_video_labels
from pipeline.call_extractor_wavlm.metrics import (
    GATE_POLICY_VERSION,
    METRICS_VERSION,
    boundaries_from_json_segments,
    compute_gate_metrics,
)
from pipeline.call_extractor_wavlm.production_metrics import merge_intervals, purity_vs_union, quantiles
from pipeline.call_extractor_wavlm.types import CallBoundary

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_segments(local_path: Path) -> list[dict]:
    data = json.loads(local_path.read_text())
    return list(data.get("segments", []))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate decoded call segments with production metrics (additive).")
    parser.add_argument("--split-config", type=Path, required=True, help="Must be split_v2 (kept untouched).")
    parser.add_argument("--output-config", type=Path, default=Path("configs/call_extractor/output_wavlm_large_v1.config.json"))
    parser.add_argument(
        "--segments-s3-prefix",
        type=str,
        required=True,
        help="Decode-scoped prefix containing segments/{video_id}.json",
    )
    parser.add_argument(
        "--include-no-call-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include train no-call videos when computing NO-CALL FP metrics (recommended; eval has only 2).",
    )
    parser.add_argument("--match-tol-s", type=float, default=0.25)
    parser.add_argument("--overlap-eps-s", type=float, default=0.10)
    parser.add_argument("--min-coverage", type=float, default=0.30)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    split_cfg = _load_json(args.split_config)
    out_cfg = _load_json(args.output_config)
    label_prefix = str(split_cfg["label_prefix"])

    segments_prefix = str(args.segments_s3_prefix).rstrip("/")
    segments_s3_prefix = segments_prefix
    if not segments_s3_prefix.endswith("/segments"):
        segments_s3_prefix = f"{segments_s3_prefix}/segments"

    cache_key = cache_key_for_s3_prefix(str(args.segments_s3_prefix))
    base_local = Path(out_cfg.get("local_artifacts_dir", "artifacts/call_extractor/wavlm_large_v1")) / "predictions" / cache_key
    base_local.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).parent.parent
    git_sha = git_short_sha(repo_root)

    eval_video_ids = [str(x) for x in list(split_cfg["eval_video_ids"])]
    train_video_ids = [str(x) for x in list(split_cfg["train_video_ids"])]

    # Load labels for eval + train so we can identify no-call videos.
    labels_by_vid: dict[str, list[tuple[float, float]]] = {}
    for vid in eval_video_ids + train_video_ids:
        key = f"{label_prefix}{vid}.json"
        data = s3_read_json(S3_BUCKET, key, region=AWS_REGION)
        bounds = parse_video_labels(data).boundaries
        labels_by_vid[vid] = [(float(b.start_s), float(b.end_s)) for b in bounds]

    eval_no_call_vids = [vid for vid in eval_video_ids if len(labels_by_vid.get(vid, [])) == 0]
    train_no_call_vids = [vid for vid in train_video_ids if len(labels_by_vid.get(vid, [])) == 0]

    eval_set_vids = list(eval_video_ids)
    no_call_vids = list(eval_no_call_vids)
    if bool(args.include_no_call_train):
        no_call_vids = list(sorted(set(no_call_vids + train_no_call_vids)))
        eval_set_vids = list(sorted(set(eval_set_vids + no_call_vids)))

    logger.info(
        "Production eval sets: eval=%d eval_no_call=%d train_no_call=%d include_train_no_call=%s total_eval_vids=%d",
        len(eval_video_ids),
        len(eval_no_call_vids),
        len(train_no_call_vids),
        bool(args.include_no_call_train),
        len(eval_set_vids),
    )

    # Audio duration resolution (prefer cached FLAC; fall back to downloading mp3).
    cache_dir = Path(out_cfg.get("local_cache_dir", ".cache/call_extractor_wavlm"))
    audio_cache_dir = cache_dir / "audio"
    flac_cache_dir = cache_dir / "audio_flac"
    audio_cache_dir.mkdir(parents=True, exist_ok=True)
    flac_cache_dir.mkdir(parents=True, exist_ok=True)

    audio_index = build_audio_index(S3_BUCKET, list(split_cfg["audio_prefixes"]), region=AWS_REGION)

    def audio_duration_s(vid: str) -> float:
        flac = flac_cache_dir / f"{vid}.flac"
        if flac.exists():
            return float(ffprobe_duration_s(flac))

        mp3_key = audio_index.get(str(vid))
        if not mp3_key:
            raise FileNotFoundError(f"Audio key not found for {vid}")
        mp3 = audio_cache_dir / f"{vid}.mp3"
        s3_download_if_missing(S3_BUCKET, mp3_key, mp3, region=AWS_REGION)
        ensure_flac_cached(mp3_path=mp3, flac_path=flac, sr_hz=16000)
        return float(ffprobe_duration_s(flac))

    # Load segments for all vids we will evaluate.
    segs_by_vid: dict[str, list[tuple[float, float]]] = {}
    missing_segments: list[str] = []
    for vid in eval_set_vids:
        local = base_local / f"{vid}.json"
        if not local.exists():
            s3_key = f"{segments_s3_prefix}/{vid}.json"
            try:
                s3_download_if_missing(S3_BUCKET, s3_key, local, region=AWS_REGION)
            except Exception:
                missing_segments.append(vid)
                continue
        seg_bounds = boundaries_from_json_segments(_load_segments(local))
        segs_by_vid[vid] = [(float(b.start_s), float(b.end_s)) for b in seg_bounds]

    if missing_segments:
        raise FileNotFoundError(
            f"Missing segments for {len(missing_segments)}/{len(eval_set_vids)} videos under {segments_s3_prefix}: {missing_segments}"
        )

    # Compute gate_metrics_v1 (eval videos only, consistent with existing eval script).
    gate_agg_eval = {
        "merges": 0,
        "oversplits": 0,
        "gt_calls": 0,
        "pred_calls": 0,
        "matched_calls": 0,
        "kept_calls_coverage": 0,
        "kept_calls_iou_0_5": 0,
        "kept_calls_iou_0_8": 0,
        "false_positive_segments": 0,
        "false_positive_segments_no_call_videos": 0,
        "false_positive_segments_call_outside_gt": 0,
    }

    per_video_rows: list[dict] = []
    segment_purities_all: list[float] = []
    non_call_seconds_total = 0.0
    pred_seconds_total = 0.0
    pred_segments_total = 0
    audio_seconds_total = 0.0

    fp_no_call_seconds = 0.0
    fp_no_call_segments = 0
    no_call_audio_seconds = 0.0

    # Per-video no-call breakdown
    no_call_rows: list[dict] = []

    for vid in eval_set_vids:
        gt_bounds = labels_by_vid.get(vid, [])
        pred_bounds = segs_by_vid.get(vid, [])

        dur_s = audio_duration_s(vid)
        audio_seconds_total += float(dur_s)

        pred_seconds = sum(max(0.0, float(e) - float(s)) for s, e in pred_bounds)
        pred_segments = int(sum(1 for s, e in pred_bounds if float(e) > float(s)))
        pred_seconds_total += float(pred_seconds)
        pred_segments_total += int(pred_segments)

        gt_union = merge_intervals(gt_bounds, join_tolerance_s=0.0)
        purity = purity_vs_union(pred_intervals=pred_bounds, gt_union_intervals=gt_union)
        segment_purities_all.extend(purity.segment_purities)
        non_call_seconds_total += float(purity.non_call_seconds)

        is_no_call = len(gt_bounds) == 0
        if is_no_call:
            fp_no_call_seconds += float(pred_seconds)
            fp_no_call_segments += int(pred_segments)
            no_call_audio_seconds += float(dur_s)
            no_call_rows.append(
                {
                    "video_id": vid,
                    "audio_seconds": float(dur_s),
                    "pred_segments": int(pred_segments),
                    "pred_seconds": float(pred_seconds),
                }
            )

        # Gate metrics (eval-only aggregation)
        if vid in eval_video_ids:
            m = compute_gate_metrics(
                gt=[CallBoundary(start_s=float(s), end_s=float(e)) for s, e in gt_bounds],
                pred=[CallBoundary(start_s=float(s), end_s=float(e)) for s, e in pred_bounds],
                match_tol_s=float(args.match_tol_s),
                overlap_eps_s=float(args.overlap_eps_s),
                min_coverage=float(args.min_coverage),
            )
            gate_agg_eval["merges"] += int(m.merges)
            gate_agg_eval["oversplits"] += int(m.oversplits)
            gate_agg_eval["gt_calls"] += int(m.gt_calls)
            gate_agg_eval["pred_calls"] += int(m.pred_calls)
            gate_agg_eval["matched_calls"] += int(m.matched_calls)
            gate_agg_eval["kept_calls_coverage"] += int(getattr(m, "kept_calls_coverage", 0))
            gate_agg_eval["kept_calls_iou_0_5"] += int(m.kept_calls_iou_0_5)
            gate_agg_eval["kept_calls_iou_0_8"] += int(m.kept_calls_iou_0_8)
            gate_agg_eval["false_positive_segments"] += int(m.false_positive_segments)
            if int(m.gt_calls) == 0:
                gate_agg_eval["false_positive_segments_no_call_videos"] += int(m.false_positive_segments)
            else:
                gate_agg_eval["false_positive_segments_call_outside_gt"] += int(m.false_positive_segments)

        per_video_rows.append(
            {
                "video_id": vid,
                "is_no_call": bool(is_no_call),
                "audio_seconds": float(dur_s),
                "gt_calls": int(len(gt_bounds)),
                "pred_segments": int(pred_segments),
                "pred_seconds": float(pred_seconds),
                "purity_video": (float(purity.overlap_seconds) / float(purity.pred_seconds)) if purity.pred_seconds > 0 else None,
                "non_call_seconds": float(purity.non_call_seconds),
            }
        )

    # Aggregate production metrics.
    pred_hours = float(audio_seconds_total) / 3600.0 if audio_seconds_total > 0 else 0.0
    segments_per_hour = (float(pred_segments_total) / float(pred_hours)) if pred_hours > 0 else 0.0
    predicted_seconds_per_hour = (float(pred_seconds_total) / float(pred_hours)) if pred_hours > 0 else 0.0

    purity_seg_p10, purity_seg_median = quantiles(segment_purities_all, qs=[0.10, 0.50])

    no_call_hours = float(no_call_audio_seconds) / 3600.0 if no_call_audio_seconds > 0 else 0.0
    fp_no_call_seconds_per_hour = (float(fp_no_call_seconds) / float(no_call_hours)) if no_call_hours > 0 else 0.0

    keep_rate = (gate_agg_eval["matched_calls"] / gate_agg_eval["gt_calls"]) if gate_agg_eval["gt_calls"] > 0 else 0.0
    keep_rate_cov = (gate_agg_eval["kept_calls_coverage"] / gate_agg_eval["gt_calls"]) if gate_agg_eval["gt_calls"] > 0 else 0.0
    keep_rate_iou_0_5 = (gate_agg_eval["kept_calls_iou_0_5"] / gate_agg_eval["gt_calls"]) if gate_agg_eval["gt_calls"] > 0 else 0.0
    keep_rate_iou_0_8 = (gate_agg_eval["kept_calls_iou_0_8"] / gate_agg_eval["gt_calls"]) if gate_agg_eval["gt_calls"] > 0 else 0.0

    report = {
        "metrics_version": str(METRICS_VERSION),
        "gate_policy_version": str(GATE_POLICY_VERSION),
        "git_sha": str(git_sha),
        "split_config_path": str(args.split_config),
        "label_prefix": str(label_prefix),
        "segments_s3_prefix": str(segments_s3_prefix),
        "match_tol_s": float(args.match_tol_s),
        "overlap_eps_s": float(args.overlap_eps_s),
        "min_coverage": float(args.min_coverage),
        "eval_video_ids": eval_video_ids,
        "eval_set_video_ids": eval_set_vids,
        "eval_no_call_video_ids": eval_no_call_vids,
        "train_no_call_video_ids": train_no_call_vids,
        "no_call_eval_plus_train_video_ids": no_call_vids,
        "gate_metrics_eval": {
            **gate_agg_eval,
            "keep_rate": float(keep_rate),
            "keep_rate_coverage": float(keep_rate_cov),
            "keep_rate_iou_0_5": float(keep_rate_iou_0_5),
            "keep_rate_iou_0_8": float(keep_rate_iou_0_8),
        },
        "production_metrics": {
            "fp_no_call_segments": int(fp_no_call_segments),
            "fp_no_call_seconds": float(fp_no_call_seconds),
            "fp_no_call_seconds_per_hour": float(fp_no_call_seconds_per_hour),
            "purity_segment_p10": purity_seg_p10,
            "purity_segment_median": purity_seg_median,
            "non_call_seconds_total": float(non_call_seconds_total),
            "segments_per_hour": float(segments_per_hour),
            "predicted_seconds_per_hour": float(predicted_seconds_per_hour),
        },
        "per_video": sorted(per_video_rows, key=lambda r: (int(r["is_no_call"]), -float(r["pred_seconds"]))),
        "no_call_per_video": sorted(no_call_rows, key=lambda r: -float(r["pred_seconds"])),
    }

    if args.out_json is None:
        args.out_json = Path("reports") / f"prod_eval_{cache_key}.json"
    if args.out_md is None:
        args.out_md = Path("reports") / f"prod_eval_{cache_key}.md"

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    md_lines = [
        "# Call extractor production report",
        "",
        f"- git_sha: `{git_sha}`",
        f"- split_config_path: `{args.split_config}`",
        f"- label_prefix: `{label_prefix}`",
        f"- segments_s3_prefix: `{segments_s3_prefix}`",
        f"- match_tol_s: {args.match_tol_s}",
        f"- overlap_eps_s: {args.overlap_eps_s}",
        f"- min_coverage: {args.min_coverage}",
        "",
        "## Gate metrics (eval only; unchanged formulas)",
        f"- merges: {gate_agg_eval['merges']}",
        f"- oversplits: {gate_agg_eval['oversplits']}",
        f"- FP_total: {gate_agg_eval['false_positive_segments']}",
        f"- keep_rate_iou_0_5: {keep_rate_iou_0_5:.3f}",
        f"- keep_rate_coverage: {keep_rate_cov:.3f}",
        "",
        "## Production metrics (selection)",
        f"- fp_no_call_seconds_per_hour: {fp_no_call_seconds_per_hour:.3f}",
        f"- fp_no_call_seconds: {fp_no_call_seconds:.3f}",
        f"- fp_no_call_segments: {fp_no_call_segments}",
        f"- purity_segment_p10: {purity_seg_p10}",
        f"- purity_segment_median: {purity_seg_median}",
        f"- non_call_seconds_total: {non_call_seconds_total:.3f}",
        f"- segments_per_hour: {segments_per_hour:.3f}",
        f"- predicted_seconds_per_hour: {predicted_seconds_per_hour:.3f}",
        "",
        "## No-call per-video breakdown",
    ]
    for r in report["no_call_per_video"]:
        md_lines.append(
            f"- {r['video_id']}: pred_segments={r['pred_segments']} pred_seconds={r['pred_seconds']:.3f} audio_seconds={r['audio_seconds']:.3f}"
        )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md_lines) + "\n")

    logger.info(f"Wrote {args.out_json} and {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
