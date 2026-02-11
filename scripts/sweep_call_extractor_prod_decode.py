#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import AWS_REGION, S3_BUCKET
from pipeline.call_extractor_wavlm.decode_prod import ProdDecodeConfig, decode_production_segments
from pipeline.call_extractor_wavlm.io import cache_key_for_s3_prefix, s3_download_if_missing, s3_read_json
from pipeline.call_extractor_wavlm.labels import parse_video_labels
from pipeline.call_extractor_wavlm.metrics import (
    GATE_POLICY_VERSION,
    METRICS_VERSION,
    compute_gate_metrics,
)
from pipeline.call_extractor_wavlm.production_metrics import (
    merge_intervals,
    overlap_seconds_between_unions,
    per_call_coverages,
    purity_vs_union,
    quantiles,
    total_seconds,
)
from pipeline.call_extractor_wavlm.types import CallBoundary

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _grid(spec: str) -> list[float]:
    vals: list[float] = []
    for part in str(spec).split(","):
        part = part.strip()
        if part == "":
            continue
        vals.append(float(part))
    return vals


def _approx_audio_duration_s(times_s: np.ndarray) -> float:
    if times_s.size == 0:
        return 0.0
    dt = np.diff(times_s.astype(np.float64, copy=False))
    dt = dt[dt > 0]
    step = float(np.median(dt)) if dt.size else 0.0
    return max(0.0, float(times_s.max()) - float(times_s.min()) + float(step))


def _load_probs(
    *,
    vid: str,
    run_s3_prefix: str,
    probs_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    local = probs_dir / f"{vid}.npz"
    if not local.exists():
        s3_key = f"{run_s3_prefix}/probs/{vid}.npz"
        s3_download_if_missing(S3_BUCKET, s3_key, local, region=AWS_REGION)
    arr = np.load(local)
    return arr["times_s"], arr["in_call"], arr["start"], arr["end"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep production decoder params using cached probs (no training).")
    parser.add_argument("--split-config", type=Path, default=Path("configs/call_extractor/split_v2.config.json"))
    parser.add_argument("--output-config", type=Path, default=Path("configs/call_extractor/output_wavlm_large_v1.config.json"))
    parser.add_argument("--run-s3-prefix", type=str, required=True, help="Run prefix containing probs/{video_id}.npz")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument(
        "--write-selected-config",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write selected config to configs/call_extractor/decode_prod_v1.config.json",
    )

    # Frozen comparability params (must be passed explicitly when comparing; defaults match current policy).
    parser.add_argument("--match-tol-s", type=float, default=0.25)
    parser.add_argument("--overlap-eps-s", type=float, default=0.10)
    parser.add_argument("--min-coverage", type=float, default=0.30)

    # Grid (defaults from the user-approved production plan; must stay <=500).
    parser.add_argument("--viterbi-smooth-win-s", type=str, default="0.1,0.5")
    parser.add_argument("--viterbi-min-on-s", type=str, default="0.5,1.0,2.0")
    parser.add_argument("--viterbi-min-off-s", type=str, default="0.0,0.25,0.5")
    parser.add_argument("--viterbi-cost-mult", type=str, default="0.5,1.0")

    parser.add_argument("--split-lo", type=str, default="0.30,0.35,0.40")
    parser.add_argument("--split-min-off-s", type=str, default="0.25,0.5,1.0")
    parser.add_argument("--max-segment-s", type=str, default="600,900,1800")

    # Optional boundary-cue splitter (start/end peaks). Disabled by default.
    parser.add_argument("--use-boundary-cues", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--boundary-mode", type=str, default="pair_end_start", choices=["pair_end_start", "max_score"])
    parser.add_argument("--boundary-start-thr", type=str, default="0.70")
    parser.add_argument("--boundary-end-thr", type=str, default="0.70")
    parser.add_argument("--boundary-score-thr", type=str, default="0.70")
    parser.add_argument("--boundary-smooth-win-s", type=str, default="0.10")
    parser.add_argument("--boundary-nms-sep-s", type=str, default="0.50")
    parser.add_argument("--boundary-pair-max-gap-s", type=str, default="5.00")
    parser.add_argument("--boundary-split-margin-s", type=str, default="2.00")
    parser.add_argument("--boundary-split-gap-s", type=str, default="0.20")

    parser.add_argument("--mean-in-call-min", type=str, default="0.35,0.45,0.55")
    parser.add_argument("--max-in-call-min", type=str, default="0.60,0.70")

    parser.add_argument("--trim-s", type=str, default="0.0,1.0,2.0")
    args = parser.parse_args()

    split_cfg = _load_json(args.split_config)
    out_cfg = _load_json(args.output_config)

    run_s3_prefix = str(args.run_s3_prefix).rstrip("/")
    cache_key = cache_key_for_s3_prefix(run_s3_prefix)
    base = Path(out_cfg.get("local_artifacts_dir", "artifacts/call_extractor/wavlm_large_v1")) / "predictions" / cache_key
    probs_dir = base / "probs"
    probs_dir.mkdir(parents=True, exist_ok=True)

    eval_video_ids = [str(x) for x in list(split_cfg["eval_video_ids"])]
    train_video_ids = [str(x) for x in list(split_cfg["train_video_ids"])]
    label_prefix = str(split_cfg["label_prefix"])

    # Identify no-call videos (eval + train) from labels (one-time).
    labels_by_vid: dict[str, list[CallBoundary]] = {}
    for vid in eval_video_ids + train_video_ids:
        key = f"{label_prefix}{vid}.json"
        data = s3_read_json(S3_BUCKET, key, region=AWS_REGION)
        bounds = parse_video_labels(data).boundaries
        labels_by_vid[vid] = list(bounds)

    eval_no_call_vids = [vid for vid in eval_video_ids if len(labels_by_vid.get(vid, [])) == 0]
    train_no_call_vids = [vid for vid in train_video_ids if len(labels_by_vid.get(vid, [])) == 0]
    no_call_vids = list(sorted(set(eval_no_call_vids + train_no_call_vids)))

    # Eval set for production metrics uses eval videos plus extra no-call videos (to estimate NO-CALL FP).
    eval_set_vids = list(sorted(set(eval_video_ids + no_call_vids)))

    logger.info(
        "Sweep setup: run=%s eval=%d eval_no_call=%d train_no_call=%d total_eval_vids=%d",
        run_s3_prefix,
        len(eval_video_ids),
        len(eval_no_call_vids),
        len(train_no_call_vids),
        len(eval_set_vids),
    )

    # Load probs once.
    probs_by_vid: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    missing_probs: list[str] = []
    for vid in eval_set_vids:
        try:
            probs_by_vid[vid] = _load_probs(vid=str(vid), run_s3_prefix=run_s3_prefix, probs_dir=probs_dir)
        except Exception:
            missing_probs.append(str(vid))
    if missing_probs:
        logger.warning("Missing probs for %d/%d eval-set vids; continuing without them: %s", len(missing_probs), len(eval_set_vids), missing_probs)
        eval_set_vids = [vid for vid in eval_set_vids if vid not in set(missing_probs)]

    # Build grid.
    grids = {
        "viterbi_smooth_win_s": _grid(args.viterbi_smooth_win_s),
        "viterbi_min_on_s": _grid(args.viterbi_min_on_s),
        "viterbi_min_off_s": _grid(args.viterbi_min_off_s),
        "viterbi_cost_mult": _grid(args.viterbi_cost_mult),
        "split_lo": _grid(args.split_lo),
        "split_min_off_s": _grid(args.split_min_off_s),
        "max_segment_s": _grid(args.max_segment_s),
        "mean_in_call_min": _grid(args.mean_in_call_min),
        "max_in_call_min": _grid(args.max_in_call_min),
        "trim_s": _grid(args.trim_s),
    }
    if bool(args.use_boundary_cues):
        grids.update(
            {
                "boundary_start_thr": _grid(args.boundary_start_thr),
                "boundary_end_thr": _grid(args.boundary_end_thr),
                "boundary_score_thr": _grid(args.boundary_score_thr),
                "boundary_smooth_win_s": _grid(args.boundary_smooth_win_s),
                "boundary_nms_sep_s": _grid(args.boundary_nms_sep_s),
                "boundary_pair_max_gap_s": _grid(args.boundary_pair_max_gap_s),
                "boundary_split_margin_s": _grid(args.boundary_split_margin_s),
                "boundary_split_gap_s": _grid(args.boundary_split_gap_s),
            }
        )

    keys = list(grids.keys())
    combos = list(itertools.product(*(grids[k] for k in keys)))
    if len(combos) > 500:
        raise ValueError(f"Grid too large: {len(combos)} > 500 (refuse to run)")

    # Per-config evaluation.
    rows: list[dict[str, Any]] = []
    for idx, vals in enumerate(combos, start=1):
        params = {k: float(v) for k, v in zip(keys, vals)}

        cfg = ProdDecodeConfig(
            # Keep base penalties fixed for now; cost_mult scales both together.
            viterbi_off_to_on_penalty=ProdDecodeConfig.viterbi_off_to_on_penalty,
            viterbi_on_to_off_penalty=ProdDecodeConfig.viterbi_on_to_off_penalty,
            viterbi_cost_mult=float(params["viterbi_cost_mult"]),
            viterbi_start_scale=0.0,
            viterbi_end_scale=0.0,
            viterbi_min_on_s=float(params["viterbi_min_on_s"]),
            viterbi_min_off_s=float(params["viterbi_min_off_s"]),
            viterbi_smooth_win_s=float(params["viterbi_smooth_win_s"]),
            split_lo=float(params["split_lo"]),
            split_min_off_s=float(params["split_min_off_s"]),
            use_boundary_cues=bool(args.use_boundary_cues),
            boundary_mode=str(args.boundary_mode),
            boundary_start_thr=float(params.get("boundary_start_thr", ProdDecodeConfig.boundary_start_thr)),
            boundary_end_thr=float(params.get("boundary_end_thr", ProdDecodeConfig.boundary_end_thr)),
            boundary_score_thr=float(params.get("boundary_score_thr", ProdDecodeConfig.boundary_score_thr)),
            boundary_smooth_win_s=float(params.get("boundary_smooth_win_s", ProdDecodeConfig.boundary_smooth_win_s)),
            boundary_nms_sep_s=float(params.get("boundary_nms_sep_s", ProdDecodeConfig.boundary_nms_sep_s)),
            boundary_pair_max_gap_s=float(params.get("boundary_pair_max_gap_s", ProdDecodeConfig.boundary_pair_max_gap_s)),
            boundary_split_margin_s=float(params.get("boundary_split_margin_s", ProdDecodeConfig.boundary_split_margin_s)),
            boundary_split_gap_s=float(params.get("boundary_split_gap_s", ProdDecodeConfig.boundary_split_gap_s)),
            max_segment_s=float(params["max_segment_s"]),
            mean_in_call_min=float(params["mean_in_call_min"]),
            max_in_call_min=float(params["max_in_call_min"]),
            trim_s=float(params["trim_s"]),
        )

        # Aggregates
        merges = 0
        oversplits = 0
        fp_total = 0
        gt_calls = 0
        pred_calls = 0
        matched_calls = 0
        kept_iou_0_5 = 0
        kept_cov = 0

        segment_purities: list[float] = []
        call_coverages: list[float] = []
        calls_cov_ge_0_8 = 0
        gt_union_seconds_total = 0.0
        pred_union_seconds_total = 0.0
        union_overlap_seconds_total = 0.0
        non_call_seconds_total = 0.0
        overlap_seconds_total = 0.0
        pred_seconds_total = 0.0
        pred_segments_total = 0
        audio_seconds_total = 0.0

        fp_no_call_seconds = 0.0
        fp_no_call_segments = 0
        no_call_audio_seconds = 0.0

        for vid in eval_set_vids:
            times_s, in_call, start, end = probs_by_vid[str(vid)]
            segs = decode_production_segments(times_s=times_s, in_call_p=in_call, start_p=start, end_p=end, cfg=cfg)
            pred_intervals = [(float(s["start_s"]), float(s["end_s"])) for s in segs]
            pred_segments_total += int(len(pred_intervals))
            pred_seconds_total += float(sum(max(0.0, e - s) for s, e in pred_intervals))

            audio_dur = _approx_audio_duration_s(times_s)
            audio_seconds_total += float(audio_dur)

            gt_bounds = labels_by_vid.get(str(vid), [])
            gt_intervals = [(float(b.start_s), float(b.end_s)) for b in gt_bounds]
            gt_union = merge_intervals(gt_intervals, join_tolerance_s=0.0)
            pred_union = merge_intervals(pred_intervals, join_tolerance_s=0.0)

            purity = purity_vs_union(pred_intervals=pred_intervals, gt_union_intervals=gt_union)
            segment_purities.extend(purity.segment_purities)
            non_call_seconds_total += float(purity.non_call_seconds)
            overlap_seconds_total += float(purity.overlap_seconds)

            if str(vid) in eval_video_ids and len(gt_intervals) > 0:
                covs = per_call_coverages(gt_calls=gt_intervals, pred_intervals=pred_intervals)
                call_coverages.extend(covs)
                calls_cov_ge_0_8 += int(sum(1 for c in covs if float(c) >= 0.8))

                gt_union_s = total_seconds(gt_union)
                pred_union_s = total_seconds(pred_union)
                ov_union = overlap_seconds_between_unions(pred_union, gt_union)
                gt_union_seconds_total += float(gt_union_s)
                pred_union_seconds_total += float(pred_union_s)
                union_overlap_seconds_total += float(ov_union)

            # No-call FP metrics
            if len(gt_bounds) == 0:
                fp_no_call_seconds += float(purity.pred_seconds)
                fp_no_call_segments += int(len(pred_intervals))
                no_call_audio_seconds += float(audio_dur)

            # Frozen comparability gate metrics: aggregate on eval split only.
            if str(vid) in eval_video_ids:
                m = compute_gate_metrics(
                    gt=list(gt_bounds),
                    pred=[CallBoundary(start_s=float(s), end_s=float(e)) for s, e in pred_intervals],
                    match_tol_s=float(args.match_tol_s),
                    overlap_eps_s=float(args.overlap_eps_s),
                    min_coverage=float(args.min_coverage),
                )
                merges += int(m.merges)
                oversplits += int(m.oversplits)
                fp_total += int(m.false_positive_segments)
                gt_calls += int(m.gt_calls)
                pred_calls += int(m.pred_calls)
                matched_calls += int(m.matched_calls)
                kept_iou_0_5 += int(m.kept_calls_iou_0_5)
                kept_cov += int(m.kept_calls_coverage)

        no_call_hours = float(no_call_audio_seconds) / 3600.0 if no_call_audio_seconds > 0 else 0.0
        fp_no_call_sec_per_hr = (float(fp_no_call_seconds) / float(no_call_hours)) if no_call_hours > 0 else 0.0

        total_hours = float(audio_seconds_total) / 3600.0 if audio_seconds_total > 0 else 0.0
        segments_per_hour = (float(pred_segments_total) / float(total_hours)) if total_hours > 0 else 0.0
        predicted_seconds_per_hour = (float(pred_seconds_total) / float(total_hours)) if total_hours > 0 else 0.0

        purity_p10, purity_med = quantiles(segment_purities, qs=[0.10, 0.50])
        cov_p10, cov_med = quantiles(call_coverages, qs=[0.10, 0.50])
        calls_cov_ge_0_8_rate = (float(calls_cov_ge_0_8) / float(gt_calls)) if gt_calls > 0 else 0.0
        union_recall = (float(union_overlap_seconds_total) / float(gt_union_seconds_total)) if gt_union_seconds_total > 0 else 0.0
        union_purity = (float(union_overlap_seconds_total) / float(pred_union_seconds_total)) if pred_union_seconds_total > 0 else 0.0

        keep_rate_iou_0_5 = (kept_iou_0_5 / gt_calls) if gt_calls > 0 else 0.0
        keep_rate_cov = (kept_cov / gt_calls) if gt_calls > 0 else 0.0
        raw_keep = (matched_calls / gt_calls) if gt_calls > 0 else 0.0

        rows.append(
            {
                "idx": int(idx),
                "params": params,
                "merges": int(merges),
                "oversplits": int(oversplits),
                "fp_total": int(fp_total),
                "gt_calls": int(gt_calls),
                "pred_calls": int(pred_calls),
                "matched_calls": int(matched_calls),
                "kept_calls_iou_0_5": int(kept_iou_0_5),
                "keep_rate_iou_0_5": float(keep_rate_iou_0_5),
                "keep_rate_coverage": float(keep_rate_cov),
                "raw_keep": float(raw_keep),
                "fp_no_call_seconds": float(fp_no_call_seconds),
                "fp_no_call_seconds_per_hour": float(fp_no_call_sec_per_hr),
                "fp_no_call_segments": int(fp_no_call_segments),
                "purity_segment_p10": purity_p10,
                "purity_segment_median": purity_med,
                "non_call_seconds_total": float(non_call_seconds_total),
                "recovered_call_seconds": float(overlap_seconds_total),
                "calls_coverage_ge_0_8": int(calls_cov_ge_0_8),
                "calls_coverage_ge_0_8_rate": float(calls_cov_ge_0_8_rate),
                "call_coverage_p10": cov_p10,
                "call_coverage_median": cov_med,
                "union_recall": float(union_recall),
                "union_purity": float(union_purity),
                "segments_per_hour": float(segments_per_hour),
                "predicted_seconds_per_hour": float(predicted_seconds_per_hour),
            }
        )

        if idx % 25 == 0 or idx == len(combos):
            logger.info("sweep %d/%d done (latest merges=%d fp_no_call_sec/hr=%.3f)", idx, len(combos), merges, fp_no_call_sec_per_hr)

    # Rank configs
    def sort_key(r: dict[str, Any]) -> tuple:
        return (
            int(r["merges"]),
            -float(r.get("calls_coverage_ge_0_8_rate") or 0.0),
            -float(r.get("union_recall") or 0.0),
            -float(r.get("union_purity") or 0.0),
            float(r["non_call_seconds_total"]),
            float(r["fp_no_call_seconds_per_hour"]),
            float(r["segments_per_hour"]),
        )

    rows_sorted = sorted(rows, key=sort_key)
    best = rows_sorted[0] if rows_sorted else None

    # Selected config: best row params merged into defaults.
    selected_cfg = None
    if best is not None:
        selected_cfg = ProdDecodeConfig(
            viterbi_off_to_on_penalty=ProdDecodeConfig.viterbi_off_to_on_penalty,
            viterbi_on_to_off_penalty=ProdDecodeConfig.viterbi_on_to_off_penalty,
            viterbi_cost_mult=float(best["params"]["viterbi_cost_mult"]),
            viterbi_start_scale=0.0,
            viterbi_end_scale=0.0,
            viterbi_min_on_s=float(best["params"]["viterbi_min_on_s"]),
            viterbi_min_off_s=float(best["params"]["viterbi_min_off_s"]),
            viterbi_smooth_win_s=float(best["params"]["viterbi_smooth_win_s"]),
            split_lo=float(best["params"]["split_lo"]),
            split_min_off_s=float(best["params"]["split_min_off_s"]),
            use_boundary_cues=bool(args.use_boundary_cues),
            boundary_mode=str(args.boundary_mode),
            boundary_start_thr=float(best["params"].get("boundary_start_thr", ProdDecodeConfig.boundary_start_thr)),
            boundary_end_thr=float(best["params"].get("boundary_end_thr", ProdDecodeConfig.boundary_end_thr)),
            boundary_score_thr=float(best["params"].get("boundary_score_thr", ProdDecodeConfig.boundary_score_thr)),
            boundary_smooth_win_s=float(best["params"].get("boundary_smooth_win_s", ProdDecodeConfig.boundary_smooth_win_s)),
            boundary_nms_sep_s=float(best["params"].get("boundary_nms_sep_s", ProdDecodeConfig.boundary_nms_sep_s)),
            boundary_pair_max_gap_s=float(best["params"].get("boundary_pair_max_gap_s", ProdDecodeConfig.boundary_pair_max_gap_s)),
            boundary_split_margin_s=float(best["params"].get("boundary_split_margin_s", ProdDecodeConfig.boundary_split_margin_s)),
            boundary_split_gap_s=float(best["params"].get("boundary_split_gap_s", ProdDecodeConfig.boundary_split_gap_s)),
            max_segment_s=float(best["params"]["max_segment_s"]),
            mean_in_call_min=float(best["params"]["mean_in_call_min"]),
            max_in_call_min=float(best["params"]["max_in_call_min"]),
            trim_s=float(best["params"]["trim_s"]),
        )

    report = {
        "metrics_version": str(METRICS_VERSION),
        "gate_policy_version": str(GATE_POLICY_VERSION),
        "split_config_path": str(args.split_config),
        "label_prefix": str(label_prefix),
        "run_s3_prefix": str(run_s3_prefix),
        "comparability": {
            "match_tol_s": float(args.match_tol_s),
            "overlap_eps_s": float(args.overlap_eps_s),
            "min_coverage": float(args.min_coverage),
        },
        "eval_video_ids": eval_video_ids,
        "eval_no_call_video_ids": eval_no_call_vids,
        "train_no_call_video_ids": train_no_call_vids,
        "eval_set_video_ids": eval_set_vids,
        "missing_probs_video_ids": missing_probs,
        "grid_size": int(len(combos)),
        "top": rows_sorted[:25],
        "best": best,
        "selected_config": selected_cfg.__dict__ if selected_cfg is not None else None,
    }

    if args.out_json is None:
        args.out_json = Path("reports") / f"prod_decode_sweep_{cache_key}_{_utc_ts()}.json"
    if args.out_md is None:
        args.out_md = Path("reports") / f"prod_decode_sweep_{cache_key}_{_utc_ts()}.md"

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    md: list[str] = []
    md.append("# Production decode sweep report")
    md.append("")
    md.append(f"- run_s3_prefix: `{run_s3_prefix}`")
    md.append(f"- split_config_path: `{args.split_config}`")
    md.append(f"- metrics_version: `{METRICS_VERSION}`")
    md.append(f"- gate_policy_version: `{GATE_POLICY_VERSION}`")
    md.append(f"- match_tol_s: {args.match_tol_s}")
    md.append(f"- overlap_eps_s: {args.overlap_eps_s}")
    md.append(f"- min_coverage: {args.min_coverage}")
    md.append(f"- grid_size: {len(combos)}")
    if missing_probs:
        md.append(f"- missing_probs_video_ids: {missing_probs}")
    md.append("")
    md.append("## Best config (ranked by objective)")
    if best is None:
        md.append("- (none)")
    else:
        md.append(f"- merges: {best['merges']}")
        md.append(f"- calls_coverage_ge_0_8_rate: {best.get('calls_coverage_ge_0_8_rate', 0.0):.3f}")
        md.append(f"- union_recall: {best.get('union_recall', 0.0):.3f}")
        md.append(f"- union_purity: {best.get('union_purity', 0.0):.3f}")
        md.append(f"- non_call_seconds_total: {best['non_call_seconds_total']:.3f}")
        md.append(f"- fp_no_call_seconds_per_hour (reporting): {best['fp_no_call_seconds_per_hour']:.3f}")
        md.append(f"- segments_per_hour: {best['segments_per_hour']:.3f}")
        md.append(f"- keep_rate_iou_0_5 (reporting-only): {best['keep_rate_iou_0_5']:.3f}")
        md.append(f"- params: `{json.dumps(best['params'], sort_keys=True)}`")
    md.append("")
    md.append("## Top 10 table")
    md.append("")
    md.append("| idx | merges | cov>=0.8 | union_recall | union_purity | fp_no_call_sec/hr | seg/hr | keep@0.5 | params |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows_sorted[:10]:
        md.append(
            f"| {r['idx']} | {r['merges']} | {r.get('calls_coverage_ge_0_8_rate', 0.0):.3f} | {r.get('union_recall', 0.0):.3f} | {r.get('union_purity', 0.0):.3f} | {r['fp_no_call_seconds_per_hour']:.3f} | {r['segments_per_hour']:.1f} | {r['keep_rate_iou_0_5']:.3f} | `{json.dumps(r['params'], sort_keys=True)}` |"
        )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md) + "\n")

    if bool(args.write_selected_config) and selected_cfg is not None:
        out_cfg_path = Path("configs/call_extractor/decode_prod_v1.config.json")
        out_cfg_path.write_text(json.dumps(selected_cfg.__dict__, indent=2, sort_keys=True) + "\n")
        logger.info("Wrote selected config to %s", out_cfg_path)

    logger.info("Wrote %s and %s", args.out_json, args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
