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
from pipeline.call_extractor_wavlm.io import cache_key_for_s3_prefix, s3_download_if_missing, s3_read_json
from pipeline.call_extractor_wavlm.labels import parse_video_labels
from pipeline.call_extractor_wavlm.metrics import (
    GATE_POLICY_VERSION,
    METRICS_VERSION,
    boundaries_from_json_segments,
    compute_gate_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate predicted call segments against boundary labels.")
    parser.add_argument(
        "--split-config",
        type=Path,
        default=None,
        help="Path to split config. If omitted, uses configs/call_extractor/split_v2.config.json (per AGENTS.md, pass explicitly).",
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=Path("configs/call_extractor/output_wavlm_large_v1.config.json"),
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default=None,
        help="Override S3 prefix for downloading segments (default: output-config's s3_output_prefix)",
    )
    parser.add_argument(
        "--segments-dir",
        type=Path,
        default=None,
        help="Local directory containing {video_id}.json (default: <local_artifacts_dir>/predictions)",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("artifacts/call_extractor/wavlm_large_v1/eval_report.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("artifacts/call_extractor/wavlm_large_v1/eval_report.md"),
    )
    parser.add_argument("--match-tol-s", type=float, default=0.25, help="Matching tolerance in seconds (tol=0.0 is always reported too).")
    parser.add_argument("--overlap-eps-s", type=float, default=0.10, help="Minimum tolerance-adjusted overlap (seconds) to count as overlap.")
    parser.add_argument("--min-coverage", type=float, default=0.30, help="Minimum exact coverage required to count a GT call as kept.")
    args = parser.parse_args()

    default_split_config = Path("configs/call_extractor/split_v2.config.json")
    split_config_path = args.split_config
    if split_config_path is None:
        split_config_path = default_split_config
        logger.warning(f"--split-config omitted; using default {split_config_path} (per AGENTS.md, pass --split-config explicitly).")

    split_cfg = _load_json(split_config_path)
    out_cfg = _load_json(args.output_config)
    s3_prefix = str(args.s3_prefix or out_cfg.get("s3_output_prefix", "call_extractor/wavlm_large_v1/")).rstrip("/")

    segments_dir = args.segments_dir
    if segments_dir is None:
        cache_key = cache_key_for_s3_prefix(s3_prefix)
        segments_dir = (
            Path(out_cfg.get("local_artifacts_dir", "artifacts/call_extractor/wavlm_large_v1"))
            / "predictions"
            / cache_key
        )
    segments_dir.mkdir(parents=True, exist_ok=True)

    eval_video_ids = list(split_cfg["eval_video_ids"])
    label_prefix = str(split_cfg["label_prefix"])

    def build_report(*, match_tol_s: float) -> dict:
        per_video: list[dict] = []
        agg = {
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
        start_err_sum = 0.0
        end_err_sum = 0.0
        iou_sum = 0.0
        cov_sum = 0.0

        for vid in eval_video_ids:
            vid = str(vid)
            label_key = f"{label_prefix}{vid}.json"
            labels = parse_video_labels(s3_read_json(S3_BUCKET, label_key, region=AWS_REGION))

            local_seg = segments_dir / f"{vid}.json"
            if not local_seg.exists():
                s3_key = f"{s3_prefix}/segments/{vid}.json"
                s3_download_if_missing(S3_BUCKET, s3_key, local_seg, region=AWS_REGION)

            pred_json = json.loads(local_seg.read_text())
            pred_bounds = boundaries_from_json_segments(pred_json.get("segments", []))

            m = compute_gate_metrics(
                gt=labels.boundaries,
                pred=pred_bounds,
                match_tol_s=float(match_tol_s),
                overlap_eps_s=float(args.overlap_eps_s),
                min_coverage=float(args.min_coverage),
            )
            agg["merges"] += int(m.merges)
            agg["oversplits"] += int(m.oversplits)
            agg["gt_calls"] += int(m.gt_calls)
            agg["pred_calls"] += int(m.pred_calls)
            agg["matched_calls"] += int(m.matched_calls)
            agg["kept_calls_coverage"] += int(getattr(m, "kept_calls_coverage", 0))
            agg["kept_calls_iou_0_5"] += int(m.kept_calls_iou_0_5)
            agg["kept_calls_iou_0_8"] += int(m.kept_calls_iou_0_8)
            agg["false_positive_segments"] += int(m.false_positive_segments)
            if int(m.gt_calls) == 0:
                agg["false_positive_segments_no_call_videos"] += int(m.false_positive_segments)
            else:
                agg["false_positive_segments_call_outside_gt"] += int(m.false_positive_segments)

            if m.mean_start_abs_err_s is not None and int(m.matched_calls) > 0:
                start_err_sum += float(m.mean_start_abs_err_s) * float(m.matched_calls)
            if m.mean_end_abs_err_s is not None and int(m.matched_calls) > 0:
                end_err_sum += float(m.mean_end_abs_err_s) * float(m.matched_calls)
            if m.mean_iou is not None and int(m.matched_calls) > 0:
                iou_sum += float(m.mean_iou) * float(m.matched_calls)
            if getattr(m, "mean_coverage_exact", None) is not None and int(m.matched_calls) > 0:
                cov_sum += float(m.mean_coverage_exact) * float(m.matched_calls)

            per_video.append(
                {
                    "video_id": vid,
                    "gt_calls": m.gt_calls,
                    "pred_calls": m.pred_calls,
                    "matched_calls": m.matched_calls,
                    "kept_calls_coverage": int(getattr(m, "kept_calls_coverage", 0)),
                    "kept_calls_iou_0_5": m.kept_calls_iou_0_5,
                    "kept_calls_iou_0_8": m.kept_calls_iou_0_8,
                    "merges": m.merges,
                    "oversplits": m.oversplits,
                    "keep_rate": m.keep_rate,
                    "keep_rate_coverage": float(getattr(m, "keep_rate_coverage", 0.0)),
                    "keep_rate_iou_0_5": m.keep_rate_iou_0_5,
                    "keep_rate_iou_0_8": m.keep_rate_iou_0_8,
                    "false_positive_segments": m.false_positive_segments,
                    "false_positive_segments_no_call_videos": int(m.false_positive_segments) if int(m.gt_calls) == 0 else 0,
                    "false_positive_segments_call_outside_gt": int(m.false_positive_segments) if int(m.gt_calls) > 0 else 0,
                    "mean_start_abs_err_s": m.mean_start_abs_err_s,
                    "mean_end_abs_err_s": m.mean_end_abs_err_s,
                    "mean_iou": m.mean_iou,
                    "mean_coverage_exact": getattr(m, "mean_coverage_exact", None),
                }
            )

        keep_rate = (agg["matched_calls"] / agg["gt_calls"]) if agg["gt_calls"] > 0 else 0.0
        keep_rate_cov = (agg["kept_calls_coverage"] / agg["gt_calls"]) if agg["gt_calls"] > 0 else 0.0
        keep_rate_iou_0_5 = (agg["kept_calls_iou_0_5"] / agg["gt_calls"]) if agg["gt_calls"] > 0 else 0.0
        keep_rate_iou_0_8 = (agg["kept_calls_iou_0_8"] / agg["gt_calls"]) if agg["gt_calls"] > 0 else 0.0
        mean_start_abs_err_s = (start_err_sum / float(agg["matched_calls"])) if int(agg["matched_calls"]) > 0 else None
        mean_end_abs_err_s = (end_err_sum / float(agg["matched_calls"])) if int(agg["matched_calls"]) > 0 else None
        mean_iou = (iou_sum / float(agg["matched_calls"])) if int(agg["matched_calls"]) > 0 else None
        mean_coverage_exact = (cov_sum / float(agg["matched_calls"])) if int(agg["matched_calls"]) > 0 else None

        return {
            "eval_videos": len(eval_video_ids),
            "match_tol_s": float(match_tol_s),
            "overlap_eps_s": float(args.overlap_eps_s),
            "min_coverage": float(args.min_coverage),
            "keep_rate": float(keep_rate),
            "keep_rate_coverage": float(keep_rate_cov),
            "keep_rate_iou_0_5": float(keep_rate_iou_0_5),
            "keep_rate_iou_0_8": float(keep_rate_iou_0_8),
            **agg,
            "mean_start_abs_err_s": mean_start_abs_err_s,
            "mean_end_abs_err_s": mean_end_abs_err_s,
            "mean_iou": mean_iou,
            "mean_coverage_exact": mean_coverage_exact,
            "per_video": sorted(per_video, key=lambda r: (r["merges"], r["oversplits"], -r["keep_rate"]), reverse=True),
        }

    tol_sel = float(args.match_tol_s)
    report_sel = build_report(match_tol_s=tol_sel)
    report_0 = build_report(match_tol_s=0.0)

    report = {
        "metrics_version": str(METRICS_VERSION),
        "gate_policy_version": str(GATE_POLICY_VERSION),
        "split_config_path": str(split_config_path),
        "label_prefix": str(label_prefix),
        **{k: v for k, v in report_sel.items() if k not in {"per_video"}},
        "per_video": report_sel["per_video"],
        "report_tol_0_0": report_0,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    md_lines = [
        "# Call extractor eval report",
        "",
        "## Summary (selection tol)",
        f"- metrics_version: {report['metrics_version']}",
        f"- gate_policy_version: {report['gate_policy_version']}",
        f"- split_config_path: {report['split_config_path']}",
        f"- label_prefix: {report['label_prefix']}",
        f"- match_tol_s: {report_sel['match_tol_s']}",
        f"- overlap_eps_s: {report_sel['overlap_eps_s']}",
        f"- min_coverage: {report_sel['min_coverage']}",
        f"- eval_videos: {report_sel['eval_videos']}",
        f"- keep_rate: {report_sel['keep_rate']:.3f}",
        f"- keep_rate_coverage: {report_sel['keep_rate_coverage']:.3f}",
        f"- merges: {report_sel['merges']}",
        f"- oversplits: {report_sel['oversplits']}",
        f"- gt_calls: {report_sel['gt_calls']}",
        f"- pred_calls: {report_sel['pred_calls']}",
        f"- false_positive_segments: {report_sel['false_positive_segments']}",
        f"- false_positive_segments_no_call_videos: {report_sel['false_positive_segments_no_call_videos']}",
        f"- false_positive_segments_call_outside_gt: {report_sel['false_positive_segments_call_outside_gt']}",
        f"- keep_rate_iou_0_5: {report_sel['keep_rate_iou_0_5']:.3f}",
        f"- keep_rate_iou_0_8: {report_sel['keep_rate_iou_0_8']:.3f}",
        f"- mean_start_abs_err_s: {report_sel['mean_start_abs_err_s']}",
        f"- mean_end_abs_err_s: {report_sel['mean_end_abs_err_s']}",
        f"- mean_iou: {report_sel['mean_iou']}",
        f"- mean_coverage_exact: {report_sel['mean_coverage_exact']}",
        "",
        "## Summary (tol=0.0)",
        f"- keep_rate: {report_0['keep_rate']:.3f}",
        f"- keep_rate_coverage: {report_0['keep_rate_coverage']:.3f}",
        f"- merges: {report_0['merges']}",
        f"- oversplits: {report_0['oversplits']}",
        f"- false_positive_segments: {report_0['false_positive_segments']}",
        f"- keep_rate_iou_0_5: {report_0['keep_rate_iou_0_5']:.3f}",
        f"- keep_rate_iou_0_8: {report_0['keep_rate_iou_0_8']:.3f}",
        f"- mean_start_abs_err_s: {report_0['mean_start_abs_err_s']}",
        f"- mean_end_abs_err_s: {report_0['mean_end_abs_err_s']}",
        f"- mean_iou: {report_0['mean_iou']}",
        f"- mean_coverage_exact: {report_0['mean_coverage_exact']}",
        "",
        "## Per-video",
        "| video_id | gt | pred | matched | kept_cov | kept@iou0.5 | keep_rate | keep@cov | keep@iou0.5 | merges | oversplits | fp | fp(no_call) | fp(call_outside) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in report["per_video"]:
        md_lines.append(
            f"| {r['video_id']} | {r['gt_calls']} | {r['pred_calls']} | {r['matched_calls']} | {r['kept_calls_coverage']} | {r['kept_calls_iou_0_5']} | {r['keep_rate']:.3f} | {r['keep_rate_coverage']:.3f} | {r['keep_rate_iou_0_5']:.3f} | {r['merges']} | {r['oversplits']} | {r['false_positive_segments']} | {r['false_positive_segments_no_call_videos']} | {r['false_positive_segments_call_outside_gt']} |"
        )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md_lines) + "\n")
    logger.info(f"Wrote {args.out_json} and {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
