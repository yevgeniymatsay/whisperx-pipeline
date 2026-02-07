#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import AWS_REGION, S3_BUCKET
from pipeline.call_extractor_wavlm.io import s3_download_if_missing, s3_read_json
from pipeline.call_extractor_wavlm.labels import parse_video_labels
from pipeline.call_extractor_wavlm.metrics import boundaries_from_json_segments, compute_gate_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate predicted call segments against boundary labels.")
    parser.add_argument("--split-config", type=Path, default=Path("configs/call_extractor/split_v1.config.json"))
    parser.add_argument("--output-config", type=Path, default=Path("configs/call_extractor/output_wavlm_large_v1.config.json"))
    parser.add_argument(
        "--segments-dir",
        type=Path,
        default=None,
        help="Local directory containing {video_id}.json (default: <local_artifacts_dir>/predictions)",
    )
    parser.add_argument("--out-json", type=Path, default=Path("artifacts/call_extractor/wavlm_large_v1/eval_report.json"))
    parser.add_argument("--out-md", type=Path, default=Path("artifacts/call_extractor/wavlm_large_v1/eval_report.md"))
    args = parser.parse_args()

    split_cfg = _load_json(args.split_config)
    out_cfg = _load_json(args.output_config)
    s3_prefix = str(out_cfg.get("s3_output_prefix", "call_extractor/wavlm_large_v1/")).rstrip("/")

    segments_dir = args.segments_dir
    if segments_dir is None:
        segments_dir = Path(out_cfg.get("local_artifacts_dir", "artifacts/call_extractor/wavlm_large_v1")) / "predictions"
    segments_dir.mkdir(parents=True, exist_ok=True)

    eval_video_ids = list(split_cfg["eval_video_ids"])
    label_prefix = str(split_cfg["label_prefix"])

    per_video: list[dict] = []
    agg = {
        "merges": 0,
        "oversplits": 0,
        "gt_calls": 0,
        "pred_calls": 0,
        "matched_calls": 0,
        "false_positive_segments": 0,
    }

    for vid in eval_video_ids:
        vid = str(vid)
        label_key = f"{label_prefix}{vid}.json"
        labels = parse_video_labels(s3_read_json(S3_BUCKET, label_key, region=AWS_REGION))

        local_seg = segments_dir / f\"{vid}.json\"
        if not local_seg.exists():
            s3_key = f\"{s3_prefix}/segments/{vid}.json\"
            s3_download_if_missing(S3_BUCKET, s3_key, local_seg, region=AWS_REGION)
        pred_json = json.loads(local_seg.read_text())
        pred_bounds = boundaries_from_json_segments(pred_json.get(\"segments\", []))

        m = compute_gate_metrics(gt=labels.boundaries, pred=pred_bounds)
        agg[\"merges\"] += int(m.merges)
        agg[\"oversplits\"] += int(m.oversplits)
        agg[\"gt_calls\"] += int(m.gt_calls)
        agg[\"pred_calls\"] += int(m.pred_calls)
        agg[\"matched_calls\"] += int(m.matched_calls)
        agg[\"false_positive_segments\"] += int(m.false_positive_segments)

        per_video.append(
            {
                \"video_id\": vid,
                \"gt_calls\": m.gt_calls,
                \"pred_calls\": m.pred_calls,
                \"matched_calls\": m.matched_calls,
                \"merges\": m.merges,
                \"oversplits\": m.oversplits,
                \"keep_rate\": m.keep_rate,
                \"false_positive_segments\": m.false_positive_segments,
                \"mean_start_abs_err_s\": m.mean_start_abs_err_s,
                \"mean_end_abs_err_s\": m.mean_end_abs_err_s,
            }
        )

    keep_rate = (agg[\"matched_calls\"] / agg[\"gt_calls\"]) if agg[\"gt_calls\"] > 0 else 0.0

    report = {
        \"eval_videos\": len(eval_video_ids),
        \"keep_rate\": float(keep_rate),
        **agg,
        \"per_video\": sorted(per_video, key=lambda r: (r[\"merges\"], r[\"oversplits\"], -r[\"keep_rate\"]), reverse=True),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + \"\\n\")

    md_lines = [
        \"# Call extractor eval report\",
        \"\",
        f\"- eval_videos: {report['eval_videos']}\",
        f\"- keep_rate: {report['keep_rate']:.3f}\",
        f\"- merges: {report['merges']}\",
        f\"- oversplits: {report['oversplits']}\",
        f\"- gt_calls: {report['gt_calls']}\",
        f\"- pred_calls: {report['pred_calls']}\",
        f\"- false_positive_segments: {report['false_positive_segments']}\",
        \"\",
        \"## Per-video\",
        \"| video_id | gt | pred | matched | keep_rate | merges | oversplits | fp |\" ,
        \"|---|---:|---:|---:|---:|---:|---:|---:|\",
    ]
    for r in report[\"per_video\"]:
        md_lines.append(
            f\"| {r['video_id']} | {r['gt_calls']} | {r['pred_calls']} | {r['matched_calls']} | {r['keep_rate']:.3f} | {r['merges']} | {r['oversplits']} | {r['false_positive_segments']} |\"
        )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(\"\\n\".join(md_lines) + \"\\n\")
    logger.info(f\"Wrote {args.out_json} and {args.out_md}\")
    return 0


if __name__ == \"__main__\":
    raise SystemExit(main())

