#!/usr/bin/env python3
"""Generate a side-by-side report: truth call boundaries vs predicted segments.

Outputs:
- Markdown report (human-readable)
- JSON report (machine-readable)

By default, writes under: artifacts/s3_audit/call_segmenter/<report_name>_<timestamp>/
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import boto3

# Add project root to path for imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import S3_BUCKET, AWS_REGION
from eval_call_segmenter_predictions import (
    Segment,
    compute_time_metrics,
    compute_boundary_metrics,
)


def get_s3_client():
    return boto3.client("s3", region_name=AWS_REGION)


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    rest = uri[len("s3://") :]
    parts = rest.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid s3 uri: {uri}")
    bucket = parts[0]
    key = parts[1].lstrip("/")
    return bucket, key


def ensure_prefix(prefix: str) -> str:
    return prefix if prefix.endswith("/") else prefix + "/"


def sanitize_name(s: str) -> str:
    s = s.strip().strip("/")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:120] or "report"


def load_video_ids_from_file(path: Path) -> List[str]:
    ids: List[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line)
    return ids


def load_video_ids_from_split_meta(path: Path) -> List[str]:
    meta = json.loads(path.read_text())
    train = [str(v) for v in meta.get("train_video_ids", [])]
    eval_ = [str(v) for v in meta.get("eval_video_ids", [])]
    return train + eval_


def list_json_keys(s3_client, bucket: str, prefix: str) -> List[str]:
    keys: List[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.endswith(".json"):
                keys.append(k)
    return keys


def load_labels_for_video(s3_client, bucket: str, labels_prefix: str, video_id: str) -> Optional[List[Segment]]:
    key = f"{labels_prefix}{video_id}.json"
    try:
        resp = s3_client.get_object(Bucket=bucket, Key=key)
        doc = json.loads(resp["Body"].read())
        segs = [Segment(start_s=float(b["start_s"]), end_s=float(b["end_s"])) for b in doc.get("boundaries", [])]
        segs.sort(key=lambda s: s.start_s)
        return segs
    except Exception:
        return None


def load_predictions_for_video(
    s3_client, bucket: str, preds_prefix: str, video_id: str
) -> Optional[Tuple[List[Tuple[Segment, Optional[float]]], Dict]]:
    key = f"{preds_prefix}{video_id}.json"
    try:
        resp = s3_client.get_object(Bucket=bucket, Key=key)
        doc = json.loads(resp["Body"].read())
        out: List[Tuple[Segment, Optional[float]]] = []
        for s in doc.get("segments", []):
            seg = Segment(start_s=float(s["start_s"]), end_s=float(s["end_s"]))
            mean_p = s.get("mean_p")
            out.append((seg, float(mean_p) if mean_p is not None else None))
        out.sort(key=lambda t: t[0].start_s)
        header_keys = [
            "video_id",
            "run_id",
            "model_git_sha",
            "predicted_at",
            "calibration",
            "calibration_applied",
            "decode_mode",
            "enter_cost",
            "exit_cost",
            "threshold",
            "threshold_off",
            "gap_merge_s",
            "gap_merge_min_p",
            "gap_merge_stat",
            "min_seg_s",
            "min_seg_short_s",
            "keep_short_p",
            "win_s",
            "hop_s",
        ]
        header = {k: doc.get(k) for k in header_keys if k in doc}
        return out, header
    except Exception:
        return None


def format_seg(seg: Segment) -> str:
    return f"{seg.start_s:.3f} \u2192 {seg.end_s:.3f}"


def format_pred(seg: Segment, mean_p: Optional[float]) -> str:
    if mean_p is None:
        return f"{seg.start_s:.3f} \u2192 {seg.end_s:.3f}"
    return f"{seg.start_s:.3f} \u2192 {seg.end_s:.3f} (p={mean_p:.4f})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate side-by-side call boundary report")
    parser.add_argument("--pred-prefix", type=str, required=True, help="Predictions prefix (s3://bucket/prefix/)")
    parser.add_argument(
        "--labels-prefix",
        type=str,
        default=f"s3://{S3_BUCKET}/labeling/corrected_boundaries/v1/",
        help="Labels prefix (s3://bucket/prefix/)",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("artifacts/s3_audit/call_segmenter"),
        help="Root directory for output",
    )
    parser.add_argument("--report-name", type=str, default=None, help="Optional name for the report directory")
    parser.add_argument("--video-list", type=Path, default=None, help="Optional file of video_ids to include")
    parser.add_argument("--split-meta", type=Path, default=None, help="Optional split meta JSON to choose videos")
    parser.add_argument(
        "--s3-out-prefix",
        type=str,
        default=None,
        help="Optional S3 prefix (within bucket) to upload report files",
    )
    args = parser.parse_args()

    pred_bucket, pred_key_prefix = parse_s3_uri(args.pred_prefix)
    label_bucket, label_key_prefix = parse_s3_uri(args.labels_prefix)
    pred_key_prefix = ensure_prefix(pred_key_prefix)
    label_key_prefix = ensure_prefix(label_key_prefix)

    s3 = get_s3_client()

    # Determine which videos to include.
    if args.video_list is not None:
        video_ids = load_video_ids_from_file(args.video_list)
    elif args.split_meta is not None:
        video_ids = load_video_ids_from_split_meta(args.split_meta)
    else:
        # Default: include all videos that have labels under the prefix.
        label_keys = list_json_keys(s3, label_bucket, label_key_prefix)
        video_ids = sorted([Path(k).stem for k in label_keys])

    # Load all rows
    rows: List[Dict] = []
    missing_labels: List[str] = []
    missing_preds: List[str] = []
    pred_meta: Optional[Dict] = None

    for vid in video_ids:
        truth = load_labels_for_video(s3, label_bucket, label_key_prefix, vid)
        if truth is None:
            missing_labels.append(vid)
            truth = []

        preds_pack = load_predictions_for_video(s3, pred_bucket, pred_key_prefix, vid)
        if preds_pack is None:
            missing_preds.append(vid)
            preds = []
            header = {}
        else:
            preds, header = preds_pack
            if pred_meta is None and header:
                pred_meta = header

        pred_segs = [p[0] for p in preds]
        time_m = compute_time_metrics(pred_segs, truth)
        bound_m = compute_boundary_metrics(pred_segs, truth) if truth else None

        # Segment-level metrics (IoU-matching) from match counts.
        matched = int(bound_m.matched_pairs) if bound_m else 0
        if pred_segs:
            seg_precision = matched / len(pred_segs)
        else:
            seg_precision = 1.0 if not truth else 0.0

        if truth:
            seg_recall = matched / len(truth)
        else:
            seg_recall = 1.0 if not pred_segs else 0.0

        seg_f1 = 0.0
        if seg_precision + seg_recall > 0:
            seg_f1 = 2 * seg_precision * seg_recall / (seg_precision + seg_recall)

        row = {
            "video_id": vid,
            "truth_count": int(len(truth)),
            "pred_count": int(len(pred_segs)),
            "truth_segments": [[s.start_s, s.end_s] for s in truth],
            "pred_segments": [[p[0].start_s, p[0].end_s, p[1]] for p in preds],
            "time_f1": float(time_m.f1),
            "time_precision": float(time_m.precision),
            "time_recall": float(time_m.recall),
            "seg_precision": float(seg_precision),
            "seg_recall": float(seg_recall),
            "seg_f1": float(seg_f1),
            "matched_pairs": int(bound_m.matched_pairs) if bound_m else 0,
            "unmatched_pred": int(bound_m.unmatched_pred) if bound_m else int(len(pred_segs)) if truth else int(len(pred_segs)),
            "unmatched_truth": int(bound_m.unmatched_truth) if bound_m else int(len(truth)) if truth else 0,
            "mae_start_s": bound_m.mae_start_s if bound_m else None,
            "mae_end_s": bound_m.mae_end_s if bound_m else None,
            "mean_iou": bound_m.mean_iou if bound_m else None,
        }
        rows.append(row)

    # Sort for stable output.
    rows.sort(key=lambda r: r["video_id"])

    # Output directory
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_name = args.report_name
    if not report_name:
        report_name = sanitize_name(Path(pred_key_prefix.rstrip("/")).name)
    out_dir = args.out_root / f"{report_name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "side_by_side.md"
    json_path = out_dir / "side_by_side.json"

    # Build markdown
    md_lines: List[str] = []
    md_lines.append(f"# {report_name}: Truth vs Predicted Call Boundaries (side-by-side)")
    md_lines.append("")
    md_lines.append(f"Bucket (pred): `{pred_bucket}`")
    md_lines.append(f"Predictions: `{args.pred_prefix}`")
    md_lines.append(f"Labels: `{args.labels_prefix}`")
    if pred_meta:
        md_lines.append("")
        md_lines.append(f"Model git sha: `{pred_meta.get('model_git_sha', 'unknown')}`")
        if pred_meta.get("calibration_applied"):
            md_lines.append(f"Calibration: applied (`{pred_meta.get('calibration')}`)")
        dm = str(pred_meta.get('decode_mode', 'threshold'))
        if dm == "viterbi":
            md_lines.append(
                f"Decode: `viterbi` (enter_cost={pred_meta.get('enter_cost')}, exit_cost={pred_meta.get('exit_cost')}, "
                f"min_seg_s={pred_meta.get('min_seg_s')})"
            )
        else:
            md_lines.append(
                f"Decode: `threshold` (thr_on={pred_meta.get('threshold')}, thr_off={pred_meta.get('threshold_off')}, "
                f"gap_merge_s={pred_meta.get('gap_merge_s')}, gap_stat={pred_meta.get('gap_merge_stat')}, "
                f"gap_min_p={pred_meta.get('gap_merge_min_p')}, min_seg_s={pred_meta.get('min_seg_s')})"
            )
    md_lines.append("")
    md_lines.append("Format:")
    md_lines.append("- Truth boundaries are your manual call labels.")
    md_lines.append("- Pred boundaries are model outputs (post-processed) with mean probability `p` across the segment.")
    md_lines.append("")

    # Top mismatch summaries
    by_delta = sorted(rows, key=lambda r: abs(r["pred_count"] - r["truth_count"]), reverse=True)
    md_lines.append("## Top 10 by Absolute Call-Count Delta")
    md_lines.append("")
    md_lines.append("| video_id | truth_calls | pred_calls | delta | time_f1 | seg_f1 |")
    md_lines.append("|---|---:|---:|---:|---:|---:|")
    for r in by_delta[:10]:
        delta = int(r["pred_count"] - r["truth_count"])
        md_lines.append(
            f"| `{r['video_id']}` | {r['truth_count']} | {r['pred_count']} | {delta:+d} | {r['time_f1']:.3f} | {r['seg_f1']:.3f} |"
        )
    md_lines.append("")

    by_worst = sorted(rows, key=lambda r: r["seg_f1"])
    md_lines.append("## Top 10 Worst by Segment IoU-F1")
    md_lines.append("")
    md_lines.append("| video_id | truth_calls | pred_calls | time_f1 | seg_f1 | unmatched_pred | unmatched_truth |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in by_worst[:10]:
        md_lines.append(
            f"| `{r['video_id']}` | {r['truth_count']} | {r['pred_count']} | {r['time_f1']:.3f} | {r['seg_f1']:.3f} | {r['unmatched_pred']} | {r['unmatched_truth']} |"
        )
    md_lines.append("")

    md_lines.append("## Counts Summary (all videos)")
    md_lines.append("")
    md_lines.append("| video_id | truth_calls | pred_calls |")
    md_lines.append("|---|---:|---:|")
    for r in rows:
        md_lines.append(f"| `{r['video_id']}` | {r['truth_count']} | {r['pred_count']} |")
    md_lines.append("")

    md_lines.append("## Detailed Boundaries")
    md_lines.append("")
    for r in rows:
        vid = r["video_id"]
        md_lines.append(f"### {vid}")
        md_lines.append(
            f"Truth calls: **{r['truth_count']}**; Pred calls: **{r['pred_count']}**; "
            f"time_f1={r['time_f1']:.3f}; seg_f1={r['seg_f1']:.3f}"
        )
        md_lines.append("")

        md_lines.append("Truth boundaries:")
        if r["truth_segments"]:
            for s0, s1 in r["truth_segments"]:
                md_lines.append(f"- {s0:.3f} \u2192 {s1:.3f}")
        else:
            md_lines.append("- (none)")
        md_lines.append("")

        md_lines.append("Pred boundaries (mean_p):")
        if r["pred_segments"]:
            for s0, s1, p in r["pred_segments"]:
                if p is None:
                    md_lines.append(f"- {s0:.3f} \u2192 {s1:.3f}")
                else:
                    md_lines.append(f"- {s0:.3f} \u2192 {s1:.3f} (p={float(p):.4f})")
        else:
            md_lines.append("- (none)")
        md_lines.append("")

    md_path.write_text("\n".join(md_lines) + "\n")

    out_json = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bucket_pred": pred_bucket,
        "bucket_labels": label_bucket,
        "pred_prefix": args.pred_prefix,
        "label_prefix": args.labels_prefix,
        "pred_meta": pred_meta,
        "videos": rows,
        "missing_labels": missing_labels,
        "missing_predictions": missing_preds,
    }
    json_path.write_text(json.dumps(out_json, indent=2) + "\n")

    print(f"Wrote: {md_path}")
    print(f"Wrote: {json_path}")
    if missing_labels:
        print(f"Missing labels ({len(missing_labels)}): {missing_labels[:10]}")
    if missing_preds:
        print(f"Missing predictions ({len(missing_preds)}): {missing_preds[:10]}")

    # Optional upload to S3 (within the default bucket).
    if args.s3_out_prefix:
        s3_out_prefix = args.s3_out_prefix.strip("/")
        md_key = f"{s3_out_prefix}/{report_name}_{ts}/side_by_side.md"
        js_key = f"{s3_out_prefix}/{report_name}_{ts}/side_by_side.json"
        s3.put_object(Bucket=S3_BUCKET, Key=md_key, Body=md_path.read_bytes(), ContentType="text/markdown")
        s3.put_object(Bucket=S3_BUCKET, Key=js_key, Body=json_path.read_bytes(), ContentType="application/json")
        print(f"Uploaded: s3://{S3_BUCKET}/{md_key}")
        print(f"Uploaded: s3://{S3_BUCKET}/{js_key}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
