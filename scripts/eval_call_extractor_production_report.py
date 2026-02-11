#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

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


def _load_segments_file(local_path: Path) -> dict[str, Any]:
    data = json.loads(local_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Invalid segments file (expected object): {local_path}")
    return data


def _intersection_tol_s(a0: float, a1: float, b0: float, b1: float, tol_s: float) -> float:
    tol = float(tol_s)
    s = max(float(a0), float(b0) - tol)
    e = min(float(a1), float(b1) + tol)
    return max(0.0, e - s)


def _median_dt_s(times_s) -> float:
    import numpy as np

    times_s = np.asarray(times_s, dtype=np.float64)
    if times_s.size < 2:
        return 0.0
    d = np.diff(times_s)
    d = d[d > 0]
    if d.size == 0:
        return 0.0
    return float(np.median(d))


def _smooth_in_call(*, times_s, in_call_p, smooth_win_s: float):
    import numpy as np

    eps = 1e-6
    p = np.clip(np.asarray(in_call_p, dtype=np.float32), eps, 1.0 - eps)
    dt_s = _median_dt_s(times_s)
    if float(smooth_win_s) <= 0.0 or dt_s <= 0.0:
        return p
    win_steps = int(max(1, round(float(smooth_win_s) / float(dt_s))))
    if win_steps <= 1:
        return p
    k = np.ones((int(win_steps),), dtype=np.float32) / float(win_steps)
    p = np.convolve(p, k, mode="same")
    return np.clip(p.astype(np.float32, copy=False), eps, 1.0 - eps)


def _load_probs_npz(
    *,
    vid: str,
    probs_s3_prefix: str,
    local_probs_dir: Path,
) -> tuple[Any, Any, Any, Any]:
    import numpy as np

    local_probs_dir.mkdir(parents=True, exist_ok=True)
    local = local_probs_dir / f"{vid}.npz"
    if not local.exists():
        s3_key = f"{str(probs_s3_prefix).rstrip('/')}/probs/{vid}.npz"
        s3_download_if_missing(S3_BUCKET, s3_key, local, region=AWS_REGION)
    arr = np.load(local)
    return arr["times_s"], arr["in_call"], arr["start"], arr["end"]


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
    parser.add_argument(
        "--probs-s3-prefix",
        type=str,
        default=None,
        help="Optional S3 prefix containing probs/{video_id}.npz for merge autopsy (defaults to run_s3_prefix read from segments JSON, if present).",
    )
    parser.add_argument(
        "--merge-autopsy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute merge autopsy details for merged predicted segments on eval (requires probs).",
    )
    parser.add_argument(
        "--autopsy-boundary-win-s",
        type=float,
        default=2.0,
        help="Window (seconds) around GT boundary for max(start_p)/max(end_p) stats.",
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
    meta_by_vid: dict[str, dict[str, Any]] = {}
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
        seg_file = _load_segments_file(local)
        meta_by_vid[vid] = {
            "decode_config": seg_file.get("decode_config"),
            "run_s3_prefix": seg_file.get("run_s3_prefix"),
        }
        seg_bounds = boundaries_from_json_segments(list(seg_file.get("segments", [])))
        segs_by_vid[vid] = [(float(b.start_s), float(b.end_s)) for b in seg_bounds]

    if missing_segments:
        logger.warning(
            "Missing segments for %d/%d videos under %s: %s",
            len(missing_segments),
            len(eval_set_vids),
            segments_s3_prefix,
            missing_segments,
        )
        eval_set_vids = [vid for vid in eval_set_vids if vid not in set(missing_segments)]

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
    call_coverages_all: list[float] = []
    calls_cov_ge_0_8 = 0
    gt_union_seconds_total = 0.0
    pred_union_seconds_total = 0.0
    union_overlap_seconds_total = 0.0
    non_call_seconds_total = 0.0
    pred_seconds_total = 0.0
    pred_segments_total = 0
    audio_seconds_total = 0.0

    fp_no_call_seconds = 0.0
    fp_no_call_segments = 0
    no_call_audio_seconds = 0.0

    # Per-video no-call breakdown
    no_call_rows: list[dict] = []

    # Merge autopsy (eval only)
    merge_autopsy_rows: list[dict[str, Any]] = []
    probs_s3_prefix = str(args.probs_s3_prefix).rstrip("/") if args.probs_s3_prefix else None
    local_probs_dir = base_local / "probs"
    missing_audio_video_ids: list[str] = []

    for vid in eval_set_vids:
        gt_bounds = labels_by_vid.get(vid, [])
        pred_bounds = segs_by_vid.get(vid, [])

        dur_s: float | None = None
        try:
            dur_s = float(audio_duration_s(vid))
        except Exception as e:
            logger.warning("Audio duration missing for %s: %s", vid, e)
            missing_audio_video_ids.append(str(vid))

        if dur_s is not None:
            audio_seconds_total += float(dur_s)

        pred_seconds = sum(max(0.0, float(e) - float(s)) for s, e in pred_bounds)
        pred_segments = int(sum(1 for s, e in pred_bounds if float(e) > float(s)))
        pred_seconds_total += float(pred_seconds)
        pred_segments_total += int(pred_segments)

        gt_union = merge_intervals(gt_bounds, join_tolerance_s=0.0)
        pred_union = merge_intervals(pred_bounds, join_tolerance_s=0.0)
        purity = purity_vs_union(pred_intervals=pred_bounds, gt_union_intervals=gt_union)
        segment_purities_all.extend(purity.segment_purities)
        non_call_seconds_total += float(purity.non_call_seconds)

        # Coverage / union metrics are computed for eval call videos only (split_v2 eval set).
        if vid in eval_video_ids and len(gt_bounds) > 0:
            covs = per_call_coverages(gt_calls=gt_bounds, pred_intervals=pred_bounds)
            call_coverages_all.extend(covs)
            calls_cov_ge_0_8 += int(sum(1 for c in covs if float(c) >= 0.8))

            gt_union_s = total_seconds(gt_union)
            pred_union_s = total_seconds(pred_union)
            ov_union = overlap_seconds_between_unions(pred_union, gt_union)
            gt_union_seconds_total += float(gt_union_s)
            pred_union_seconds_total += float(pred_union_s)
            union_overlap_seconds_total += float(ov_union)

        is_no_call = len(gt_bounds) == 0
        if is_no_call:
            fp_no_call_seconds += float(pred_seconds)
            fp_no_call_segments += int(pred_segments)
            if dur_s is not None:
                no_call_audio_seconds += float(dur_s)
            no_call_rows.append(
                {
                    "video_id": vid,
                    "audio_seconds": float(dur_s) if dur_s is not None else None,
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

            if bool(args.merge_autopsy):
                # Determine where probs live (explicit arg preferred; else infer from segments JSON metadata).
                vid_meta = meta_by_vid.get(vid, {})
                inferred = vid_meta.get("run_s3_prefix")
                ps3 = probs_s3_prefix or (str(inferred).rstrip("/") if inferred else None)
                decode_cfg = vid_meta.get("decode_config") or {}
                smooth_win_s = float(decode_cfg.get("viterbi_smooth_win_s", 0.0))

                if ps3 is None:
                    merge_autopsy_rows.append(
                        {
                            "video_id": vid,
                            "error": "missing probs_s3_prefix (pass --probs-s3-prefix or decode with decode_from_probs_prod)",
                        }
                    )
                else:
                    tol_s = float(args.match_tol_s)
                    eps_s = float(args.overlap_eps_s)
                    pred_to_gt: list[list[int]] = [[] for _ in pred_bounds]
                    for pi, (ps, pe) in enumerate(pred_bounds):
                        for gi, (gs, ge) in enumerate(gt_bounds):
                            if _intersection_tol_s(ps, pe, gs, ge, tol_s) >= eps_s:
                                pred_to_gt[int(pi)].append(int(gi))

                    merged_pred_idxs = [pi for pi, lst in enumerate(pred_to_gt) if len(lst) > 1]
                    if merged_pred_idxs:
                        try:
                            times_s, in_call_p, start_p, end_p = _load_probs_npz(
                                vid=str(vid),
                                probs_s3_prefix=str(ps3),
                                local_probs_dir=local_probs_dir,
                            )
                            p_sm = _smooth_in_call(times_s=times_s, in_call_p=in_call_p, smooth_win_s=float(smooth_win_s))
                        except Exception as e:
                            merge_autopsy_rows.append(
                                {
                                    "video_id": vid,
                                    "error": f"failed to load probs for autopsy: {e}",
                                    "probs_s3_prefix": str(ps3),
                                    "merged_pred_indices": merged_pred_idxs,
                                }
                            )
                            times_s = None
                            p_sm = None
                            start_p = None
                            end_p = None

                        for pi in merged_pred_idxs:
                            ps, pe = pred_bounds[int(pi)]
                            overlapped = [gt_bounds[int(gi)] for gi in pred_to_gt[int(pi)]]
                            overlapped_sorted = sorted(overlapped, key=lambda x: (float(x[0]), float(x[1])))

                            entry: dict[str, Any] = {
                                "video_id": vid,
                                "pred_index": int(pi),
                                "pred_start_s": float(ps),
                                "pred_end_s": float(pe),
                                "pred_duration_s": float(max(0.0, float(pe) - float(ps))),
                                "gt_overlaps": [
                                    {
                                        "start_s": float(gs),
                                        "end_s": float(ge),
                                        "duration_s": float(max(0.0, float(ge) - float(gs))),
                                    }
                                    for gs, ge in overlapped_sorted
                                ],
                                "smooth_win_s": float(smooth_win_s),
                                "probs_s3_prefix": str(ps3),
                            }

                            if times_s is not None and p_sm is not None:
                                inside = (times_s >= float(ps)) & (times_s <= float(pe))
                                entry["min_in_call_smoothed_in_pred"] = float(p_sm[inside].min()) if bool(inside.any()) else None

                                gap_rows: list[dict[str, Any]] = []
                                win = float(args.autopsy_boundary_win_s)
                                for (gs0, ge0), (gs1, ge1) in zip(overlapped_sorted, overlapped_sorted[1:]):
                                    gap_start = float(ge0)
                                    gap_end = float(gs1)
                                    gap_dur = float(max(0.0, gap_end - gap_start))

                                    gap_mask = (times_s >= gap_start) & (times_s <= gap_end) if gap_dur > 0 else None
                                    min_gap = float(p_sm[gap_mask].min()) if gap_mask is not None and bool(gap_mask.any()) else None

                                    b_t = float(ge0)  # boundary time between calls (end of earlier call)
                                    win_mask = (times_s >= (b_t - win)) & (times_s <= (b_t + win))
                                    max_start = float(start_p[win_mask].max()) if start_p is not None and bool(win_mask.any()) else None
                                    max_end = float(end_p[win_mask].max()) if end_p is not None and bool(win_mask.any()) else None

                                    gap_rows.append(
                                        {
                                            "boundary_t_s": float(b_t),
                                            "gap_start_s": float(gap_start),
                                            "gap_end_s": float(gap_end),
                                            "gap_duration_s": float(gap_dur),
                                            "min_in_call_smoothed_in_gap": min_gap,
                                            "max_start_p_win": max_start,
                                            "max_end_p_win": max_end,
                                        }
                                    )
                                entry["between_call_gaps"] = gap_rows

                            merge_autopsy_rows.append(entry)

        per_video_rows.append(
            {
                "video_id": vid,
                "is_no_call": bool(is_no_call),
                "audio_seconds": float(dur_s) if dur_s is not None else None,
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
    cov_p10, cov_median = quantiles(call_coverages_all, qs=[0.10, 0.50])
    calls_cov_ge_0_8_rate = (float(calls_cov_ge_0_8) / float(gate_agg_eval["gt_calls"])) if gate_agg_eval["gt_calls"] > 0 else 0.0
    union_recall = (float(union_overlap_seconds_total) / float(gt_union_seconds_total)) if gt_union_seconds_total > 0 else 0.0
    union_purity = (float(union_overlap_seconds_total) / float(pred_union_seconds_total)) if pred_union_seconds_total > 0 else 0.0

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
        "probs_s3_prefix": probs_s3_prefix,
        "match_tol_s": float(args.match_tol_s),
        "overlap_eps_s": float(args.overlap_eps_s),
        "min_coverage": float(args.min_coverage),
        "eval_video_ids": eval_video_ids,
        "eval_set_video_ids": eval_set_vids,
        "eval_no_call_video_ids": eval_no_call_vids,
        "train_no_call_video_ids": train_no_call_vids,
        "no_call_eval_plus_train_video_ids": no_call_vids,
        "missing_segments_video_ids": missing_segments,
        "missing_audio_video_ids": missing_audio_video_ids,
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
            "calls_coverage_ge_0_8": int(calls_cov_ge_0_8),
            "calls_coverage_ge_0_8_rate": float(calls_cov_ge_0_8_rate),
            "call_coverage_p10": cov_p10,
            "call_coverage_median": cov_median,
            "union_recall": float(union_recall),
            "union_purity": float(union_purity),
        },
        "merge_autopsy": merge_autopsy_rows,
        "per_video": sorted(per_video_rows, key=lambda r: (int(r["is_no_call"]), -float(r["pred_seconds"]))),
        "no_call_per_video": sorted(no_call_rows, key=lambda r: -float(r["pred_seconds"])),
    }

    if args.out_json is None:
        args.out_json = Path("reports") / f"prod_eval_{cache_key}.json"
    if args.out_md is None:
        args.out_md = Path("reports") / f"prod_eval_{cache_key}.md"

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    md_lines: list[str] = [
        "# Call extractor production report",
        "",
        f"- git_sha: `{git_sha}`",
        f"- split_config_path: `{args.split_config}`",
        f"- label_prefix: `{label_prefix}`",
        f"- segments_s3_prefix: `{segments_s3_prefix}`",
        f"- probs_s3_prefix: `{probs_s3_prefix}`",
        f"- match_tol_s: {args.match_tol_s}",
        f"- overlap_eps_s: {args.overlap_eps_s}",
        f"- min_coverage: {args.min_coverage}",
    ]
    if missing_segments:
        md_lines.append(f"- missing_segments_video_ids: {missing_segments}")
    if missing_audio_video_ids:
        md_lines.append(f"- missing_audio_video_ids: {missing_audio_video_ids}")

    md_lines += [
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
        f"- calls_coverage_ge_0_8_rate: {calls_cov_ge_0_8_rate:.3f}",
        f"- call_coverage_p10: {cov_p10}",
        f"- call_coverage_median: {cov_median}",
        f"- union_recall: {union_recall:.3f}",
        f"- union_purity: {union_purity:.3f}",
        "",
        "## No-call per-video breakdown",
    ]
    for r in report["no_call_per_video"]:
        a = r.get("audio_seconds")
        a_str = f"{float(a):.3f}" if isinstance(a, (int, float)) else "n/a"
        md_lines.append(
            f"- {r['video_id']}: pred_segments={r['pred_segments']} pred_seconds={r['pred_seconds']:.3f} audio_seconds={a_str}"
        )

    md_lines.append("")
    md_lines.append("## Merge autopsy (eval only)")
    merged = [r for r in merge_autopsy_rows if isinstance(r, dict) and r.get("gt_overlaps")]
    md_lines.append(f"- merged_pred_segments: {len(merged)}")
    for r in merged:
        gt_overlaps = r.get("gt_overlaps") or []
        md_lines.append(
            f"- {r['video_id']} pred=[{float(r['pred_start_s']):.2f},{float(r['pred_end_s']):.2f}] gt_calls={len(gt_overlaps)} min_in_call_smoothed={r.get('min_in_call_smoothed_in_pred')}"
        )
        for g in (r.get("between_call_gaps") or []):
            md_lines.append(
                f"  - boundary_t={g.get('boundary_t_s')} gap_dur={g.get('gap_duration_s')} min_gap_in_call={g.get('min_in_call_smoothed_in_gap')} max_start_p_win={g.get('max_start_p_win')} max_end_p_win={g.get('max_end_p_win')}"
            )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md_lines) + "\n")

    logger.info(f"Wrote {args.out_json} and {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
