#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import AWS_REGION, S3_BUCKET
from pipeline.call_extractor_wavlm.decode import DecodeConfig, probabilities_to_segments
from pipeline.call_extractor_wavlm.io import s3_download_if_missing, s3_read_json
from pipeline.call_extractor_wavlm.labels import parse_video_labels
from pipeline.call_extractor_wavlm.metrics import boundaries_from_json_segments, compute_gate_metrics
from pipeline.call_extractor_wavlm.types import CallBoundary

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _grid(values: str) -> list[float]:
    return [float(x) for x in values.split(",") if x.strip() != ""]


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep decode thresholds on eval set with strict gates.")
    parser.add_argument("--split-config", type=Path, default=Path("configs/call_extractor/split_v1.config.json"))
    parser.add_argument("--output-config", type=Path, default=Path("configs/call_extractor/output_wavlm_large_v1.config.json"))
    parser.add_argument("--decode-config", type=Path, default=Path("configs/call_extractor/decode_wavlm_large_v1.config.json"))
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default=None,
        help="Override S3 prefix for downloading probs (default: output-config's s3_output_prefix)",
    )
    parser.add_argument(
        "--probs-dir",
        type=Path,
        default=None,
        help="Local dir containing {video_id}.npz (default: <local_artifacts_dir>/predictions)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["peaks", "viterbi"],
        default=None,
        help="Decode mode to sweep (default: use decode-config's mode)",
    )

    # Peaks mode grids (WavLM frame heads can be low-amplitude early in training).
    parser.add_argument("--start-thresholds", type=str, default="0.05,0.07,0.09,0.11,0.13,0.15")
    parser.add_argument("--end-thresholds", type=str, default="0.05,0.07,0.09,0.11,0.13,0.15")
    parser.add_argument("--in-call-mean-min", type=str, default="0.50,0.52,0.54,0.56,0.58,0.60")

    # Viterbi mode grids.
    parser.add_argument("--viterbi-off-to-on", type=str, default="2,4,6,8")
    parser.add_argument("--viterbi-on-to-off", type=str, default="2,4,6,8")
    parser.add_argument("--viterbi-start-scale", type=str, default="0,20,50,100")
    parser.add_argument("--viterbi-end-scale", type=str, default="0,20,50,100")
    parser.add_argument("--viterbi-min-on-s", type=str, default="1.0,2.0")
    parser.add_argument("--viterbi-min-off-s", type=str, default="0.0")
    parser.add_argument("--viterbi-smooth-win-s", type=str, default="0.0,0.1,0.2")
    parser.add_argument("--write-best", type=Path, default=None, help="Write best DecodeConfig JSON here")
    args = parser.parse_args()

    split_cfg = _load_json(args.split_config)
    out_cfg = _load_json(args.output_config)
    base_decode = DecodeConfig(**_load_json(args.decode_config))

    s3_prefix = str(args.s3_prefix or out_cfg.get("s3_output_prefix", "call_extractor/wavlm_large_v1/")).rstrip("/")

    probs_dir = args.probs_dir
    if probs_dir is None:
        probs_dir = Path(out_cfg.get("local_artifacts_dir", "artifacts/call_extractor/wavlm_large_v1")) / "predictions"
    probs_dir.mkdir(parents=True, exist_ok=True)

    eval_video_ids = list(split_cfg["eval_video_ids"])
    label_prefix = str(split_cfg["label_prefix"])

    gt_by_vid: dict[str, list[CallBoundary]] = {}
    for vid in eval_video_ids:
        key = f"{label_prefix}{vid}.json"
        data = s3_read_json(S3_BUCKET, key, region=AWS_REGION)
        gt_by_vid[str(vid)] = parse_video_labels(data).boundaries

    def load_probs(vid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        local = probs_dir / f"{vid}.npz"
        if not local.exists():
            s3_key = f"{s3_prefix}/probs/{vid}.npz"
            s3_download_if_missing(S3_BUCKET, s3_key, local, region=AWS_REGION)
        arr = np.load(local)
        return (
            arr["times_s"].astype(np.float32, copy=False),
            arr["in_call"].astype(np.float32, copy=False),
            arr["start"].astype(np.float32, copy=False),
            arr["end"].astype(np.float32, copy=False),
        )

    # Pre-load all probs once to avoid re-reading the same npz for every candidate config.
    probs_by_vid: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for vid in eval_video_ids:
        probs_by_vid[str(vid)] = load_probs(str(vid))

    mode = (args.mode or getattr(base_decode, "mode", None) or "peaks").lower()

    start_grid = _grid(args.start_thresholds)
    end_grid = _grid(args.end_thresholds)
    in_call_grid = _grid(args.in_call_mean_min)

    vit_off_on_grid = _grid(args.viterbi_off_to_on)
    vit_on_off_grid = _grid(args.viterbi_on_to_off)
    vit_start_scale_grid = _grid(args.viterbi_start_scale)
    vit_end_scale_grid = _grid(args.viterbi_end_scale)
    vit_min_on_grid = _grid(args.viterbi_min_on_s)
    vit_min_off_grid = _grid(args.viterbi_min_off_s)
    vit_smooth_grid = _grid(args.viterbi_smooth_win_s)

    best: dict | None = None

    if mode == "peaks":
        grid_iter = itertools.product(start_grid, end_grid, in_call_grid)
    else:
        grid_iter = itertools.product(
            vit_off_on_grid,
            vit_on_off_grid,
            vit_start_scale_grid,
            vit_end_scale_grid,
            vit_min_on_grid,
            vit_min_off_grid,
            vit_smooth_grid,
            in_call_grid,
        )

    for vals in grid_iter:
        if mode == "peaks":
            s_thr, e_thr, ic_min = vals
            cfg = DecodeConfig(
                mode="peaks",
                start_peak_threshold=float(s_thr),
                end_peak_threshold=float(e_thr),
                in_call_mean_min=float(ic_min),
                nms_min_sep_s=float(base_decode.nms_min_sep_s),
                min_duration_s=float(base_decode.min_duration_s),
                max_duration_s=float(base_decode.max_duration_s),
                internal_peak_drop_threshold=float(base_decode.internal_peak_drop_threshold),
                boundary_join_tolerance_s=float(base_decode.boundary_join_tolerance_s),
            )
        else:
            off_on, on_off, ss, es, min_on_s, min_off_s, smooth_s, ic_min = vals
            cfg = DecodeConfig(
                mode="viterbi",
                start_peak_threshold=float(base_decode.start_peak_threshold),
                end_peak_threshold=float(base_decode.end_peak_threshold),
                in_call_mean_min=float(ic_min),
                nms_min_sep_s=float(base_decode.nms_min_sep_s),
                min_duration_s=float(base_decode.min_duration_s),
                max_duration_s=float(base_decode.max_duration_s),
                internal_peak_drop_threshold=float(base_decode.internal_peak_drop_threshold),
                boundary_join_tolerance_s=float(base_decode.boundary_join_tolerance_s),
                viterbi_off_to_on_penalty=float(off_on),
                viterbi_on_to_off_penalty=float(on_off),
                viterbi_start_scale=float(ss),
                viterbi_end_scale=float(es),
                viterbi_min_on_s=float(min_on_s),
                viterbi_min_off_s=float(min_off_s),
                viterbi_smooth_win_s=float(smooth_s),
            )

        merges = 0
        oversplits = 0
        gt_calls = 0
        matched = 0
        pred_calls = 0
        fps = 0

        for vid in eval_video_ids:
            times_s, in_call, start, end = probs_by_vid[str(vid)]
            segs = probabilities_to_segments(times_s=times_s, in_call_p=in_call, start_p=start, end_p=end, cfg=cfg)
            pred_bounds = boundaries_from_json_segments(segs)
            gt_bounds = gt_by_vid[str(vid)]

            m = compute_gate_metrics(gt=gt_bounds, pred=pred_bounds)
            merges += int(m.merges)
            oversplits += int(m.oversplits)
            gt_calls += int(m.gt_calls)
            matched += int(m.matched_calls)
            pred_calls += int(m.pred_calls)
            fps += int(m.false_positive_segments)

        keep_rate = (matched / gt_calls) if gt_calls > 0 else 0.0
        ok = (merges == 0) and (oversplits == 0) and (fps == 0)
        if not ok:
            continue

        cand = {
            "cfg": cfg,
            "keep_rate": float(keep_rate),
            "matched": int(matched),
            "gt_calls": int(gt_calls),
            "pred_calls": int(pred_calls),
            "merges": int(merges),
            "oversplits": int(oversplits),
            "fps": int(fps),
        }
        if best is None or float(cand["keep_rate"]) > float(best["keep_rate"]):
            best = cand

    if best is None:
        logger.error("No decode config satisfied strict gates on eval set.")
        return 2

    best_cfg: DecodeConfig = best["cfg"]
    logger.info(
        f"Best keep_rate={best['keep_rate']:.3f} matched={best['matched']}/{best['gt_calls']} "
        f"mode={best_cfg.mode} "
        f"start_thr={best_cfg.start_peak_threshold} end_thr={best_cfg.end_peak_threshold} "
        f"in_call_mean_min={best_cfg.in_call_mean_min}"
    )

    if args.write_best:
        args.write_best.parent.mkdir(parents=True, exist_ok=True)
        args.write_best.write_text(json.dumps(best_cfg.__dict__, indent=2, sort_keys=True) + "\n")
        logger.info(f"Wrote {args.write_best}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
