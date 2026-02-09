#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import AWS_REGION, S3_BUCKET
from pipeline.call_extractor_wavlm.decode import DecodeConfig, probabilities_to_segments
from pipeline.call_extractor_wavlm.io import cache_key_for_s3_prefix, s3_download_if_missing, s3_read_json
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
    parser.add_argument("--split-config", type=Path, default=Path("configs/call_extractor/split_v2.config.json"))
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
        choices=["peaks", "viterbi", "in_call", "transitions"],
        default=None,
        help="Decode mode to sweep (default: use decode-config's mode)",
    )

    parser.add_argument(
        "--match-tol-s",
        type=float,
        default=0.25,
        help="Tolerance in seconds for overlap-based matching (selection uses this tol; tol=0.0 is always reported).",
    )
    parser.add_argument(
        "--overlap-eps-s",
        type=float,
        default=0.10,
        help="Minimum (tolerance-adjusted) intersection duration required to count as an overlap for matching.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.30,
        help="Coverage threshold for counting a GT call as kept (anti-gaming gate). Coverage uses exact intersection.",
    )

    # Peaks mode grids (WavLM frame heads can be low-amplitude early in training).
    parser.add_argument("--start-thresholds", type=str, default="0.05,0.07,0.09,0.11,0.13,0.15")
    parser.add_argument("--end-thresholds", type=str, default="0.05,0.07,0.09,0.11,0.13,0.15")
    parser.add_argument("--in-call-mean-min", type=str, default="0.20,0.25,0.30")

    # Viterbi mode grids.
    parser.add_argument("--viterbi-off-to-on", type=str, default="2,4,6,8")
    parser.add_argument("--viterbi-on-to-off", type=str, default="2,4,6,8")
    parser.add_argument("--viterbi-start-scale", type=str, default="0,20,50,100")
    parser.add_argument("--viterbi-end-scale", type=str, default="0,20,50,100")
    parser.add_argument("--viterbi-min-on-s", type=str, default="1.0,2.0")
    parser.add_argument("--viterbi-min-off-s", type=str, default="0.0")
    parser.add_argument("--viterbi-smooth-win-s", type=str, default="0.0,0.1,0.2")

    # In-call threshold mode grids.
    parser.add_argument("--in-call-thresholds", type=str, default="0.30,0.35,0.40,0.45,0.50,0.55,0.60")
    parser.add_argument("--in-call-min-on-s", type=str, default="1.5,2.0,3.0,4.0")
    parser.add_argument("--in-call-min-off-s", type=str, default="0.5,0.7,0.9,1.2")
    parser.add_argument("--in-call-smooth-win-s", type=str, default="0.0,0.2,0.4,0.8")
    parser.add_argument("--in-call-logit-scales", type=str, default="1")

    # Transition-filtered peaks mode grids.
    parser.add_argument("--transition-win-s", type=str, default="0.2,0.4,0.6")
    parser.add_argument("--transition-margin", type=str, default="0.0,0.01,0.02")
    parser.add_argument("--write-best", type=Path, default=None, help="Write best DecodeConfig JSON here")
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Log progress every N configs (0 disables; useful for long silent sweeps)",
    )
    args = parser.parse_args()

    split_cfg = _load_json(args.split_config)
    out_cfg = _load_json(args.output_config)
    base_decode = DecodeConfig(**_load_json(args.decode_config))

    s3_prefix = str(args.s3_prefix or out_cfg.get("s3_output_prefix", "call_extractor/wavlm_large_v1/")).rstrip("/")

    probs_dir = args.probs_dir
    if probs_dir is None:
        cache_key = cache_key_for_s3_prefix(s3_prefix)
        probs_dir = Path(out_cfg.get("local_artifacts_dir", "artifacts/call_extractor/wavlm_large_v1")) / "predictions" / cache_key
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

    ic_thr_grid = _grid(args.in_call_thresholds)
    ic_min_on_grid = _grid(args.in_call_min_on_s)
    ic_min_off_grid = _grid(args.in_call_min_off_s)
    ic_smooth_grid = _grid(args.in_call_smooth_win_s)
    ic_scale_grid = _grid(args.in_call_logit_scales)

    trans_win_grid = _grid(args.transition_win_s)
    trans_margin_grid = _grid(args.transition_margin)

    tol_sel = float(args.match_tol_s)
    tols = [0.0, tol_sel]
    tols = sorted({float(t) for t in tols})

    @dataclass
    class Agg:
        merges: int = 0
        oversplits: int = 0
        gt_calls: int = 0
        pred_calls: int = 0
        matched: int = 0
        kept_cov: int = 0
        kept_iou_0_5: int = 0
        kept_iou_0_8: int = 0
        fps: int = 0
        fps_no_call: int = 0
        fps_call_outside: int = 0
        start_err_sum: float = 0.0
        end_err_sum: float = 0.0

        def add(self, m: Any) -> None:
            self.merges += int(m.merges)
            self.oversplits += int(m.oversplits)
            self.gt_calls += int(m.gt_calls)
            self.pred_calls += int(m.pred_calls)
            self.matched += int(m.matched_calls)
            self.kept_cov += int(getattr(m, "kept_calls_coverage", 0))
            self.kept_iou_0_5 += int(m.kept_calls_iou_0_5)
            self.kept_iou_0_8 += int(m.kept_calls_iou_0_8)
            self.fps += int(m.false_positive_segments)
            if int(m.gt_calls) == 0:
                self.fps_no_call += int(m.false_positive_segments)
            else:
                self.fps_call_outside += int(m.false_positive_segments)
            if m.mean_start_abs_err_s is not None and int(m.matched_calls) > 0:
                self.start_err_sum += float(m.mean_start_abs_err_s) * float(m.matched_calls)
            if m.mean_end_abs_err_s is not None and int(m.matched_calls) > 0:
                self.end_err_sum += float(m.mean_end_abs_err_s) * float(m.matched_calls)

        def keep_rate(self) -> float:
            return (float(self.matched) / float(self.gt_calls)) if int(self.gt_calls) > 0 else 0.0

        def keep_rate_cov(self) -> float:
            return (float(self.kept_cov) / float(self.gt_calls)) if int(self.gt_calls) > 0 else 0.0

        def keep_rate_iou_0_5(self) -> float:
            return (float(self.kept_iou_0_5) / float(self.gt_calls)) if int(self.gt_calls) > 0 else 0.0

        def keep_rate_iou_0_8(self) -> float:
            return (float(self.kept_iou_0_8) / float(self.gt_calls)) if int(self.gt_calls) > 0 else 0.0

        def mean_start_err(self) -> float | None:
            return (float(self.start_err_sum) / float(self.matched)) if int(self.matched) > 0 else None

        def mean_end_err(self) -> float | None:
            return (float(self.end_err_sum) / float(self.matched)) if int(self.matched) > 0 else None

        def err_score(self) -> float:
            if int(self.matched) <= 0:
                return float("inf")
            ms = self.mean_start_err()
            me = self.mean_end_err()
            if ms is None or me is None:
                return float("inf")
            return float(ms) + float(me)

    def _fmt_row(label: str, cfg: DecodeConfig, agg_sel: Agg, agg0: Agg) -> str:
        return (
            f"{label} keep@0.5={agg_sel.keep_rate_iou_0_5():.3f} "
            f"(kept={agg_sel.kept_iou_0_5}/{agg_sel.gt_calls}; cov_keep={agg_sel.keep_rate_cov():.3f}; raw_keep={agg_sel.keep_rate():.3f}) "
            f"merges={agg_sel.merges} oversplits={agg_sel.oversplits} fp={agg_sel.fps} "
            f"fp(no_call)={agg_sel.fps_no_call} fp(call_outside)={agg_sel.fps_call_outside} "
            f"mean_start_err={agg_sel.mean_start_err()} mean_end_err={agg_sel.mean_end_err()} "
            f"|| tol0 keep@0.5={agg0.keep_rate_iou_0_5():.3f} merges={agg0.merges} oversplits={agg0.oversplits} fp={agg0.fps} "
            f"mode={cfg.mode}"
        )

    TOP_K = 10

    def _add_topk(topk: list[tuple[float, float, DecodeConfig, Agg, Agg]], cfg: DecodeConfig, agg_sel: Agg, agg0: Agg) -> None:
        # Sort key: keep_rate_iou_0_5 desc, err_score asc.
        k = float(agg_sel.keep_rate_iou_0_5())
        e = float(agg_sel.err_score())
        topk.append((k, e, cfg, agg_sel, agg0))
        topk.sort(key=lambda x: (-float(x[0]), float(x[1])))
        del topk[TOP_K:]

    best_strict_sel: tuple[float, float, DecodeConfig, Agg, Agg] | None = None
    best_strict_tol0: tuple[float, float, DecodeConfig, Agg, Agg] | None = None
    best_almost_sel: tuple[float, float, DecodeConfig, Agg, Agg] | None = None
    best_almost_tol0: tuple[float, float, DecodeConfig, Agg, Agg] | None = None

    topk_sel: list[tuple[float, float, DecodeConfig, Agg, Agg]] = []
    topk_tol0: list[tuple[float, float, DecodeConfig, Agg, Agg]] = []

    # Near-miss buckets (tracked on selection tol, but print tol0 side-by-side).
    best_m0_o0_minfp: tuple[int, float, float, DecodeConfig, Agg, Agg] | None = None  # (fp, keep, err)
    best_m0_fp0_mino: tuple[int, float, float, DecodeConfig, Agg, Agg] | None = None  # (oversplits, keep, err)
    best_o0_fp0_minm: tuple[int, float, float, DecodeConfig, Agg, Agg] | None = None  # (merges, keep, err)

    # Failure summaries (selection tol).
    fail_fp = 0
    fail_fp_no_call = 0
    fail_fp_call_outside = 0
    fail_merges = 0
    fail_oversplits = 0
    fail_zero_cov_keep = 0

    if mode == "peaks":
        grid_iter = itertools.product(start_grid, end_grid, in_call_grid)
        total = len(start_grid) * len(end_grid) * len(in_call_grid)
    elif mode == "viterbi":
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
        total = (
            len(vit_off_on_grid)
            * len(vit_on_off_grid)
            * len(vit_start_scale_grid)
            * len(vit_end_scale_grid)
            * len(vit_min_on_grid)
            * len(vit_min_off_grid)
            * len(vit_smooth_grid)
            * len(in_call_grid)
        )
    elif mode == "transitions":
        grid_iter = itertools.product(start_grid, end_grid, ic_thr_grid, ic_smooth_grid, trans_win_grid, trans_margin_grid, in_call_grid)
        total = (
            len(start_grid)
            * len(end_grid)
            * len(ic_thr_grid)
            * len(ic_smooth_grid)
            * len(trans_win_grid)
            * len(trans_margin_grid)
            * len(in_call_grid)
        )
    else:
        grid_iter = itertools.product(ic_thr_grid, ic_min_on_grid, ic_min_off_grid, ic_smooth_grid, ic_scale_grid, in_call_grid)
        total = (
            len(ic_thr_grid)
            * len(ic_min_on_grid)
            * len(ic_min_off_grid)
            * len(ic_smooth_grid)
            * len(ic_scale_grid)
            * len(in_call_grid)
        )

    total_configs = int(total)
    t0 = time.monotonic()
    for sweep_i, vals in enumerate(grid_iter, start=1):
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
        elif mode == "viterbi":
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
        elif mode == "transitions":
            s_thr, e_thr, ic_thr, ic_smooth, t_win, t_margin, ic_min = vals
            cfg = DecodeConfig(
                mode="transitions",
                start_peak_threshold=float(s_thr),
                end_peak_threshold=float(e_thr),
                in_call_mean_min=float(ic_min),
                nms_min_sep_s=float(base_decode.nms_min_sep_s),
                min_duration_s=float(base_decode.min_duration_s),
                max_duration_s=float(base_decode.max_duration_s),
                internal_peak_drop_threshold=float(base_decode.internal_peak_drop_threshold),
                boundary_join_tolerance_s=float(base_decode.boundary_join_tolerance_s),
                in_call_threshold=float(ic_thr),
                in_call_smooth_win_s=float(ic_smooth),
                transition_win_s=float(t_win),
                transition_margin=float(t_margin),
                transition_restart_on_new_start=bool(base_decode.transition_restart_on_new_start),
            )
        else:
            ic_thr, min_on_s, min_off_s, smooth_s, logit_scale, ic_min = vals
            cfg = DecodeConfig(
                mode="in_call",
                start_peak_threshold=float(base_decode.start_peak_threshold),
                end_peak_threshold=float(base_decode.end_peak_threshold),
                in_call_mean_min=float(ic_min),
                nms_min_sep_s=float(base_decode.nms_min_sep_s),
                min_duration_s=float(base_decode.min_duration_s),
                max_duration_s=float(base_decode.max_duration_s),
                internal_peak_drop_threshold=float(base_decode.internal_peak_drop_threshold),
                boundary_join_tolerance_s=float(base_decode.boundary_join_tolerance_s),
                in_call_threshold=float(ic_thr),
                in_call_min_on_s=float(min_on_s),
                in_call_min_off_s=float(min_off_s),
                in_call_smooth_win_s=float(smooth_s),
                in_call_logit_scale=float(logit_scale),
            )

        agg_by_tol: dict[float, Agg] = {float(t): Agg() for t in tols}

        for vid in eval_video_ids:
            times_s, in_call, start, end = probs_by_vid[str(vid)]
            segs = probabilities_to_segments(times_s=times_s, in_call_p=in_call, start_p=start, end_p=end, cfg=cfg)
            pred_bounds = boundaries_from_json_segments(segs)
            gt_bounds = gt_by_vid[str(vid)]

            for tol in tols:
                m = compute_gate_metrics(
                    gt=gt_bounds,
                    pred=pred_bounds,
                    match_tol_s=float(tol),
                    overlap_eps_s=float(args.overlap_eps_s),
                    min_coverage=float(args.min_coverage),
                )
                agg_by_tol[float(tol)].add(m)

        agg_sel = agg_by_tol[float(tol_sel)]
        agg0 = agg_by_tol[0.0]

        # Track top-K regardless of gates (selection tol).
        _add_topk(topk_sel, cfg, agg_sel, agg0)
        _add_topk(topk_tol0, cfg, agg0, agg0)  # store tol0 metrics in the "sel" slot for printing

        # Failure summaries (selection tol).
        if int(agg_sel.fps) > 0:
            fail_fp += 1
        if int(agg_sel.fps_no_call) > 0:
            fail_fp_no_call += 1
        if int(agg_sel.fps_call_outside) > 0:
            fail_fp_call_outside += 1
        if int(agg_sel.merges) > 0:
            fail_merges += 1
        if int(agg_sel.oversplits) > 0:
            fail_oversplits += 1
        if int(agg_sel.merges) == 0 and int(agg_sel.oversplits) == 0 and int(agg_sel.fps) == 0:
            # Safety passes but nothing meets the coverage gate.
            if int(agg_sel.matched) > 0 and int(agg_sel.kept_cov) == 0:
                fail_zero_cov_keep += 1

        # Strict-valid and strict-almost (selection tol).
        strict_sel_ok = (int(agg_sel.merges) == 0) and (int(agg_sel.oversplits) == 0) and (int(agg_sel.fps) == 0)
        strict0_ok = (int(agg0.merges) == 0) and (int(agg0.oversplits) == 0) and (int(agg0.fps) == 0)

        almost_sel_ok = (int(agg_sel.merges) == 0) and (int(agg_sel.fps) == 0) and (int(agg_sel.oversplits) <= 1)
        almost0_ok = (int(agg0.merges) == 0) and (int(agg0.fps) == 0) and (int(agg0.oversplits) <= 1)

        if strict_sel_ok:
            k = float(agg_sel.keep_rate_iou_0_5())
            e = float(agg_sel.err_score())
            cand = (k, e, cfg, agg_sel, agg0)
            if best_strict_sel is None or (k > float(best_strict_sel[0])) or (k == float(best_strict_sel[0]) and e < float(best_strict_sel[1])):
                best_strict_sel = cand

        if strict0_ok:
            k = float(agg0.keep_rate_iou_0_5())
            e = float(agg0.err_score())
            cand = (k, e, cfg, agg0, agg0)
            if best_strict_tol0 is None or (k > float(best_strict_tol0[0])) or (k == float(best_strict_tol0[0]) and e < float(best_strict_tol0[1])):
                best_strict_tol0 = cand

        if almost_sel_ok:
            k = float(agg_sel.keep_rate_iou_0_5())
            e = float(agg_sel.err_score())
            cand = (k, e, cfg, agg_sel, agg0)
            if best_almost_sel is None or (k > float(best_almost_sel[0])) or (k == float(best_almost_sel[0]) and e < float(best_almost_sel[1])):
                best_almost_sel = cand

        if almost0_ok:
            k = float(agg0.keep_rate_iou_0_5())
            e = float(agg0.err_score())
            cand = (k, e, cfg, agg0, agg0)
            if best_almost_tol0 is None or (k > float(best_almost_tol0[0])) or (k == float(best_almost_tol0[0]) and e < float(best_almost_tol0[1])):
                best_almost_tol0 = cand

        # Near-miss buckets on selection tol.
        if int(agg_sel.merges) == 0 and int(agg_sel.oversplits) == 0:
            fp = int(agg_sel.fps)
            keep = float(agg_sel.keep_rate_iou_0_5())
            err = float(agg_sel.err_score())
            cand = (fp, keep, err, cfg, agg_sel, agg0)
            if best_m0_o0_minfp is None or (fp < int(best_m0_o0_minfp[0])) or (fp == int(best_m0_o0_minfp[0]) and keep > float(best_m0_o0_minfp[1])) or (
                fp == int(best_m0_o0_minfp[0]) and keep == float(best_m0_o0_minfp[1]) and err < float(best_m0_o0_minfp[2])
            ):
                best_m0_o0_minfp = cand

        if int(agg_sel.merges) == 0 and int(agg_sel.fps) == 0:
            o = int(agg_sel.oversplits)
            keep = float(agg_sel.keep_rate_iou_0_5())
            err = float(agg_sel.err_score())
            cand = (o, keep, err, cfg, agg_sel, agg0)
            if best_m0_fp0_mino is None or (o < int(best_m0_fp0_mino[0])) or (o == int(best_m0_fp0_mino[0]) and keep > float(best_m0_fp0_mino[1])) or (
                o == int(best_m0_fp0_mino[0]) and keep == float(best_m0_fp0_mino[1]) and err < float(best_m0_fp0_mino[2])
            ):
                best_m0_fp0_mino = cand

        if int(agg_sel.oversplits) == 0 and int(agg_sel.fps) == 0:
            m = int(agg_sel.merges)
            keep = float(agg_sel.keep_rate_iou_0_5())
            err = float(agg_sel.err_score())
            cand = (m, keep, err, cfg, agg_sel, agg0)
            if best_o0_fp0_minm is None or (m < int(best_o0_fp0_minm[0])) or (m == int(best_o0_fp0_minm[0]) and keep > float(best_o0_fp0_minm[1])) or (
                m == int(best_o0_fp0_minm[0]) and keep == float(best_o0_fp0_minm[1]) and err < float(best_o0_fp0_minm[2])
            ):
                best_o0_fp0_minm = cand

        log_every = int(args.log_every)
        if log_every > 0 and (sweep_i % log_every == 0 or sweep_i == total):
            elapsed_s = float(time.monotonic() - t0)
            best_keep = float(best_strict_sel[0]) if best_strict_sel is not None else 0.0
            logger.info(
                f"Progress {sweep_i}/{total} configs; best_strict_keep@0.5={best_keep:.3f}; "
                f"tol_sel={tol_sel} eps={args.overlap_eps_s} cov={args.min_coverage}; elapsed_s={elapsed_s:.1f}"
            )

    # NOTE: The sweep loops are intentionally silent by default (fast), but can be hard to
    # distinguish from a hang. When --log-every is set, re-run with unbuffered output:
    #   PYTHONUNBUFFERED=1 python -u scripts/sweep_call_extractor_decode.py ... --log-every 100

    logger.info(
        f"Sweep complete: mode={mode} tol_sel={tol_sel} eps={args.overlap_eps_s} min_coverage={args.min_coverage} total_configs={total_configs}"
    )

    logger.info("Top-K overall by keep@0.5 (selection tol):")
    for i, (k, e, cfg, agg_sel, agg0) in enumerate(topk_sel, start=1):
        logger.info(_fmt_row(f"  [{i:02d}]", cfg, agg_sel, agg0))

    logger.info("Top-K overall by keep@0.5 (tol=0.0):")
    for i, (k, e, cfg, agg_sel, agg0) in enumerate(topk_tol0, start=1):
        # Here agg_sel==agg0 by construction.
        logger.info(_fmt_row(f"  [{i:02d}]", cfg, agg_sel, agg0))

    if best_strict_sel is not None:
        _k, _e, best_cfg, agg_sel, agg0 = best_strict_sel
        logger.info("Best strict-valid @ tol_sel:")
        logger.info(_fmt_row("  [BEST_STRICT_SEL]", best_cfg, agg_sel, agg0))
    else:
        logger.error("No strict-valid config found @ tol_sel.")

    if best_strict_tol0 is not None:
        _k, _e, best_cfg, agg_sel, agg0 = best_strict_tol0
        logger.info("Best strict-valid @ tol=0.0:")
        logger.info(_fmt_row("  [BEST_STRICT_TOL0]", best_cfg, agg_sel, agg0))
    else:
        logger.info("No strict-valid config found @ tol=0.0.")

    if best_almost_sel is not None:
        _k, _e, best_cfg, agg_sel, agg0 = best_almost_sel
        logger.info("Best strict-almost @ tol_sel (reporting-only):")
        logger.info(_fmt_row("  [BEST_ALMOST_SEL]", best_cfg, agg_sel, agg0))

    if best_almost_tol0 is not None:
        _k, _e, best_cfg, agg_sel, agg0 = best_almost_tol0
        logger.info("Best strict-almost @ tol=0.0 (reporting-only):")
        logger.info(_fmt_row("  [BEST_ALMOST_TOL0]", best_cfg, agg_sel, agg0))

    if best_m0_o0_minfp is not None:
        fp, keep, err, best_cfg, agg_sel, agg0 = best_m0_o0_minfp
        logger.info(f"Near-miss (merges=0, oversplits=0; minimize FP): fp={fp}")
        logger.info(_fmt_row("  [NEAR_M0_O0_MINFP]", best_cfg, agg_sel, agg0))

    if best_m0_fp0_mino is not None:
        o, keep, err, best_cfg, agg_sel, agg0 = best_m0_fp0_mino
        logger.info(f"Near-miss (merges=0, FP=0; minimize oversplits): oversplits={o}")
        logger.info(_fmt_row("  [NEAR_M0_FP0_MINO]", best_cfg, agg_sel, agg0))

    if best_o0_fp0_minm is not None:
        m, keep, err, best_cfg, agg_sel, agg0 = best_o0_fp0_minm
        logger.info(f"Near-miss (oversplits=0, FP=0; minimize merges): merges={m}")
        logger.info(_fmt_row("  [NEAR_O0_FP0_MINM]", best_cfg, agg_sel, agg0))

    logger.info(
        "Failure summary (@ tol_sel): "
        f"fp_fail={fail_fp}/{total_configs} (no_call={fail_fp_no_call}, call_outside={fail_fp_call_outside}) "
        f"merges_fail={fail_merges}/{total_configs} oversplits_fail={fail_oversplits}/{total_configs} "
        f"safety_pass_but_zero_cov_keep={fail_zero_cov_keep}/{total_configs}"
    )

    if best_strict_sel is None:
        return 2

    if args.write_best:
        args.write_best.parent.mkdir(parents=True, exist_ok=True)
        best_cfg: DecodeConfig = best_strict_sel[2]
        args.write_best.write_text(json.dumps(best_cfg.__dict__, indent=2, sort_keys=True) + "\n")
        logger.info(f"Wrote {args.write_best}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
