#!/usr/bin/env python3
"""Orchestrate a call-segmenter experiment: build -> train -> sweep -> predict -> eval -> report -> log.

This is intentionally a thin wrapper around the existing scripts so the shipped
behavior matches what we evaluate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure sibling script imports work when invoked as `python scripts/...`.
import sys

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import S3_BUCKET
from pipeline.call_segmenter.calibration import PlattCalibration, load_calibration

import sweep_call_segmenter
from eval_call_segmenter_predictions import (
    get_s3_client as get_eval_s3_client,
    Segment,
    load_ground_truth_s3,
    load_predictions_s3,
    evaluate_video,
    aggregate_results,
)


def run(cmd: List[str]) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def write_video_list(video_ids: List[str]) -> Path:
    fd, path = tempfile.mkstemp(prefix="call_segmenter_videos_", suffix=".txt")
    p = Path(path)
    p.write_text("\n".join(video_ids) + "\n")
    return p


def evaluate_predictions(
    predictions_key_prefix: str,
    labels: Dict[str, List[Segment]],
    video_ids: List[str],
) -> Tuple[object, int]:
    s3 = get_eval_s3_client()
    preds = load_predictions_s3(s3, predictions_key_prefix)

    per_video = []
    no_call_fp_videos = 0
    for vid in video_ids:
        truth = labels.get(vid, [])
        pred = preds.get(vid, [])
        if len(truth) == 0 and len(pred) > 0:
            no_call_fp_videos += 1
        per_video.append(evaluate_video(vid, pred, truth))

    agg = aggregate_results(per_video)
    return agg, no_call_fp_videos


def append_experiment_log(
    log_path: Path,
    *,
    exp_name: str,
    dataset_dir: Path,
    model_dir: Path,
    split_meta: Path,
    best_params: sweep_call_segmenter.SweepResult,
    best_overall: sweep_call_segmenter.SweepResult,
    eval_result: Optional[sweep_call_segmenter.SweepResult],
    predictions_s3_prefix: str,
    full_agg: object,
    no_call_fp_videos: int,
    report_md_path: Optional[Path],
    report_json_path: Optional[Path],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    lines: List[str] = []
    lines.append(f"\n---\n\n## {ts}: {exp_name}\n")
    lines.append(f"- Dataset: `{dataset_dir}`")
    lines.append(f"- Model dir: `{model_dir}`")
    lines.append(f"- Split meta: `{split_meta}`")
    lines.append("")
    lines.append("Best sweep params (train, constrained):")
    if best_params.decode_mode == "viterbi":
        lines.append(
            f"- decode=viterbi enter_cost={float(best_params.enter_cost):.2f}, exit_cost={float(best_params.exit_cost):.2f}, "
            f"call_bias={float(best_params.call_bias or 0.0):.2f}, "
            f"min_seg={best_params.min_seg_s:.0f}"
        )
    else:
        lines.append(
            f"- decode=threshold thr_on={float(best_params.threshold):.2f}, thr_off={float(best_params.threshold_off):.2f}, "
            f"gap={float(best_params.gap_merge_s):.0f}, gap_min_p={float(best_params.gap_merge_min_p):.2f}, min_seg={best_params.min_seg_s:.0f}"
        )
    lines.append(
        f"- Train: TimeF1={best_params.f1:.4f}, SegF1={best_params.seg_f1:.4f}, SegRatio={best_params.seg_ratio:.2f}, "
        f"NoCallFPVideos={best_params.no_call_fp_videos}"
    )
    lines.append("")
    lines.append("Best sweep params (train, overall score):")
    if best_overall.decode_mode == "viterbi":
        lines.append(
            f"- decode=viterbi enter_cost={float(best_overall.enter_cost):.2f}, exit_cost={float(best_overall.exit_cost):.2f}, "
            f"call_bias={float(best_overall.call_bias or 0.0):.2f}, "
            f"min_seg={best_overall.min_seg_s:.0f}"
        )
    else:
        lines.append(
            f"- decode=threshold thr_on={float(best_overall.threshold):.2f}, thr_off={float(best_overall.threshold_off):.2f}, "
            f"gap={float(best_overall.gap_merge_s):.0f}, gap_min_p={float(best_overall.gap_merge_min_p):.2f}, min_seg={best_overall.min_seg_s:.0f}"
        )
    lines.append(
        f"- Train: TimeF1={best_overall.f1:.4f}, SegF1={best_overall.seg_f1:.4f}, SegRatio={best_overall.seg_ratio:.2f}, "
        f"NoCallFPVideos={best_overall.no_call_fp_videos}"
    )

    if eval_result is not None:
        lines.append("")
        lines.append("Holdout eval (6 videos) using best constrained params:")
        lines.append(
            f"- TimeF1={eval_result.f1:.4f}, SegF1={eval_result.seg_f1:.4f}, SegRatio={eval_result.seg_ratio:.2f}, "
            f"Pred={eval_result.total_pred}, Truth={eval_result.total_truth}"
        )

    lines.append("")
    lines.append("Full labeled-set eval (train+eval ids; no-call FP==0 required):")
    lines.append(f"- Predictions prefix: `{predictions_s3_prefix}`")
    lines.append(
        f"- TimeF1(micro)={full_agg.micro_f1:.4f}, SegF1(IoU micro)={full_agg.seg_f1:.4f}, "
        f"Pred={full_agg.total_pred_segments}, Truth={full_agg.total_truth_segments}, NoCallFPVideos={no_call_fp_videos}"
    )

    if report_md_path and report_json_path:
        lines.append("")
        lines.append("Side-by-side report:")
        lines.append(f"- `{report_md_path}`")
        lines.append(f"- `{report_json_path}`")

    log_path.write_text(log_path.read_text() + "\n".join(lines) if log_path.exists() else "\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run call segmenter experiment end-to-end")
    parser.add_argument("--exp-name", type=str, required=True, help="Experiment name (used for dirs/prefixes)")
    parser.add_argument("--split-meta", type=Path, required=True, help="Fixed split meta JSON (train/eval ids)")
    parser.add_argument("--labels-prefix", type=str, default="labeling/corrected_boundaries/v1/", help="S3 key prefix for labels")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Output dataset dir (default: data/call_segmenter/<exp-name>)")
    parser.add_argument("--model-dir", type=Path, default=None, help="Output model dir (default: data/call_segmenter/models/<exp-name>)")
    parser.add_argument("--video-list", type=Path, default=None, help="Optional video list file (one id per line). If omitted, uses split-meta ids.")
    parser.add_argument("--text-context-s", type=float, default=0.0)
    parser.add_argument("--text-max-chars", type=int, default=300)
    parser.add_argument("--neg-weight", type=float, default=5.0)
    parser.add_argument("--no-call-video-weight", type=float, default=1.0)
    parser.add_argument("--no-calibration", action="store_true",
                        help="Disable probability calibration even if model_dir/calibration.json exists")
    parser.add_argument("--decode-mode", type=str, default="threshold", choices=["threshold", "viterbi"])
    parser.add_argument("--enter-cost-values", type=str, default="0.25,0.5,0.75,1.0,1.5,2.0,3.0,4.0",
                        help="(viterbi) Comma-separated NO_CALL->CALL transition costs to sweep")
    parser.add_argument("--exit-cost-values", type=str, default="0.0,0.25,0.5,0.75,1.0,1.5,2.0",
                        help="(viterbi) Comma-separated CALL->NO_CALL transition costs to sweep")
    parser.add_argument("--call-bias-values", type=str, default="0.0,0.5,1.0,2.0,3.0,4.0",
                        help="(viterbi) Comma-separated per-step CALL bias costs to sweep (acts like a soft threshold)")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Cache dir for sweep probabilities")
    parser.add_argument("--predictions-base-prefix", type=str, default="call_segmenter/predictions",
                        help="S3 key prefix base for predictions (within bucket)")
    parser.add_argument("--log-path", type=Path, default=Path("docs/CALL_SEGMENTER_EXPERIMENT_LOG.md"))
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir or Path(f"data/call_segmenter/{args.exp_name}")
    model_dir = args.model_dir or Path(f"data/call_segmenter/models/{args.exp_name}")
    cache_dir = args.cache_dir or Path(f"data/call_segmenter/prob_cache_{args.exp_name}")

    split_meta = json.loads(args.split_meta.read_text())
    train_vids = [str(v) for v in split_meta.get("train_video_ids", [])]
    eval_vids = [str(v) for v in split_meta.get("eval_video_ids", [])]
    if not train_vids or not eval_vids:
        raise ValueError("--split-meta must contain train_video_ids and eval_video_ids")
    all_vids = train_vids + eval_vids

    video_list_path = args.video_list
    if video_list_path is None:
        video_list_path = write_video_list(all_vids)

    if not args.skip_build:
        run(
            [
                sys.executable,
                "scripts/build_call_segmenter_dataset.py",
                "--labels-s3",
                "--labels-s3-prefix",
                args.labels_prefix,
                "--output-dir",
                str(dataset_dir),
                "--video-list",
                str(video_list_path),
                "--text-context-s",
                str(args.text_context_s),
                "--text-max-chars",
                str(args.text_max_chars),
            ]
        )

    if not args.skip_train:
        run(
            [
                sys.executable,
                "scripts/train_call_segmenter.py",
                "--data",
                str(dataset_dir),
                "--split-meta",
                str(args.split_meta),
                "--neg-weight",
                str(args.neg_weight),
                "--no-call-video-weight",
                str(args.no_call_video_weight),
                "--output-dir",
                str(model_dir),
            ]
        )

    # Sweep params
    best_constrained = None
    best_overall = None
    eval_sweep = None
    if not args.skip_sweep:
        # Use the sweep script as a library so we can programmatically reuse best params.
        model, meta = sweep_call_segmenter.load_model_and_meta(model_dir)
        feature_columns = meta["feature_columns"]
        text_hashing = meta["text_hashing"]
        window_cfg_dict = meta["window_config"]

        vectorizer = sweep_call_segmenter.create_text_vectorizer(text_hashing)
        window_cfg = sweep_call_segmenter.WindowConfig(
            win_s=window_cfg_dict["win_s"],
            hop_s=window_cfg_dict["hop_s"],
            ignore_s=window_cfg_dict.get("ignore_s", 0.75),
        )

        s3 = sweep_call_segmenter.get_s3_client()
        labels = load_ground_truth_s3(s3, prefix=args.labels_prefix)

        # Optional calibration (kept consistent between sweep + prediction).
        calib: Optional[PlattCalibration] = None
        calib_path = model_dir / "calibration.json"
        if not args.no_calibration and calib_path.exists():
            try:
                calib = load_calibration(calib_path)
                print(f"Loaded calibration: {calib_path}")
            except Exception as e:
                print(f"WARNING: failed to load calibration.json (ignoring): {e}")

        # Cache probabilities (train)
        train_cache = sweep_call_segmenter.cache_all_probabilities(
            s3_client=s3,
            video_ids=train_vids,
            model=model,
            meta=meta,
            vectorizer=vectorizer,
            window_config=window_cfg,
            feature_columns=feature_columns,
            cache_dir=cache_dir,
            force_recompute=False,
        )

        train_gt = {vid: labels.get(vid, []) for vid in train_vids}

        decode_mode = str(args.decode_mode).lower().strip()
        if decode_mode == "viterbi":
            enter_vals = [float(x.strip()) for x in args.enter_cost_values.split(",") if x.strip()]
            exit_vals = [float(x.strip()) for x in args.exit_cost_values.split(",") if x.strip()]
            bias_vals = [float(x.strip()) for x in args.call_bias_values.split(",") if x.strip()]
            train_results = sweep_call_segmenter.run_parameter_sweep(
                prob_cache=train_cache,
                ground_truth=train_gt,
                win_s=float(window_cfg.win_s),
                threshold_min=0.0,
                threshold_max=0.0,
                threshold_step=1.0,
                threshold_off_delta_values=[0.0],
                gap_merge_values=[0.0],
                gap_merge_min_p_values=[0.0],
                gap_merge_stat_values=["max"],
                min_seg_values=[5, 10],
                min_seg_short_values=[None],
                keep_short_p_values=[None],
                decode_mode="viterbi",
                enter_cost_values=enter_vals,
                exit_cost_values=exit_vals,
                call_bias_values=bias_vals,
                calibration=calib,
            )
        else:
            train_results = sweep_call_segmenter.run_parameter_sweep(
                prob_cache=train_cache,
                ground_truth=train_gt,
                win_s=float(window_cfg.win_s),
                threshold_min=0.55,
                threshold_max=0.95,
                threshold_step=0.05,
                threshold_off_delta_values=[0.0, 0.10, 0.20],
                gap_merge_values=[0, 1, 2, 5, 10, 20, 30],
                gap_merge_min_p_values=[0, 0.2, 0.4, 0.6, 0.8],
                gap_merge_stat_values=["max", "mean", "p90"],
                min_seg_values=[1, 3, 5, 10],
                min_seg_short_values=[None],
                keep_short_p_values=[None],
                decode_mode="threshold",
                calibration=calib,
            )
        best_constrained, best_overall = sweep_call_segmenter.select_best_params(train_results)

        # Cache probabilities (eval) and evaluate chosen params.
        eval_cache = sweep_call_segmenter.cache_all_probabilities(
            s3_client=s3,
            video_ids=eval_vids,
            model=model,
            meta=meta,
            vectorizer=vectorizer,
            window_config=window_cfg,
            feature_columns=feature_columns,
            cache_dir=cache_dir,
            force_recompute=False,
        )
        eval_gt = {vid: labels.get(vid, []) for vid in eval_vids}
        eval_sweep = sweep_call_segmenter.evaluate_params_on_videos(
            prob_cache=eval_cache,
            ground_truth=eval_gt,
            win_s=float(window_cfg.win_s),
            decode_mode=best_constrained.decode_mode,
            threshold=best_constrained.threshold,
            threshold_off=best_constrained.threshold_off,
            gap_merge_s=best_constrained.gap_merge_s,
            gap_merge_min_p=best_constrained.gap_merge_min_p,
            gap_merge_stat=best_constrained.gap_merge_stat,
            enter_cost=best_constrained.enter_cost,
            exit_cost=best_constrained.exit_cost,
            call_bias=best_constrained.call_bias,
            min_seg_s=best_constrained.min_seg_s,
            min_seg_short_s=best_constrained.min_seg_short_s,
            keep_short_p=best_constrained.keep_short_p,
            calibration=calib,
        )

        # Print a short sweep summary for convenience.
        print("\nBEST (train, constrained):", best_constrained)
        print("BEST (train, overall):", best_overall)
        if eval_sweep:
            print("EVAL (holdout) using best constrained:", eval_sweep)

    # Predict to S3
    preds_s3_prefix = None
    report_md = None
    report_json = None
    full_agg = None
    no_call_fp_videos = -1
    if not args.skip_predict:
        if best_constrained is None:
            raise ValueError("Need sweep results to predict; rerun without --skip-sweep")

        ts = now_stamp()
        if best_constrained.decode_mode == "viterbi":
            preds_key_prefix = (
                f"{args.predictions_base_prefix}/{args.exp_name}_{ts}_"
                f"viterbi_enter{int(round(float(best_constrained.enter_cost)*100)):03d}_"
                f"exit{int(round(float(best_constrained.exit_cost)*100)):03d}_"
                f"bias{int(round(float(best_constrained.call_bias or 0.0)*100)):03d}_"
                f"min{int(round(best_constrained.min_seg_s)):02d}"
            )
        else:
            preds_key_prefix = (
                f"{args.predictions_base_prefix}/{args.exp_name}_{ts}_"
                f"thr{int(round(float(best_constrained.threshold)*100)):02d}_"
                f"off{int(round(float(best_constrained.threshold_off)*100)):02d}_"
                f"gap{int(round(float(best_constrained.gap_merge_s))):02d}_"
                f"g{best_constrained.gap_merge_stat}_"
                f"p{int(round(float(best_constrained.gap_merge_min_p)*100)):02d}_"
                f"min{int(round(best_constrained.min_seg_s)):02d}"
            )
        preds_s3_prefix = f"s3://{S3_BUCKET}/{preds_key_prefix}/"

        cmd = [
            sys.executable,
            "scripts/predict_call_segmenter.py",
            "--video-list",
            str(video_list_path),
            "--model-dir",
            str(model_dir),
            "--s3-out-prefix",
            preds_key_prefix,
            *(["--min-seg-short-s", str(best_constrained.min_seg_short_s)] if best_constrained.min_seg_short_s is not None else []),
            *(["--keep-short-p", str(best_constrained.keep_short_p)] if best_constrained.keep_short_p is not None else []),
            "--min-seg-s",
            str(best_constrained.min_seg_s),
        ]
        if args.no_calibration:
            cmd.append("--no-calibration")
        if best_constrained.decode_mode == "viterbi":
            cmd += [
                "--decode-mode",
                "viterbi",
                "--enter-cost",
                str(best_constrained.enter_cost),
                "--exit-cost",
                str(best_constrained.exit_cost),
                "--call-bias",
                str(best_constrained.call_bias or 0.0),
            ]
        else:
            cmd += [
                "--decode-mode",
                "threshold",
                "--threshold",
                str(best_constrained.threshold),
                "--threshold-off",
                str(best_constrained.threshold_off),
                "--gap-merge-s",
                str(best_constrained.gap_merge_s),
                "--gap-merge-stat",
                str(best_constrained.gap_merge_stat),
                "--gap-merge-min-p",
                str(best_constrained.gap_merge_min_p),
            ]
        run(cmd)

        # Evaluate on the same set (train+eval ids) so we can compare across experiments.
        s3 = get_eval_s3_client()
        labels = load_ground_truth_s3(s3, prefix=args.labels_prefix)
        full_agg, no_call_fp_videos = evaluate_predictions(preds_key_prefix + "/", labels, all_vids)
        print("\nFULL LABELED-SET AGG:")
        print(
            f"  TimeF1(micro)={full_agg.micro_f1:.4f}  SegF1(IoU micro)={full_agg.seg_f1:.4f}  "
            f"Pred={full_agg.total_pred_segments} Truth={full_agg.total_truth_segments}  NoCallFPVideos={no_call_fp_videos}"
        )

    # Side-by-side report
    if not args.skip_report and preds_s3_prefix is not None:
        run(
            [
                sys.executable,
                "scripts/report_call_segmenter_side_by_side.py",
                "--pred-prefix",
                preds_s3_prefix,
                "--labels-prefix",
                f"s3://{S3_BUCKET}/{args.labels_prefix}",
                "--split-meta",
                str(args.split_meta),
                "--out-root",
                "artifacts/s3_audit/call_segmenter",
                "--report-name",
                args.exp_name,
            ]
        )
        # Best-effort: locate the most recent report directory.
        root = Path("artifacts/s3_audit/call_segmenter")
        dirs = sorted([p for p in root.glob(f"{args.exp_name}_*") if p.is_dir()])
        if dirs:
            last = dirs[-1]
            report_md = last / "side_by_side.md"
            report_json = last / "side_by_side.json"

    # Append log
    if preds_s3_prefix is not None and full_agg is not None and best_constrained is not None and best_overall is not None:
        append_experiment_log(
            args.log_path,
            exp_name=args.exp_name,
            dataset_dir=dataset_dir,
            model_dir=model_dir,
            split_meta=args.split_meta,
            best_params=best_constrained,
            best_overall=best_overall,
            eval_result=eval_sweep,
            predictions_s3_prefix=preds_s3_prefix.replace("s3://", "s3://"),
            full_agg=full_agg,
            no_call_fp_videos=no_call_fp_videos,
            report_md_path=report_md if report_md and report_md.exists() else None,
            report_json_path=report_json if report_json and report_json.exists() else None,
        )
        print(f"Appended experiment log: {args.log_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
