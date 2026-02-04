#!/usr/bin/env python3
"""Parameter sweep to optimize call segmenter post-processing.

Runs feature extraction once per video, caches (t_mids, probs), then sweeps
post-processing parameters (threshold, gap_merge_s, min_seg_s) to find optimal
settings that minimize over-segmentation while maintaining high F1.

Key features:
- Tunes on train_video_ids, reports final metrics on eval_video_ids
- No dependence on S3 predictions - recomputes from model + run artifacts
- Scoring prevents "win by spamming segments"
- Cache validation with metadata checks

Usage:
    python scripts/sweep_call_segmenter.py \
        --model-dir data/call_segmenter/models/v1 \
        --cache-dir data/call_segmenter/prob_cache \
        --output-csv data/call_segmenter/sweep_results.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

if TYPE_CHECKING:
    import xgboost as xgb

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.call_segmenter.window_generator import WindowConfig

# Import from sibling scripts (not package) to ensure identical logic
from predict_call_segmenter import (  # type: ignore[import-not-found]
    probabilities_to_segments,
    load_model_and_meta,
    create_text_vectorizer,
    load_video_data_for_inference,
    generate_features_for_video,
    get_s3_client,
    EXCLUDED_VIDEO_IDS,
)
from eval_call_segmenter_predictions import (  # type: ignore[import-not-found]
    Segment,
    TimeMetrics,
    BoundaryMetrics,
    compute_time_metrics,
    compute_boundary_metrics,
    load_ground_truth_s3,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SweepResult:
    """Result for a single parameter combination."""
    threshold: float
    threshold_off: float
    gap_merge_s: float
    gap_merge_min_p: float
    gap_merge_stat: str
    min_seg_s: float
    min_seg_short_s: Optional[float]
    keep_short_p: Optional[float]
    f1: float  # time-level F1
    precision: float
    recall: float
    seg_precision: float
    seg_recall: float
    seg_f1: float
    seg_ratio: float  # predicted / truth segment count
    mae_start_s: Optional[float]
    mae_end_s: Optional[float]
    mean_iou: Optional[float]
    unmatched_pred: int
    unmatched_truth: int
    total_pred: int
    total_truth: int
    # Diagnostics for "no-call" videos (truth has 0 segments). Useful for hard-negative tuning.
    no_call_videos: int = 0
    no_call_fp_videos: int = 0
    no_call_pred_segments: int = 0
    no_call_pred_duration_s: float = 0.0
    score: float = 0.0  # Combined score for ranking


# =============================================================================
# Cache Management
# =============================================================================


def sanitize_video_id(video_id: str) -> str:
    """Sanitize video_id for use in filenames."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", video_id)


def make_cache_filename(video_id: str, run_id: str) -> str:
    """Create cache filename from video_id and run_id."""
    return f"{sanitize_video_id(video_id)}_{run_id}.npz"


def save_cache(
    path: Path,
    t_mids: np.ndarray,
    probs: np.ndarray,
    mp3_duration_s: Optional[float],
    video_id: str,
    run_id: str,
    meta: Dict,
) -> None:
    """Save extracted probabilities to cache file."""
    wc = meta["window_config"]
    th = meta["text_hashing"]
    text_hash = json.dumps(th, sort_keys=True)
    audio_hash = json.dumps(meta.get("audio_features", {}), sort_keys=True)

    np.savez(
        path,
        t_mids=t_mids,
        probs=probs,
        mp3_duration_s=mp3_duration_s if mp3_duration_s is not None else np.nan,
        video_id=video_id,
        run_id=run_id,
        model_git_sha=meta["git_sha"],
        win_s=wc["win_s"],
        hop_s=wc["hop_s"],
        ignore_s=wc.get("ignore_s", 0.75),
        n_features=th["n_features"],
        ngram_range=np.array(th["ngram_range"]),
        analyzer=th["analyzer"],
        text_context_s=float(th.get("context_s", 0.0)),
        text_max_chars=int(th.get("max_chars", 300)),
        text_hash=text_hash,
        audio_hash=audio_hash,
    )


def load_cache(
    path: Path,
    expected_video_id: str,
    expected_run_id: str,
    meta: Dict,
) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """Load and validate cache file.

    Raises ValueError if metadata doesn't match.
    """
    data = np.load(path, allow_pickle=True)
    wc = meta["window_config"]
    th = meta["text_hashing"]

    # Extract scalars with .item() for all reads
    cached_video_id = data["video_id"].item()
    cached_run_id = data["run_id"].item()
    cached_git_sha = data["model_git_sha"].item()
    cached_win_s = float(data["win_s"].item())
    cached_hop_s = float(data["hop_s"].item())
    cached_ignore_s = float(data["ignore_s"].item())
    cached_n_features = int(data["n_features"].item())
    cached_analyzer = data["analyzer"].item()
    cached_text_context_s = float(data["text_context_s"].item()) if "text_context_s" in data.files else None
    cached_text_max_chars = int(data["text_max_chars"].item()) if "text_max_chars" in data.files else None
    cached_text_hash = data["text_hash"].item() if "text_hash" in data.files else None
    cached_audio_hash = data["audio_hash"].item() if "audio_hash" in data.files else None

    # Handle ngram_range array
    ngram = data["ngram_range"]
    ngram_list = ngram.tolist() if hasattr(ngram, "tolist") else list(ngram)

    # Validate all metadata
    expected_ignore_s = wc.get("ignore_s", 0.75)

    if cached_video_id != expected_video_id:
        raise ValueError(f"Cache video_id mismatch: {cached_video_id} != {expected_video_id}")
    if cached_run_id != expected_run_id:
        raise ValueError(f"Cache run_id mismatch: {cached_run_id} != {expected_run_id}")
    if cached_git_sha != meta["git_sha"]:
        raise ValueError(f"Cache git_sha mismatch: {cached_git_sha} != {meta['git_sha']}")
    if not math.isclose(cached_win_s, wc["win_s"], rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"Cache win_s mismatch: {cached_win_s} != {wc['win_s']}")
    if not math.isclose(cached_hop_s, wc["hop_s"], rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"Cache hop_s mismatch: {cached_hop_s} != {wc['hop_s']}")
    if not math.isclose(cached_ignore_s, expected_ignore_s, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"Cache ignore_s mismatch: {cached_ignore_s} != {expected_ignore_s}")
    if cached_n_features != th["n_features"]:
        raise ValueError(f"Cache n_features mismatch: {cached_n_features} != {th['n_features']}")
    if ngram_list != th["ngram_range"]:
        raise ValueError(f"Cache ngram_range mismatch: {ngram_list} != {th['ngram_range']}")
    if cached_analyzer != th["analyzer"]:
        raise ValueError(f"Cache analyzer mismatch: {cached_analyzer} != {th['analyzer']}")
    # Text context affects the hashed features -> probabilities. Validate or invalidate.
    expected_text_context_s = float(th.get("context_s", 0.0))
    expected_text_max_chars = int(th.get("max_chars", 300))
    if cached_text_context_s is None or cached_text_max_chars is None:
        raise ValueError("Cache missing text_context_s/text_max_chars")
    if not math.isclose(cached_text_context_s, expected_text_context_s, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"Cache text_context_s mismatch: {cached_text_context_s} != {expected_text_context_s}")
    if cached_text_max_chars != expected_text_max_chars:
        raise ValueError(f"Cache text_max_chars mismatch: {cached_text_max_chars} != {expected_text_max_chars}")
    expected_text_hash = json.dumps(th, sort_keys=True)
    expected_audio_hash = json.dumps(meta.get("audio_features", {}), sort_keys=True)
    if cached_text_hash is None or cached_audio_hash is None:
        raise ValueError("Cache missing text_hash/audio_hash")
    if cached_text_hash != expected_text_hash:
        raise ValueError("Cache text_hash mismatch")
    if cached_audio_hash != expected_audio_hash:
        raise ValueError("Cache audio_hash mismatch")

    # Extract arrays
    t_mids = data["t_mids"]
    probs = data["probs"]
    mp3_duration_s_val = data["mp3_duration_s"].item()
    mp3_duration_s = None if np.isnan(mp3_duration_s_val) else float(mp3_duration_s_val)

    return t_mids, probs, mp3_duration_s


# =============================================================================
# Feature Extraction
# =============================================================================


def extract_video_probabilities(
    s3_client,
    video_id: str,
    model: "xgb.XGBClassifier",
    vectorizer: HashingVectorizer,
    window_config: WindowConfig,
    feature_columns: List[str],
    text_context_s: float,
    text_max_chars: int,
    audio_features_meta: Optional[Dict] = None,
) -> Optional[Tuple[str, np.ndarray, np.ndarray, Optional[float]]]:
    """Extract probabilities for a video using the trained model.

    Returns:
        Tuple of (run_id, t_mids, probs, mp3_duration_s) or None if failed.
    """
    # Load video data
    video_data = load_video_data_for_inference(s3_client, video_id)
    if video_data is None:
        return None

    run_id, segments, words, processed_end_s, mp3_duration_s, mp3_key, chunk_coverage = video_data

    # Use mp3_duration_s as timeline end if available
    timeline_end = min(processed_end_s, mp3_duration_s) if mp3_duration_s else processed_end_s

    # Generate features
    df, X_combined = generate_features_for_video(
        segments=segments,
        words=words,
        timeline_end=timeline_end,
        window_config=window_config,
        feature_columns=feature_columns,
        vectorizer=vectorizer,
        text_context_s=text_context_s,
        text_max_chars=text_max_chars,
        s3_client=s3_client,
        mp3_key=mp3_key,
        chunk_coverage=chunk_coverage,
        audio_features_meta=audio_features_meta,
    )

    if len(df) == 0:
        logger.warning(f"No windows generated for {video_id}")
        return None

    # Get probabilities from model
    probs = model.predict_proba(X_combined)[:, 1]
    t_mids = df["t_mid"].to_numpy()

    return run_id, t_mids, probs, mp3_duration_s


def cache_all_probabilities(
    s3_client,
    video_ids: List[str],
    model: "xgb.XGBClassifier",
    meta: Dict,
    vectorizer: HashingVectorizer,
    window_config: WindowConfig,
    feature_columns: List[str],
    cache_dir: Optional[Path],
    force_recompute: bool,
) -> Dict[str, Tuple[str, np.ndarray, np.ndarray, Optional[float]]]:
    """Extract or load cached probabilities for all videos.

    Returns:
        Dict mapping video_id -> (run_id, t_mids, probs, mp3_duration_s)
    """
    results: Dict[str, Tuple[str, np.ndarray, np.ndarray, Optional[float]]] = {}

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    for idx, video_id in enumerate(video_ids):
        logger.info(f"[{idx + 1}/{len(video_ids)}] Processing {video_id}")

        # First, get run_id to check cache
        run_id = None
        if cache_dir and not force_recompute:
            # Try to find existing cache file (need to know run_id first)
            from predict_call_segmenter import load_latest_run_id
            run_id = load_latest_run_id(s3_client, video_id)

            if run_id:
                cache_file = cache_dir / make_cache_filename(video_id, run_id)
                if cache_file.exists():
                    try:
                        t_mids, probs, mp3_duration_s = load_cache(
                            cache_file, video_id, run_id, meta
                        )
                        logger.info(f"  Loaded from cache: {cache_file.name}")
                        results[video_id] = (run_id, t_mids, probs, mp3_duration_s)
                        continue
                    except ValueError as e:
                        logger.warning(f"  Cache invalid, re-extracting: {e}")

        # Extract features and get probabilities
        result = extract_video_probabilities(
            s3_client,
            video_id,
            model,
            vectorizer,
            window_config,
            feature_columns,
            text_context_s=float(meta["text_hashing"].get("context_s", 0.0)),
            text_max_chars=int(meta["text_hashing"].get("max_chars", 300)),
            audio_features_meta=meta.get("audio_features"),
        )

        if result is None:
            logger.warning(f"  Skipping {video_id}: extraction failed")
            continue

        run_id, t_mids, probs, mp3_duration_s = result
        results[video_id] = (run_id, t_mids, probs, mp3_duration_s)
        logger.info(f"  Extracted: {len(t_mids)} windows")

        # Save to cache
        if cache_dir:
            cache_file = cache_dir / make_cache_filename(video_id, run_id)
            save_cache(cache_file, t_mids, probs, mp3_duration_s, video_id, run_id, meta)
            logger.info(f"  Cached to: {cache_file.name}")

    return results


# =============================================================================
# Sweep Logic
# =============================================================================


def evaluate_params_on_videos(
    prob_cache: Dict[str, Tuple[str, np.ndarray, np.ndarray, Optional[float]]],
    ground_truth: Dict[str, List[Segment]],
    win_s: float,
    threshold: float,
    threshold_off: float,
    gap_merge_s: float,
    gap_merge_min_p: float,
    gap_merge_stat: str,
    min_seg_s: float,
    min_seg_short_s: Optional[float],
    keep_short_p: Optional[float],
) -> Optional[SweepResult]:
    """Evaluate parameter combination on a set of videos."""
    # Collect all predictions and truths
    all_time_metrics: List[TimeMetrics] = []
    all_boundary_metrics: List[BoundaryMetrics] = []
    total_pred_segments = 0
    total_truth_segments = 0
    total_matched_pairs = 0
    total_intersection_s = 0.0
    total_pred_duration_s = 0.0
    total_truth_duration_s = 0.0
    total_iou_weighted = 0.0
    no_call_videos = 0
    no_call_fp_videos = 0
    no_call_pred_segments = 0
    no_call_pred_duration_s = 0.0

    for video_id in prob_cache:
        if video_id not in ground_truth:
            continue

        _run_id, t_mids, probs, mp3_duration_s = prob_cache[video_id]
        truth_segments = ground_truth[video_id]

        # Convert probabilities to segments with current params
        pred_segments_raw = probabilities_to_segments(
            t_mids=t_mids,
            probs=probs,
            win_s=win_s,
            threshold=threshold,
            threshold_off=threshold_off,
            gap_merge_s=gap_merge_s,
            min_seg_s=min_seg_s,
            gap_merge_min_p=gap_merge_min_p,
            gap_merge_stat=gap_merge_stat,
            min_seg_short_s=min_seg_short_s,
            keep_short_p=keep_short_p,
            mp3_duration_s=mp3_duration_s,
        )

        # Convert to Segment format for eval
        pred_segments = [
            Segment(start_s=s.start_s, end_s=s.end_s) for s in pred_segments_raw
        ]

        # Compute metrics
        time_metrics = compute_time_metrics(pred_segments, truth_segments)
        all_time_metrics.append(time_metrics)

        total_intersection_s += time_metrics.intersection_s
        total_pred_duration_s += time_metrics.pred_duration_s
        total_truth_duration_s += time_metrics.truth_duration_s

        # Track "no-call" false positives separately. These can get washed out in micro averages.
        if len(truth_segments) == 0:
            no_call_videos += 1
            no_call_pred_segments += len(pred_segments)
            no_call_pred_duration_s += time_metrics.pred_duration_s
            if len(pred_segments) > 0:
                no_call_fp_videos += 1

        # Only compute boundary metrics if there are truth segments
        if len(truth_segments) > 0:
            boundary_metrics = compute_boundary_metrics(pred_segments, truth_segments)
            all_boundary_metrics.append(boundary_metrics)
            total_matched_pairs += boundary_metrics.matched_pairs
            if boundary_metrics.mean_iou is not None:
                total_iou_weighted += boundary_metrics.mean_iou * boundary_metrics.matched_pairs

        total_pred_segments += len(pred_segments)
        total_truth_segments += len(truth_segments)

    if not all_time_metrics:
        return None

    # Micro-averaged time metrics
    if total_pred_duration_s > 0:
        precision = total_intersection_s / total_pred_duration_s
    else:
        precision = 1.0 if total_truth_duration_s == 0 else 0.0

    if total_truth_duration_s > 0:
        recall = total_intersection_s / total_truth_duration_s
    else:
        recall = 1.0 if total_pred_duration_s == 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    # Segment ratio (handle edge cases)
    if total_truth_segments > 0:
        seg_ratio = total_pred_segments / total_truth_segments
    else:
        # No truth segments - use inf to heavily penalize any predictions
        seg_ratio = float("inf") if total_pred_segments > 0 else 1.0

    # Aggregate boundary metrics
    total_unmatched_pred = sum(m.unmatched_pred for m in all_boundary_metrics)
    total_unmatched_truth = sum(m.unmatched_truth for m in all_boundary_metrics)

    mae_starts = [m.mae_start_s for m in all_boundary_metrics if m.mae_start_s is not None]
    mae_ends = [m.mae_end_s for m in all_boundary_metrics if m.mae_end_s is not None]

    mae_start_s = float(np.mean(mae_starts)) if mae_starts else None
    mae_end_s = float(np.mean(mae_ends)) if mae_ends else None

    mean_iou = None
    if total_matched_pairs > 0:
        mean_iou = total_iou_weighted / total_matched_pairs

    # Segment-level micro metrics from IoU matches
    if total_pred_segments > 0:
        seg_precision = total_matched_pairs / total_pred_segments
    else:
        seg_precision = 1.0 if total_truth_segments == 0 else 0.0

    if total_truth_segments > 0:
        seg_recall = total_matched_pairs / total_truth_segments
    else:
        seg_recall = 1.0 if total_pred_segments == 0 else 0.0

    if seg_precision + seg_recall > 0:
        seg_f1 = 2 * seg_precision * seg_recall / (seg_precision + seg_recall)
    else:
        seg_f1 = 0.0

    return SweepResult(
        threshold=threshold,
        threshold_off=threshold_off,
        gap_merge_s=gap_merge_s,
        gap_merge_min_p=gap_merge_min_p,
        gap_merge_stat=gap_merge_stat,
        min_seg_s=min_seg_s,
        min_seg_short_s=min_seg_short_s,
        keep_short_p=keep_short_p,
        f1=f1,
        precision=precision,
        recall=recall,
        seg_precision=seg_precision,
        seg_recall=seg_recall,
        seg_f1=seg_f1,
        seg_ratio=seg_ratio,
        mae_start_s=mae_start_s,
        mae_end_s=mae_end_s,
        mean_iou=mean_iou,
        unmatched_pred=total_unmatched_pred,
        unmatched_truth=total_unmatched_truth,
        total_pred=total_pred_segments,
        total_truth=total_truth_segments,
        no_call_videos=no_call_videos,
        no_call_fp_videos=no_call_fp_videos,
        no_call_pred_segments=no_call_pred_segments,
        no_call_pred_duration_s=float(no_call_pred_duration_s),
    )


def run_parameter_sweep(
    prob_cache: Dict[str, Tuple[str, np.ndarray, np.ndarray, Optional[float]]],
    ground_truth: Dict[str, List[Segment]],
    win_s: float,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
    threshold_off_delta_values: List[float],
    gap_merge_values: List[float],
    gap_merge_min_p_values: List[float],
    gap_merge_stat_values: List[str],
    min_seg_values: List[float],
    min_seg_short_values: List[Optional[float]],
    keep_short_p_values: List[Optional[float]],
) -> List[SweepResult]:
    """Run sweep over all parameter combinations."""
    results: List[SweepResult] = []

    # Build threshold list (MAX-inclusive using while loop)
    thresholds: List[float] = []
    t = threshold_min
    while t <= threshold_max + 1e-9:
        thresholds.append(round(t, 4))
        t += threshold_step

    total_combos = (
        len(thresholds)
        * len(threshold_off_delta_values)
        * len(gap_merge_values)
        * len(gap_merge_min_p_values)
        * len(gap_merge_stat_values)
        * len(min_seg_values)
        * len(min_seg_short_values)
        * len(keep_short_p_values)
    )
    logger.info(
        f"Running sweep: {len(thresholds)} thresholds x {len(threshold_off_delta_values)} off-deltas x "
        f"{len(gap_merge_values)} gaps x {len(gap_merge_min_p_values)} gap-min-p x {len(gap_merge_stat_values)} gap-stats x "
        f"{len(min_seg_values)} min_segs x {len(min_seg_short_values)} min_short x {len(keep_short_p_values)} keep_p = "
        f"{total_combos} combinations"
    )

    combo_idx = 0
    for threshold in thresholds:
        for off_delta in threshold_off_delta_values:
            threshold_off = max(0.0, threshold - off_delta)
            for gap_merge_s in gap_merge_values:
                for gap_merge_min_p in gap_merge_min_p_values:
                    for gap_merge_stat in gap_merge_stat_values:
                        for min_seg_s in min_seg_values:
                            for min_seg_short_s in min_seg_short_values:
                                for keep_short_p in keep_short_p_values:
                                    # Enforce pairing: either both None, or both set
                                    if (min_seg_short_s is None) ^ (keep_short_p is None):
                                        continue

                                    combo_idx += 1
                                    if combo_idx % 200 == 0:
                                        logger.info(f"  Progress: {combo_idx}/{total_combos}")

                                    result = evaluate_params_on_videos(
                                        prob_cache=prob_cache,
                                        ground_truth=ground_truth,
                                        win_s=win_s,
                                        threshold=threshold,
                                        threshold_off=threshold_off,
                                        gap_merge_s=gap_merge_s,
                                        gap_merge_min_p=gap_merge_min_p,
                                        gap_merge_stat=gap_merge_stat,
                                        min_seg_s=min_seg_s,
                                        min_seg_short_s=min_seg_short_s,
                                        keep_short_p=keep_short_p,
                                    )

                                    if result:
                                        results.append(result)

    return results


def compute_scores(results: List[SweepResult]) -> None:
    """Compute combined score for each result (modifies in place)."""
    for r in results:
        # Score trades off:
        # - segment-level F1 (IoU-matched) to reward correct call splits
        # - time-level F1 to keep "call time" coverage
        # - mild seg_ratio penalty to avoid pathological under/over-segmentation
        # - small penalties for unmatched segments on either side
        # Guard against seg_ratio == 0 or inf
        if r.seg_ratio == float("inf"):
            log_penalty = 10.0  # Large penalty for inf
        else:
            log_penalty = abs(math.log(max(r.seg_ratio, 1e-6)))

        r.score = (
            0.65 * r.seg_f1
            + 0.35 * r.f1
            - 0.02 * log_penalty
            - 0.0005 * r.unmatched_pred
            - 0.0005 * r.unmatched_truth
            # Penalize false positives on no-call videos more aggressively than micro averages would.
            - 0.03 * r.no_call_fp_videos
            - 0.0002 * r.no_call_pred_duration_s
        )


def select_best_params(results: List[SweepResult]) -> Tuple[SweepResult, SweepResult]:
    """Select best parameters with constraints and fallback.

    Returns:
        Tuple of (best_constrained, best_overall)
    """
    if not results:
        raise ValueError("No results to select from")

    # Compute scores
    compute_scores(results)

    # Best unconstrained (for comparison)
    best_overall = max(results, key=lambda r: r.score)

    # Primary gates: we care about correct splits, not just time coverage.
    # - seg_ratio close to 1.0 prevents "winning by spam" or over-merging
    # - time F1 gate prevents degenerate low-coverage solutions
    seg_ratio_min, seg_ratio_max = 0.9, 1.2
    time_f1_min = 0.85
    no_call_fp_max = 0

    valid = [
        r
        for r in results
        if (
            seg_ratio_min <= r.seg_ratio <= seg_ratio_max
            and r.f1 >= time_f1_min
            and r.no_call_fp_videos <= no_call_fp_max
        )
    ]

    if valid:
        winner = max(valid, key=lambda r: r.score)
        return winner, best_overall

    # Fallback: widen seg_ratio constraints, but keep time-F1 gate.
    fallback = [
        r
        for r in results
        if (
            0.8 <= r.seg_ratio <= 1.5
            and r.f1 >= time_f1_min
            and r.no_call_fp_videos <= no_call_fp_max
        )
    ]
    if fallback:
        logger.warning(
            "No params in [0.9, 1.2] seg_ratio with time-F1>=0.85 and no-call FP==0, using [0.8, 1.5]"
        )
        winner = max(fallback, key=lambda r: r.score)
        return winner, best_overall

    fallback2 = [
        r
        for r in results
        if (
            0.5 <= r.seg_ratio <= 2.0
            and r.f1 >= time_f1_min
            and r.no_call_fp_videos <= no_call_fp_max
        )
    ]
    if fallback2:
        logger.warning(
            "No params in [0.8, 1.5] seg_ratio with time-F1>=0.85 and no-call FP==0, using [0.5, 2.0]"
        )
        winner = max(fallback2, key=lambda r: r.score)
        return winner, best_overall

    # If we still have no valid params, relax the time-F1 gate but keep seg-ratio sanity.
    fallback3 = [
        r for r in results if (0.8 <= r.seg_ratio <= 1.5 and r.no_call_fp_videos <= no_call_fp_max)
    ]
    if fallback3:
        logger.warning(
            "No params meet time-F1>=0.85; using best in [0.8, 1.5] seg_ratio with no-call FP==0"
        )
        winner = max(fallback3, key=lambda r: r.score)
        return winner, best_overall

    fallback4 = [
        r for r in results if (0.5 <= r.seg_ratio <= 2.0 and r.no_call_fp_videos <= no_call_fp_max)
    ]
    if fallback4:
        logger.warning(
            "No params meet time-F1>=0.85; using best in [0.5, 2.0] seg_ratio with no-call FP==0"
        )
        winner = max(fallback4, key=lambda r: r.score)
        return winner, best_overall

    # Last resort: best score overall
    logger.warning("No params meet seg_ratio constraints, using best overall")
    return best_overall, best_overall


# =============================================================================
# Output Formatting
# =============================================================================


def format_result_row(r: SweepResult) -> str:
    """Format a single result row for display."""
    mae_s = f"{r.mae_start_s:.2f}s" if r.mae_start_s is not None else "N/A"
    mae_e = f"{r.mae_end_s:.2f}s" if r.mae_end_s is not None else "N/A"
    iou = f"{r.mean_iou:.3f}" if r.mean_iou is not None else "N/A"
    seg_ratio_str = f"{r.seg_ratio:.2f}" if r.seg_ratio != float("inf") else "inf"
    min_short = f"{r.min_seg_short_s:.0f}" if r.min_seg_short_s is not None else "-"
    keep_p = f"{r.keep_short_p:.2f}" if r.keep_short_p is not None else "-"
    if r.no_call_videos > 0:
        no_call_fp = f"{r.no_call_fp_videos}/{r.no_call_videos}"
        no_call_dur = f"{r.no_call_pred_duration_s:.1f}s"
    else:
        no_call_fp = "-"
        no_call_dur = "-"

    return (
        f"{r.threshold:>6.2f}  {r.threshold_off:>6.2f}  {r.gap_merge_s:>5.0f}  {r.gap_merge_stat:>5}  {r.gap_merge_min_p:>7.2f}  "
        f"{r.min_seg_s:>6.0f}  {min_short:>5}  {keep_p:>5}  "
        f"{r.f1:>7.4f}  {r.seg_f1:>7.4f}  {seg_ratio_str:>8}  {iou:>6}  {mae_s:>7}  {mae_e:>7}  "
        f"{r.unmatched_pred:>8}  {r.unmatched_truth:>8}  {no_call_fp:>8}  {no_call_dur:>9}  {r.score:>7.4f}"
    )


def print_sweep_results(results: List[SweepResult], top_n: int, title: str) -> None:
    """Print top N results from sweep."""
    print(f"\n{'=' * 100}")
    print(f"{title}")
    print("=" * 100)
    print(
        f"{'ThrOn':>6}  {'ThrOff':>6}  {'Gap':>5}  {'GStat':>5}  {'GapMinP':>7}  {'MinSeg':>6}  {'MinSh':>5}  {'KeepP':>5}  "
        f"{'TimeF1':>7}  {'SegF1':>7}  {'SegRatio':>8}  {'IoU':>6}  {'MAE-S':>7}  {'MAE-E':>7}  "
        f"{'UnmtchP':>8}  {'UnmtchT':>8}  {'NoCallFP':>8}  {'NoCallDur':>9}  {'Score':>7}"
    )
    print("-" * 100)

    # Sort by score descending
    sorted_results = sorted(results, key=lambda r: r.score, reverse=True)

    for r in sorted_results[:top_n]:
        print(format_result_row(r))

    print("-" * 100)


def save_results_csv(results: List[SweepResult], output_path: Path) -> None:
    """Save all sweep results to CSV."""
    rows = []
    for r in results:
        rows.append({
            "threshold": r.threshold,
            "threshold_off": r.threshold_off,
            "gap_merge_s": r.gap_merge_s,
            "gap_merge_min_p": r.gap_merge_min_p,
            "gap_merge_stat": r.gap_merge_stat,
            "min_seg_s": r.min_seg_s,
            "min_seg_short_s": r.min_seg_short_s,
            "keep_short_p": r.keep_short_p,
            "time_f1": r.f1,
            "precision": r.precision,
            "recall": r.recall,
            "seg_f1": r.seg_f1,
            "seg_precision": r.seg_precision,
            "seg_recall": r.seg_recall,
            "seg_ratio": r.seg_ratio if r.seg_ratio != float("inf") else None,
            "mae_start_s": r.mae_start_s,
            "mae_end_s": r.mae_end_s,
            "mean_iou": r.mean_iou,
            "unmatched_pred": r.unmatched_pred,
            "unmatched_truth": r.unmatched_truth,
            "total_pred": r.total_pred,
            "total_truth": r.total_truth,
            "no_call_videos": r.no_call_videos,
            "no_call_fp_videos": r.no_call_fp_videos,
            "no_call_pred_segments": r.no_call_pred_segments,
            "no_call_pred_duration_s": r.no_call_pred_duration_s,
            "score": r.score,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(rows)} results to {output_path}")


# =============================================================================
# Main
# =============================================================================


def parse_float_list(s: str) -> List[float]:
    """Parse comma-separated list of floats."""
    return [float(x.strip()) for x in s.split(",")]


def parse_str_list(s: str) -> List[str]:
    """Parse comma-separated list of strings."""
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_optional_float_list(s: str) -> List[Optional[float]]:
    """Parse comma-separated list of floats, or return [None] if empty."""
    if not s.strip():
        return [None]
    parts = [x.strip() for x in s.split(",") if x.strip()]
    if any(p.lower() == "none" for p in parts):
        raise ValueError("Use empty string to disable optional lists; 'none' is not supported")
    return [float(p) for p in parts]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parameter sweep for call segmenter post-processing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", type=Path, default=Path("data/call_segmenter/models/v1"),
                        help="Directory containing model_b.ubj and meta.json")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Directory to cache (t_mids, probs) as .npz files")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Ignore existing cache, re-extract all")
    parser.add_argument("--threshold-range", type=str, default="0.35,0.75,0.05",
                        help="Threshold sweep: min,max,step (inclusive of max)")
    parser.add_argument("--threshold-off-delta-values", type=str, default="0.0,0.05,0.10,0.15,0.20",
                        help="Comma-separated list of (threshold - threshold_off) deltas to sweep. "
                             "0.0 means no hysteresis (threshold_off == threshold).")
    parser.add_argument("--gap-merge-values", type=str, default="0,1,2,5,10,20,30",
                        help="Comma-separated gap merge values (seconds)")
    parser.add_argument("--gap-merge-min-p-values", type=str, default="0,0.2,0.4,0.6,0.8",
                        help="Comma-separated values for conditional gap merging. "
                             "A gap only merges if the chosen gap statistic >= gap_merge_min_p (0 disables).")
    parser.add_argument("--gap-merge-stat-values", type=str, default="max,mean,p90",
                        help="Comma-separated gap statistics to compare against gap-merge-min-p: max, mean, p90")
    parser.add_argument("--min-seg-values", type=str, default="1,3,5,10",
                        help="Comma-separated minimum segment values (seconds)")
    parser.add_argument("--min-seg-short-values", type=str, default="",
                        help="Optional: comma-separated short min durations (seconds) to keep high-confidence short segments; empty disables")
    parser.add_argument("--keep-short-p-values", type=str, default="",
                        help="Optional: comma-separated mean_p thresholds for keeping short segments; empty disables")
    parser.add_argument("--output-csv", type=Path, default=None,
                        help="Save full results to CSV")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Print top N combinations")

    args = parser.parse_args()

    # Parse parameter ranges
    thresh_parts = args.threshold_range.split(",")
    if len(thresh_parts) != 3:
        logger.error("Invalid --threshold-range format. Use: min,max,step")
        return 1
    threshold_min, threshold_max, threshold_step = map(float, thresh_parts)

    gap_merge_values = parse_float_list(args.gap_merge_values)
    gap_merge_min_p_values = parse_float_list(args.gap_merge_min_p_values)
    gap_merge_stat_values = [s.lower() for s in parse_str_list(args.gap_merge_stat_values)]
    min_seg_values = parse_float_list(args.min_seg_values)
    threshold_off_delta_values = parse_float_list(args.threshold_off_delta_values)

    try:
        min_seg_short_values = parse_optional_float_list(args.min_seg_short_values)
        keep_short_p_values = parse_optional_float_list(args.keep_short_p_values)
    except Exception as e:
        logger.error(f"Failed to parse short-seg args: {e}")
        return 1

    # Validate gap stat values early
    allowed_stats = {"max", "mean", "p90"}
    bad_stats = [s for s in gap_merge_stat_values if s not in allowed_stats]
    if bad_stats:
        logger.error(f"Invalid --gap-merge-stat-values entries: {bad_stats}. Allowed: {sorted(allowed_stats)}")
        return 1

    # Optional short-seg sweeping: require both lists to be enabled together.
    short_enabled = args.min_seg_short_values.strip() != "" or args.keep_short_p_values.strip() != ""
    if short_enabled and not (args.min_seg_short_values.strip() != "" and args.keep_short_p_values.strip() != ""):
        logger.error("--min-seg-short-values and --keep-short-p-values must be set together (or both omitted)")
        return 1

    # If enabled, include the disabled baseline (None/None) for comparison.
    if short_enabled:
        if min_seg_short_values != [None]:
            min_seg_short_values = [None] + min_seg_short_values
        if keep_short_p_values != [None]:
            keep_short_p_values = [None] + keep_short_p_values

    # Load model and metadata
    try:
        import xgboost as xgb
        model, meta = load_model_and_meta(args.model_dir)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1

    # Get train/eval video IDs from meta.json
    train_video_ids = meta.get("train_video_ids", [])
    eval_video_ids = meta.get("eval_video_ids", [])

    if not train_video_ids:
        logger.error("No train_video_ids in meta.json")
        return 1
    if not eval_video_ids:
        logger.error("No eval_video_ids in meta.json")
        return 1

    logger.info(f"Train videos: {len(train_video_ids)}, Eval videos: {len(eval_video_ids)}")

    # Build window config
    window_config_dict = meta["window_config"]
    window_config = WindowConfig(
        win_s=window_config_dict["win_s"],
        hop_s=window_config_dict["hop_s"],
        ignore_s=window_config_dict.get("ignore_s", 0.75),
    )

    # Create text vectorizer
    vectorizer = create_text_vectorizer(meta["text_hashing"])
    feature_columns = meta["feature_columns"]

    # Get S3 client and load ground truth
    s3 = get_s3_client()
    ground_truth = load_ground_truth_s3(s3)
    logger.info(f"Loaded ground truth for {len(ground_truth)} videos")

    # Filter to videos with ground truth
    train_videos_with_gt = [v for v in train_video_ids if v in ground_truth and v not in EXCLUDED_VIDEO_IDS]
    eval_videos_with_gt = [v for v in eval_video_ids if v in ground_truth and v not in EXCLUDED_VIDEO_IDS]

    logger.info(f"Train videos with ground truth: {len(train_videos_with_gt)}")
    logger.info(f"Eval videos with ground truth: {len(eval_videos_with_gt)}")

    if not train_videos_with_gt:
        logger.error("No training videos have ground truth!")
        return 1

    # Extract probabilities for train set
    logger.info("\n" + "=" * 60)
    logger.info("Extracting probabilities for TRAIN set")
    logger.info("=" * 60)

    train_prob_cache = cache_all_probabilities(
        s3_client=s3,
        video_ids=train_videos_with_gt,
        model=model,
        meta=meta,
        vectorizer=vectorizer,
        window_config=window_config,
        feature_columns=feature_columns,
        cache_dir=args.cache_dir,
        force_recompute=args.force_recompute,
    )

    if not train_prob_cache:
        logger.error("No training videos could be processed!")
        return 1

    # Run parameter sweep on train set
    logger.info("\n" + "=" * 60)
    logger.info("Running parameter sweep on TRAIN set")
    logger.info("=" * 60)

    train_results = run_parameter_sweep(
        prob_cache=train_prob_cache,
        ground_truth=ground_truth,
        win_s=window_config.win_s,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_step=threshold_step,
        threshold_off_delta_values=threshold_off_delta_values,
        gap_merge_values=gap_merge_values,
        gap_merge_min_p_values=gap_merge_min_p_values,
        gap_merge_stat_values=gap_merge_stat_values,
        min_seg_values=min_seg_values,
        min_seg_short_values=min_seg_short_values,
        keep_short_p_values=keep_short_p_values,
    )

    if not train_results:
        logger.error("No valid sweep results!")
        return 1

    # Select best parameters (also computes scores)
    best_constrained, best_overall = select_best_params(train_results)

    # Print train results (after scores computed)
    print_sweep_results(train_results, args.top_n, f"TRAIN SET SWEEP (Top {args.top_n} by Score)")

    print("\n" + "=" * 100)
    print("BEST PARAMETERS")
    print("=" * 100)
    print(
        f"BEST (constrained): thr_on={best_constrained.threshold:.2f}, thr_off={best_constrained.threshold_off:.2f}, "
        f"gap={best_constrained.gap_merge_s:.0f}, gap_min_p={best_constrained.gap_merge_min_p:.2f}, gap_stat={best_constrained.gap_merge_stat}, "
        f"min_seg={best_constrained.min_seg_s:.0f}, min_short={best_constrained.min_seg_short_s}, keep_p={best_constrained.keep_short_p}, "
        f"time_f1={best_constrained.f1:.4f}, seg_f1={best_constrained.seg_f1:.4f}, seg_ratio={best_constrained.seg_ratio:.2f}"
    )
    print(
        f"BEST (overall):     thr_on={best_overall.threshold:.2f}, thr_off={best_overall.threshold_off:.2f}, "
        f"gap={best_overall.gap_merge_s:.0f}, gap_min_p={best_overall.gap_merge_min_p:.2f}, gap_stat={best_overall.gap_merge_stat}, "
        f"min_seg={best_overall.min_seg_s:.0f}, min_short={best_overall.min_seg_short_s}, keep_p={best_overall.keep_short_p}, "
        f"time_f1={best_overall.f1:.4f}, seg_f1={best_overall.seg_f1:.4f}, seg_ratio={best_overall.seg_ratio:.2f}"
    )

    # Evaluate on eval set with best params
    if eval_videos_with_gt:
        logger.info("\n" + "=" * 60)
        logger.info("Extracting probabilities for EVAL set")
        logger.info("=" * 60)

        eval_prob_cache = cache_all_probabilities(
            s3_client=s3,
            video_ids=eval_videos_with_gt,
            model=model,
            meta=meta,
            vectorizer=vectorizer,
            window_config=window_config,
            feature_columns=feature_columns,
            cache_dir=args.cache_dir,
            force_recompute=args.force_recompute,
        )

        if eval_prob_cache:
            eval_result = evaluate_params_on_videos(
                prob_cache=eval_prob_cache,
                ground_truth=ground_truth,
                win_s=window_config.win_s,
                threshold=best_constrained.threshold,
                threshold_off=best_constrained.threshold_off,
                gap_merge_s=best_constrained.gap_merge_s,
                gap_merge_min_p=best_constrained.gap_merge_min_p,
                gap_merge_stat=best_constrained.gap_merge_stat,
                min_seg_s=best_constrained.min_seg_s,
                min_seg_short_s=best_constrained.min_seg_short_s,
                keep_short_p=best_constrained.keep_short_p,
            )

            if eval_result:
                mae_s_str = f"{eval_result.mae_start_s:.2f}s" if eval_result.mae_start_s is not None else "N/A"
                mae_e_str = f"{eval_result.mae_end_s:.2f}s" if eval_result.mae_end_s is not None else "N/A"
                seg_ratio_str = f"{eval_result.seg_ratio:.2f}" if eval_result.seg_ratio != float("inf") else "inf"

                print("\n" + "=" * 100)
                print(f"EVAL SET (best params: thr_on={best_constrained.threshold:.2f}, "
                      f"thr_off={best_constrained.threshold_off:.2f}, gap={best_constrained.gap_merge_s:.0f}, "
                      f"gap_stat={best_constrained.gap_merge_stat}, gap_min_p={best_constrained.gap_merge_min_p:.2f}, "
                      f"min_seg={best_constrained.min_seg_s:.0f}, "
                      f"min_short={best_constrained.min_seg_short_s}, keep_p={best_constrained.keep_short_p})")
                print("=" * 100)
                iou_str = f"{eval_result.mean_iou:.3f}" if eval_result.mean_iou is not None else "N/A"
                print(f"TimeF1: {eval_result.f1:.4f}  SegF1: {eval_result.seg_f1:.4f}  "
                      f"SegRatio: {seg_ratio_str}  IoU: {iou_str}  MAE-S: {mae_s_str}  MAE-E: {mae_e_str}")
                print(f"Pred: {eval_result.total_pred}  Truth: {eval_result.total_truth}  "
                      f"UnmatchedP: {eval_result.unmatched_pred}  UnmatchedT: {eval_result.unmatched_truth}")
        else:
            logger.warning("No eval videos could be processed")
    else:
        logger.warning("No eval videos have ground truth - skipping eval set evaluation")

    # Save results to CSV
    if args.output_csv:
        save_results_csv(train_results, args.output_csv)

    # Print command to re-run predictions with best params
    print("\n" + "=" * 100)
    print("NEXT STEPS")
    print("=" * 100)
    print("Create a video list from the model's meta.json (train+eval ids):")
    print("  python - <<'PY'")
    print(f"import json\nfrom pathlib import Path\nm=json.loads(Path('{args.model_dir}/meta.json').read_text())\nvids=sorted(set(m.get('train_video_ids', [])) | set(m.get('eval_video_ids', [])))\nPath('/tmp/call_segmenter_videos.txt').write_text('\\n'.join(vids) + '\\n')\nprint('wrote', len(vids), 'video ids to /tmp/call_segmenter_videos.txt')\nPY")
    print()
    print("Re-run predictions with best params:")
    print(f"  python scripts/predict_call_segmenter.py \\")
    print(f"      --video-list /tmp/call_segmenter_videos.txt \\")
    print(f"      --threshold {best_constrained.threshold:.2f} \\")
    print(f"      --threshold-off {best_constrained.threshold_off:.2f} \\")
    print(f"      --gap-merge-s {best_constrained.gap_merge_s:.0f} \\")
    print(f"      --gap-merge-stat {best_constrained.gap_merge_stat} \\")
    print(f"      --gap-merge-min-p {best_constrained.gap_merge_min_p:.2f} \\")
    if best_constrained.min_seg_short_s is not None and best_constrained.keep_short_p is not None:
        print(f"      --min-seg-short-s {best_constrained.min_seg_short_s:.0f} \\")
        print(f"      --keep-short-p {best_constrained.keep_short_p:.2f} \\")
    print(f"      --min-seg-s {best_constrained.min_seg_s:.0f}")
    print()
    print("Then evaluate:")
    print("  python scripts/eval_call_segmenter_predictions.py \\")
    print("      --predictions s3://rezora-whisperx-us-east-1-864981718771/call_segmenter/predictions/v1/ \\")
    print("      --ground-truth s3")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
