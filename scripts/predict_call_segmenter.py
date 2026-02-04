#!/usr/bin/env python3
"""Predict call segments using trained XGBoost model.

Loads a trained Model B (diarization + hashed text features) and predicts
call segments for videos. Outputs predictions to S3.

Usage:
    python scripts/predict_call_segmenter.py \
        --video-list data/video_ids.txt \
        --model-dir data/call_segmenter/models/v1 \
        --s3-out-prefix call_segmenter/predictions/v1
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import boto3
from scipy import sparse
from scipy import signal
from sklearn.feature_extraction.text import HashingVectorizer

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import S3_BUCKET, AWS_REGION
from pipeline.audio_preprocess import decode_audio_to_float32
from pipeline.transcriber import DiarizationSegment
from pipeline.call_segmenter.features import merge_adjacent_segments, compute_window_features
from pipeline.call_segmenter.window_generator import WindowConfig, generate_window_times

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Videos excluded due to label/processing issues
EXCLUDED_VIDEO_IDS = {"EVwBLXWlZiI", "FBmODQn9grE", "N7XqeLuVOzk"}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PredictedSegment:
    """A predicted call segment."""
    start_s: float
    end_s: float
    mean_p: float


@dataclass
class PredictionResult:
    """Complete prediction output for a video."""
    video_id: str
    run_id: str
    model_git_sha: str
    threshold: float
    threshold_off: Optional[float]
    gap_merge_s: float
    gap_merge_min_p: float
    min_seg_s: float
    win_s: float
    hop_s: float
    mp3_duration_s: Optional[float]
    segments: List[PredictedSegment]
    predicted_at: str
    num_windows: int


# =============================================================================
# Model Loading
# =============================================================================


def load_model_and_meta(model_dir: Path) -> Tuple["xgb.XGBClassifier", Dict]:
    """Load XGBoost model and metadata from model directory."""
    import xgboost as xgb

    model_path = model_dir / "model_b.ubj"
    meta_path = model_dir / "meta.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    with open(meta_path) as f:
        meta = json.load(f)

    logger.info(f"Loaded model from {model_dir}")
    logger.info(f"  Git SHA: {meta.get('git_sha', 'unknown')}")
    logger.info(f"  Default threshold: {meta.get('default_threshold', 0.5)}")

    return model, meta


def create_text_vectorizer(text_hashing: Dict) -> HashingVectorizer:
    """Create HashingVectorizer with exact params from training."""
    ngram_range = tuple(text_hashing.get("ngram_range", [2, 5]))
    return HashingVectorizer(
        n_features=text_hashing.get("n_features", 4096),
        analyzer=text_hashing.get("analyzer", "char_wb"),
        ngram_range=ngram_range,
        alternate_sign=False,
        norm=None,
        lowercase=True,
    )


# =============================================================================
# S3 Data Loading (adapted from build_call_segmenter_dataset.py)
# =============================================================================


def get_s3_client():
    return boto3.client("s3", region_name=AWS_REGION)


def load_latest_run_id(s3_client, video_id: str) -> Optional[str]:
    key = f"latest/{video_id}.json"
    try:
        resp = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        data = json.loads(resp["Body"].read())
        return data.get("run_id")
    except Exception as e:
        logger.warning(f"Could not load latest pointer for {video_id}: {e}")
        return None


def list_chunks(s3_client, video_id: str, run_id: str) -> List[str]:
    prefix = f"runs/{video_id}/{run_id}/chunks/"
    chunk_ids = set()
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            parts = obj["Key"].split("/")
            if len(parts) >= 5:
                chunk_ids.add(parts[4])
    return sorted(chunk_ids)


def load_chunk_metadata(s3_client, video_id: str, run_id: str, chunk_id: str) -> Optional[Dict]:
    key = f"runs/{video_id}/{run_id}/chunks/{chunk_id}/chunk_metadata.json"
    try:
        resp = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        return json.loads(resp["Body"].read())
    except Exception:
        return None


def load_diarization_segments(s3_client, video_id: str, run_id: str, chunk_id: str) -> List[DiarizationSegment]:
    key = f"runs/{video_id}/{run_id}/chunks/{chunk_id}/diarization_segments.json"
    try:
        resp = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        data = json.loads(resp["Body"].read())
        return [
            DiarizationSegment(
                t0_abs=float(seg["t0_abs"]),
                t1_abs=float(seg["t1_abs"]),
                spk=str(seg["spk"]),
            )
            for seg in data.get("segments", [])
        ]
    except Exception:
        return []


def load_words(s3_client, video_id: str, run_id: str, chunk_id: str) -> List[dict]:
    key = f"runs/{video_id}/{run_id}/chunks/{chunk_id}/words.json"
    try:
        resp = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        data = json.loads(resp["Body"].read())
        words = data.get("words", [])
        for w in words:
            if "t0_abs" in w:
                w["t0_abs"] = float(w["t0_abs"])
            if "t1_abs" in w:
                w["t1_abs"] = float(w["t1_abs"])
        return words
    except Exception:
        return []


def get_mp3_key_for_video(s3_client, video_id: str) -> Optional[str]:
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix="audio/pretraining/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if video_id in key and key.endswith(".mp3"):
                return key
    return None


def get_mp3_duration_s(s3_client, mp3_key: str) -> Optional[float]:
    """Get MP3 duration using ffprobe."""
    import subprocess
    import tempfile

    try:
        head = s3_client.head_object(Bucket=S3_BUCKET, Key=mp3_key)
        if "x-amz-meta-duration-s" in head.get("Metadata", {}):
            return float(head["Metadata"]["x-amz-meta-duration-s"])
    except Exception:
        pass

    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
            s3_client.download_file(S3_BUCKET, mp3_key, tmp.name)
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    tmp.name,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return float(result.stdout.strip())
    except Exception:
        return None


# =============================================================================
# Feature Extraction
# =============================================================================


def merge_intervals(intervals: List[Tuple[float, float]], eps: float = 1e-3) -> List[Tuple[float, float]]:
    """Merge overlapping / touching intervals."""
    if not intervals:
        return []
    sorted_ints = sorted(intervals, key=lambda x: x[0])
    merged: List[Tuple[float, float]] = []
    cur_s, cur_e = sorted_ints[0]
    for s, e in sorted_ints[1:]:
        if s <= cur_e + eps:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def point_in_intervals(t: float, intervals: List[Tuple[float, float]]) -> bool:
    """Return True if t is within any merged interval (linear scan)."""
    for s, e in intervals:
        if s <= t <= e:
            return True
    return False


class WindowTextExtractor:
    """Streaming extractor of window text from a sorted word list."""

    def __init__(self, words: List[dict]):
        self.words = sorted(words, key=lambda w: w.get("t0_abs", 0.0))
        self.i = 0

    def text_for_window(self, t_start: float, t_end: float) -> str:
        while self.i < len(self.words) and float(self.words[self.i].get("t1_abs", 0.0)) <= t_start:
            self.i += 1

        tokens: List[str] = []
        j = self.i
        while j < len(self.words) and float(self.words[j].get("t0_abs", 0.0)) < t_end:
            tok = self.words[j].get("text_norm") or self.words[j].get("text") or ""
            tok = str(tok).strip()
            if tok:
                tokens.append(tok)
            j += 1

        return " ".join(tokens)


# =============================================================================
# Audio Features (phone-band / high-frequency energy)
# =============================================================================


@dataclass
class AudioEnergyContext:
    """Precomputed prefix sums for fast per-window band-energy features."""

    sr_hz: int
    n_samples: int
    total_energy_prefix: np.ndarray  # shape [n_samples + 1]
    phone_energy_prefix: np.ndarray  # shape [n_samples + 1]
    hf_energy_prefix: np.ndarray  # shape [n_samples + 1]

    def _idx(self, t_s: float) -> int:
        return int(round(float(t_s) * self.sr_hz))

    def features_for_window(self, t_start: float, t_end: float) -> Tuple[float, float, float]:
        """Return (rms_energy, phone_band_frac, hf_energy_frac) for [t_start, t_end]."""
        i0 = max(0, min(self.n_samples, self._idx(t_start)))
        i1 = max(0, min(self.n_samples, self._idx(t_end)))
        if i1 <= i0:
            return 0.0, 0.0, 0.0

        total_e = float(self.total_energy_prefix[i1] - self.total_energy_prefix[i0])
        phone_e = float(self.phone_energy_prefix[i1] - self.phone_energy_prefix[i0])
        hf_e = float(self.hf_energy_prefix[i1] - self.hf_energy_prefix[i0])

        n = i1 - i0
        rms = math.sqrt(max(total_e / max(n, 1), 0.0))

        if total_e > 0.0:
            phone_frac = phone_e / total_e
            hf_frac = hf_e / total_e
        else:
            phone_frac = 0.0
            hf_frac = 0.0

        phone_frac = float(min(1.0, max(0.0, phone_frac)))
        hf_frac = float(min(1.0, max(0.0, hf_frac)))

        return float(rms), phone_frac, hf_frac


def load_mp3_audio_16k_from_s3(s3_client, mp3_key: str) -> np.ndarray:
    """Download MP3 from S3 and load it as 16kHz mono float32."""
    with tempfile.TemporaryDirectory() as td:
        mp3_path = Path(td) / "audio.mp3"
        s3_client.download_file(S3_BUCKET, mp3_key, str(mp3_path))
        audio = decode_audio_to_float32(str(mp3_path), sr_hz=16000)
    return audio.astype(np.float32, copy=False)


def build_audio_energy_context(audio: np.ndarray, audio_meta: Optional[Dict] = None) -> AudioEnergyContext:
    """Build AudioEnergyContext for fast per-window audio features."""
    audio_meta = audio_meta or {}

    sr = int(audio_meta.get("sr_hz", 16000))
    order = int(audio_meta.get("filter_order", 4))

    phone_band = audio_meta.get("phone_band_hz", [300.0, 3400.0])
    if isinstance(phone_band, (list, tuple)) and len(phone_band) == 2:
        phone_low = float(phone_band[0])
        phone_high = float(phone_band[1])
    else:
        phone_low, phone_high = 300.0, 3400.0

    hf_low = float(audio_meta.get("hf_low_hz", 4000.0))

    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    n = int(x.shape[0])

    total_energy_prefix = np.concatenate([[0.0], np.cumsum(x * x, dtype=np.float64)])

    sos_phone = signal.butter(order, [phone_low, phone_high], btype="bandpass", fs=sr, output="sos")
    x_phone = signal.sosfilt(sos_phone, x)
    phone_energy_prefix = np.concatenate([[0.0], np.cumsum(x_phone * x_phone, dtype=np.float64)])

    sos_hf = signal.butter(order, hf_low, btype="highpass", fs=sr, output="sos")
    x_hf = signal.sosfilt(sos_hf, x)
    hf_energy_prefix = np.concatenate([[0.0], np.cumsum(x_hf * x_hf, dtype=np.float64)])

    return AudioEnergyContext(
        sr_hz=sr,
        n_samples=n,
        total_energy_prefix=total_energy_prefix,
        phone_energy_prefix=phone_energy_prefix,
        hf_energy_prefix=hf_energy_prefix,
    )


def load_video_data_for_inference(
    s3_client,
    video_id: str,
    merge_gap_s: float = 0.2,
) -> Optional[
    Tuple[
        str,
        List[DiarizationSegment],
        List[dict],
        float,
        Optional[float],
        Optional[str],
        List[Tuple[float, float]],
    ]
]:
    """Load video data needed for inference (no labels).

    Returns:
        Tuple of (run_id, segments, words, processed_end_s, mp3_duration_s) or None if failed.
    """
    run_id = load_latest_run_id(s3_client, video_id)
    if not run_id:
        logger.error(f"No latest run for {video_id}")
        return None

    chunk_ids = list_chunks(s3_client, video_id, run_id)
    if not chunk_ids:
        logger.error(f"No chunks found for {video_id}/{run_id}")
        return None

    all_segments: List[DiarizationSegment] = []
    all_words: List[dict] = []
    processed_end_s = 0.0
    coverage_ints: List[Tuple[float, float]] = []

    for chunk_id in chunk_ids:
        meta = load_chunk_metadata(s3_client, video_id, run_id, chunk_id)
        if meta:
            chunk_end = float(meta["chunk_time_offset_s"]) + float(meta["duration_s"])
            processed_end_s = max(processed_end_s, chunk_end)
            coverage_ints.append((float(meta["chunk_time_offset_s"]), float(chunk_end)))

        all_segments.extend(load_diarization_segments(s3_client, video_id, run_id, chunk_id))
        all_words.extend(load_words(s3_client, video_id, run_id, chunk_id))

    all_segments.sort(key=lambda s: s.t0_abs)
    if merge_gap_s > 0:
        all_segments = merge_adjacent_segments(all_segments, max_gap_s=merge_gap_s)

    # Get MP3 key + duration for clamping / audio features.
    mp3_key = get_mp3_key_for_video(s3_client, video_id)
    mp3_duration_s = get_mp3_duration_s(s3_client, mp3_key) if mp3_key else None

    chunk_coverage = merge_intervals(coverage_ints)

    return run_id, all_segments, all_words, processed_end_s, mp3_duration_s, mp3_key, chunk_coverage


def generate_features_for_video(
    segments: List[DiarizationSegment],
    words: List[dict],
    timeline_end: float,
    window_config: WindowConfig,
    feature_columns: List[str],
    vectorizer: HashingVectorizer,
    *,
    s3_client=None,
    mp3_key: Optional[str] = None,
    chunk_coverage: Optional[List[Tuple[float, float]]] = None,
    audio_features_meta: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, sparse.csr_matrix]:
    """Generate numeric and text features for all windows in a video.

    Returns:
        Tuple of (DataFrame with t_mid and numeric features, sparse text feature matrix)
    """
    rows: List[Dict] = []
    window_texts: List[str] = []

    text_extractor = WindowTextExtractor(words)

    audio_ctx: Optional[AudioEnergyContext] = None
    needs_audio = any(
        c in feature_columns for c in ["rms_energy", "phone_band_frac", "hf_energy_frac"]
    )
    if needs_audio:
        if s3_client is None or not mp3_key:
            raise ValueError("Audio features required by model but mp3_key/s3_client was not provided")
        audio = load_mp3_audio_16k_from_s3(s3_client, mp3_key)
        audio_ctx = build_audio_energy_context(audio, audio_features_meta)

    for t_start, t_end, t_mid in generate_window_times(0.0, timeline_end, window_config):
        # Match training distribution: only score windows where the pipeline produced chunk artifacts.
        if chunk_coverage is not None and not point_in_intervals(t_mid, chunk_coverage):
            continue

        context_10s_start = max(0.0, t_end - window_config.context_10s)
        context_30s_start = max(0.0, t_end - window_config.context_30s)

        feats = compute_window_features(
            segments=segments,
            win_start=t_start,
            win_end=t_end,
            context_10s_start=context_10s_start,
            context_30s_start=context_30s_start,
        )

        row = {
            "t_mid": t_mid,
            **feats.to_dict(),
        }
        if audio_ctx is not None:
            rms_energy, phone_band_frac, hf_energy_frac = audio_ctx.features_for_window(t_start, t_end)
            row.update({
                "rms_energy": rms_energy,
                "phone_band_frac": phone_band_frac,
                "hf_energy_frac": hf_energy_frac,
            })
        rows.append(row)
        window_texts.append(text_extractor.text_for_window(t_start, t_end))

    df = pd.DataFrame(rows)

    # Ensure feature columns are in correct order
    X_numeric = df[feature_columns].to_numpy(dtype=np.float32)

    # Extract text features
    X_text = vectorizer.transform(window_texts).tocsr()

    return df, sparse.hstack([sparse.csr_matrix(X_numeric), X_text], format="csr")


# =============================================================================
# Segment Conversion
# =============================================================================


def probabilities_to_segments(
    t_mids: np.ndarray,
    probs: np.ndarray,
    win_s: float,
    threshold: float,
    threshold_off: Optional[float],
    gap_merge_s: float,
    min_seg_s: float,
    gap_merge_min_p: float = 0.0,
    mp3_duration_s: Optional[float] = None,
) -> List[PredictedSegment]:
    """Convert per-window probabilities to merged call segments.

    Args:
        t_mids: Window center times
        probs: Prediction probabilities
        win_s: Window duration (for converting t_mid to segment bounds)
        threshold: Classification threshold
        gap_merge_s: Merge segments with gaps smaller than this
        gap_merge_min_p: Only merge across a gap if max prob in the gap is >= this value (0 disables)
        min_seg_s: Drop segments shorter than this
        mp3_duration_s: Clamp segments to this duration (if known)

    Returns:
        List of PredictedSegment with start_s, end_s, mean_p
    """
    if len(t_mids) == 0:
        return []

    # Sort by time (should already be sorted, but be safe)
    order = np.argsort(t_mids)
    t_mids = t_mids[order]
    probs = probs[order]

    # Hysteresis thresholding: start when >= threshold, stay active while >= threshold_off.
    # When threshold_off == threshold, this reduces to single-threshold behavior.
    thr_off = threshold if threshold_off is None else float(threshold_off)
    if thr_off > threshold:
        raise ValueError(f"threshold_off ({thr_off}) must be <= threshold ({threshold})")
    gap_merge_min_p = float(gap_merge_min_p)
    if gap_merge_min_p < 0.0 or gap_merge_min_p > 1.0:
        raise ValueError("gap_merge_min_p must be in [0, 1]")

    active_mask = np.zeros_like(probs, dtype=bool)
    in_seg = False
    for i, p in enumerate(probs):
        if not in_seg:
            if p >= threshold:
                in_seg = True
                active_mask[i] = True
        else:
            if p >= thr_off:
                active_mask[i] = True
            else:
                in_seg = False

    # Build raw runs as contiguous True spans (store indices so we can conditionally merge gaps).
    half_win = win_s / 2.0
    raw_runs: List[Tuple[int, int]] = []
    start_idx: Optional[int] = None
    for i, is_pos in enumerate(active_mask):
        if is_pos and start_idx is None:
            start_idx = i
        elif not is_pos and start_idx is not None:
            raw_runs.append((start_idx, i - 1))
            start_idx = None
    if start_idx is not None:
        raw_runs.append((start_idx, len(active_mask) - 1))

    if not raw_runs:
        return []

    def run_start_s(si: int) -> float:
        return float(t_mids[si] - half_win)

    def run_end_s(ei: int) -> float:
        return float(t_mids[ei] + half_win)

    merged_runs: List[Tuple[int, int]] = []
    cur_si, cur_ei = raw_runs[0]

    for next_si, next_ei in raw_runs[1:]:
        gap_s = run_start_s(next_si) - run_end_s(cur_ei)

        should_merge = gap_s <= gap_merge_s
        if should_merge and gap_merge_min_p > 0.0:
            gap_probs = probs[cur_ei + 1: next_si]
            gap_max = float(gap_probs.max()) if gap_probs.size else 1.0
            should_merge = gap_max >= gap_merge_min_p

        if should_merge:
            cur_ei = next_ei
        else:
            merged_runs.append((cur_si, cur_ei))
            cur_si, cur_ei = next_si, next_ei

    merged_runs.append((cur_si, cur_ei))

    # Filter by minimum duration and clamp to bounds
    result: List[PredictedSegment] = []
    for si, ei in merged_runs:
        start_s = run_start_s(si)
        end_s = run_end_s(ei)
        # Clamp to [0, mp3_duration_s]
        start_s = max(0.0, start_s)
        if mp3_duration_s is not None:
            end_s = min(end_s, mp3_duration_s)

        duration = end_s - start_s
        if duration >= min_seg_s:
            # Use full span (including merged gaps) so deep dips reduce confidence.
            mean_p = float(np.mean(probs[si:ei + 1])) if ei >= si else 0.0
            result.append(PredictedSegment(
                start_s=round(start_s, 3),
                end_s=round(end_s, 3),
                mean_p=round(mean_p, 4),
            ))

    return result


# =============================================================================
# S3 Upload
# =============================================================================


def upload_prediction(s3_client, s3_prefix: str, result: PredictionResult) -> str:
    """Upload prediction result to S3."""
    s3_key = f"{s3_prefix}/{result.video_id}.json"

    payload = {
        "video_id": result.video_id,
        "run_id": result.run_id,
        "model_git_sha": result.model_git_sha,
        "threshold": result.threshold,
        "threshold_off": result.threshold_off,
        "gap_merge_s": result.gap_merge_s,
        "gap_merge_min_p": result.gap_merge_min_p,
        "min_seg_s": result.min_seg_s,
        "win_s": result.win_s,
        "hop_s": result.hop_s,
        "mp3_duration_s": result.mp3_duration_s,
        "num_windows": result.num_windows,
        "predicted_at": result.predicted_at,
        "segments": [asdict(seg) for seg in result.segments],
    }

    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=json.dumps(payload, indent=2),
        ContentType="application/json",
    )

    return f"s3://{S3_BUCKET}/{s3_key}"


# =============================================================================
# Main
# =============================================================================


def load_video_ids(video_list_path: Path, exclude_ids: Set[str]) -> List[str]:
    """Load video IDs from file, applying exclusions."""
    video_ids = []
    with open(video_list_path) as f:
        for line in f:
            vid = line.strip()
            if vid and not vid.startswith("#"):
                if vid in exclude_ids:
                    logger.info(f"Skipping excluded video: {vid}")
                else:
                    video_ids.append(vid)
    return video_ids


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Predict call segments using trained XGBoost model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video-list", type=Path, required=True,
                        help="File with video IDs (one per line)")
    parser.add_argument("--model-dir", type=Path, default=Path("data/call_segmenter/models/v1"),
                        help="Directory containing model_b.ubj and meta.json")
    parser.add_argument("--s3-out-prefix", type=str, default="call_segmenter/predictions/v1",
                        help="S3 prefix for output predictions")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Classification threshold (default: from meta.json)")
    parser.add_argument("--threshold-off", type=float, default=None,
                        help="Hysteresis: keep call active while prob >= threshold-off (default: same as --threshold)")
    parser.add_argument("--gap-merge-s", type=float, default=1.0,
                        help="Merge segments with gaps smaller than this")
    parser.add_argument("--gap-merge-min-p", type=float, default=0.0,
                        help="Only merge across gaps if max prob in the gap >= this value (0 disables)")
    parser.add_argument("--min-seg-s", type=float, default=5.0,
                        help="Drop segments shorter than this")
    parser.add_argument("--exclude-video-ids", type=str, default=None,
                        help="Comma-separated video IDs to skip")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print actions without uploading to S3")

    args = parser.parse_args()

    # Build exclusion set
    exclude_ids = set(EXCLUDED_VIDEO_IDS)
    if args.exclude_video_ids:
        exclude_ids.update(args.exclude_video_ids.split(","))

    # Load model and metadata
    try:
        model, meta = load_model_and_meta(args.model_dir)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1

    # Get parameters from meta.json
    feature_columns = meta["feature_columns"]
    text_hashing = meta["text_hashing"]
    window_config_dict = meta["window_config"]
    model_git_sha = meta.get("git_sha", "unknown")

    # Use threshold from args or meta.json
    threshold = args.threshold if args.threshold is not None else meta.get("default_threshold", 0.5)
    threshold_off = args.threshold_off
    if threshold_off is not None and threshold_off > threshold:
        logger.error(f"--threshold-off ({threshold_off}) must be <= --threshold ({threshold})")
        return 1

    # Build window config
    window_config = WindowConfig(
        win_s=window_config_dict["win_s"],
        hop_s=window_config_dict["hop_s"],
        ignore_s=window_config_dict.get("ignore_s", 0.75),
    )

    # Create text vectorizer with exact training params
    vectorizer = create_text_vectorizer(text_hashing)

    # Load video IDs
    if not args.video_list.exists():
        logger.error(f"Video list not found: {args.video_list}")
        return 1

    video_ids = load_video_ids(args.video_list, exclude_ids)
    logger.info(f"Processing {len(video_ids)} videos")

    s3 = get_s3_client()

    # Process each video
    results: List[Dict] = []
    skipped: List[str] = []

    for idx, video_id in enumerate(video_ids):
        logger.info(f"[{idx + 1}/{len(video_ids)}] {video_id}")

        # Load video data
        video_data = load_video_data_for_inference(s3, video_id)
        if video_data is None:
            logger.warning(f"  Skipping {video_id}: failed to load data")
            skipped.append(video_id)
            continue

        run_id, segments, words, processed_end_s, mp3_duration_s, mp3_key, chunk_coverage = video_data

        # Use mp3_duration_s as timeline end if available, else processed_end_s
        timeline_end = min(processed_end_s, mp3_duration_s) if mp3_duration_s else processed_end_s

        logger.info(f"  run_id={run_id} segments={len(segments)} words={len(words)} timeline_end={timeline_end:.1f}s")

        # Generate features
        df, X_combined = generate_features_for_video(
            segments=segments,
            words=words,
            timeline_end=timeline_end,
            window_config=window_config,
            feature_columns=feature_columns,
            vectorizer=vectorizer,
            s3_client=s3,
            mp3_key=mp3_key,
            chunk_coverage=chunk_coverage,
            audio_features_meta=meta.get("audio_features"),
        )

        if len(df) == 0:
            logger.warning(f"  Skipping {video_id}: no windows generated")
            skipped.append(video_id)
            continue

        # Predict
        probs = model.predict_proba(X_combined)[:, 1]

        # Convert to segments
        predicted_segments = probabilities_to_segments(
            t_mids=df["t_mid"].to_numpy(),
            probs=probs,
            win_s=window_config.win_s,
            threshold=threshold,
            threshold_off=threshold_off,
            gap_merge_s=args.gap_merge_s,
            gap_merge_min_p=args.gap_merge_min_p,
            min_seg_s=args.min_seg_s,
            mp3_duration_s=mp3_duration_s,
        )

        if threshold_off is None:
            logger.info(f"  windows={len(df)} segments={len(predicted_segments)} threshold={threshold}")
        else:
            logger.info(f"  windows={len(df)} segments={len(predicted_segments)} thr_on={threshold} thr_off={threshold_off}")

        # Build result
        result = PredictionResult(
            video_id=video_id,
            run_id=run_id,
            model_git_sha=model_git_sha,
            threshold=threshold,
            threshold_off=threshold_off,
            gap_merge_s=args.gap_merge_s,
            gap_merge_min_p=args.gap_merge_min_p,
            min_seg_s=args.min_seg_s,
            win_s=window_config.win_s,
            hop_s=window_config.hop_s,
            mp3_duration_s=mp3_duration_s,
            segments=predicted_segments,
            predicted_at=datetime.now(timezone.utc).isoformat(),
            num_windows=len(df),
        )

        # Upload or print
        if args.dry_run:
            logger.info(f"  [DRY RUN] Would upload to {args.s3_out_prefix}/{video_id}.json")
            for seg in predicted_segments[:3]:
                logger.info(f"    segment: {seg.start_s:.1f}-{seg.end_s:.1f}s (p={seg.mean_p:.3f})")
        else:
            s3_path = upload_prediction(s3, args.s3_out_prefix, result)
            logger.info(f"  Uploaded to {s3_path}")

        results.append({
            "video_id": video_id,
            "segments": len(predicted_segments),
            "windows": len(df),
        })

    # Summary
    logger.info("=" * 60)
    logger.info("Prediction Complete")
    logger.info(f"  Processed: {len(results)} videos")
    logger.info(f"  Skipped: {len(skipped)} videos")
    if skipped:
        logger.info(f"  Skipped IDs: {', '.join(skipped[:10])}")
    total_segs = sum(r["segments"] for r in results)
    logger.info(f"  Total segments predicted: {total_segs}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
