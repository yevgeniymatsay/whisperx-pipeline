#!/usr/bin/env python3
"""Build v2 call-segmentation training dataset (role-agnostic).

This script aligns:
  - WhisperX diarization (runs/.../chunks/.../diarization_segments.json)
  - WhisperX ASR words (runs/.../chunks/.../words.json)
  - Human call boundaries (labeling/corrected_boundaries/v1/*.json)

Into a windowed dataset for training a call-vs-not-call classifier.

Key properties (v2):
  - NO host/nonhost assumptions (host_detector.py removed).
  - NO leaky pipeline heuristics (chunk_content_type is not written).
  - mp3_duration_s (ffprobe) is treated as the truth for QC checks, but windows are
    only emitted where the pipeline actually produced chunk artifacts.
  - Optional hashed character n-gram features are saved separately as a sparse matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import boto3
from sklearn.feature_extraction.text import HashingVectorizer

from scipy import sparse

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import S3_BUCKET, AWS_REGION
from pipeline.audio_preprocess import decode_audio_stream_to_float32
from pipeline.transcriber import DiarizationSegment
from pipeline.call_segmenter.features import merge_adjacent_segments, compute_window_features
from pipeline.call_segmenter.audio_features import AudioFeatureConfig, compute_audio_features_for_windows
from pipeline.call_segmenter.window_generator import WindowConfig, CallBoundary, assign_labels, generate_window_times

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ChunkMetadata:
    chunk_id: str
    start_abs: float
    end_abs: float
    duration_s: float


@dataclass
class VideoData:
    video_id: str
    run_id: str
    mp3_key: str
    creator_id: str
    chunks: List[ChunkMetadata]
    segments: List[DiarizationSegment]
    words: List[dict]
    call_boundaries: List[CallBoundary]
    processed_end_s: float
    mp3_duration_s: Optional[float] = None
    chunk_coverage: List[Tuple[float, float]] = field(default_factory=list)  # merged intervals


@dataclass
class DatasetConfig:
    win_s: float = 1.0
    hop_s: float = 0.5
    ignore_s: float = 0.75
    context_10s: float = 10.0
    context_30s: float = 30.0
    merge_gap_s: float = 0.2
    processed_end_tolerance_s: float = 2.0
    exclude_drift_above_s: Optional[float] = None

    # Text feature config (hashed n-grams)
    include_text: bool = True
    text_n_features: int = 2**12
    text_ngram_min: int = 2
    text_ngram_max: int = 5
    text_context_s: float = 0.0
    text_max_chars: int = 300

    # Audio-derived features (computed from the source MP3; role-agnostic)
    include_audio_features: bool = True
    audio_spectral_features: bool = True
    audio_mfcc: bool = False
    audio_sr_hz: int = 16000
    audio_filter_order: int = 4
    audio_phone_low_hz: float = 300.0
    audio_phone_high_hz: float = 3400.0
    audio_hf_low_hz: float = 4000.0


@dataclass
class VideoReport:
    video_id: str
    run_id: str
    creator_id: str
    mp3_key: str
    windows_total: int
    windows_emitted: int
    windows_dropped_uncovered: int
    calls: int
    class_balance_in_call: float
    ignored_windows: int
    mp3_duration_s: Optional[float]
    processed_end_s: float
    timebase_drift_s: Optional[float]
    call_sanity: List[Dict]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# Helpers (intervals, QC, text)
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
    """Check if t is within any merged interval (linear scan; intervals are small)."""
    for s, e in intervals:
        if s <= t <= e:
            return True
    return False


def interval_overlap_duration(a_start: float, a_end: float, intervals: List[Tuple[float, float]]) -> float:
    """Total overlap duration of [a_start,a_end] against a set of merged intervals."""
    if a_end <= a_start:
        return 0.0
    total = 0.0
    for s, e in intervals:
        o_s = max(a_start, s)
        o_e = min(a_end, e)
        if o_e > o_s:
            total += o_e - o_s
    return total


def union_speech_duration_in_range(
    segments: List[DiarizationSegment],
    t_start: float,
    t_end: float,
) -> float:
    """Union of diarization segment durations (ignoring speaker identity) within [t_start,t_end]."""
    if t_end <= t_start:
        return 0.0
    clipped: List[Tuple[float, float]] = []
    for s in segments:
        if s.t1_abs <= t_start or s.t0_abs >= t_end:
            continue
        clipped.append((max(t_start, s.t0_abs), min(t_end, s.t1_abs)))
    merged = merge_intervals(clipped)
    return sum(e - s for s, e in merged)


class WindowTextExtractor:
    """Streaming extractor of window text from a sorted word list."""

    def __init__(self, words: List[dict], *, context_s: float = 0.0, max_chars: int = 300):
        self.words = sorted(words, key=lambda w: w.get("t0_abs", 0.0))
        self.i = 0
        self.context_s = float(context_s)
        self.max_chars = int(max_chars)

    def text_for_window(self, t_start: float, t_end: float) -> str:
        span_start = float(t_start) - self.context_s
        span_end = float(t_end) + self.context_s
        # Advance start pointer to the first word that might overlap this window.
        while self.i < len(self.words) and float(self.words[self.i].get("t1_abs", 0.0)) <= span_start:
            self.i += 1

        tokens: List[str] = []
        j = self.i
        while j < len(self.words) and float(self.words[j].get("t0_abs", 0.0)) < span_end:
            tok = self.words[j].get("text_norm") or self.words[j].get("text") or ""
            tok = str(tok).strip()
            if tok:
                tokens.append(tok)
            j += 1

        text = " ".join(tokens)
        if self.max_chars and len(text) > self.max_chars:
            text = text[: self.max_chars]
        return text


def load_mp3_audio_16k_from_s3(
    s3_client, mp3_key: str, *, max_duration_s: Optional[float] = None, sr_hz: int = 16000
) -> np.ndarray:
    """Download MP3 from S3 and decode it to mono float32 PCM.

    Args:
        max_duration_s: if provided, truncate audio to this many seconds.
    """
    resp = s3_client.get_object(Bucket=S3_BUCKET, Key=mp3_key)
    audio = decode_audio_stream_to_float32(
        resp["Body"],
        sr_hz=int(sr_hz),
        max_duration_s=max_duration_s,
    ).astype(np.float32, copy=False)
    return audio


# =============================================================================
# S3 Loading
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
            # runs/{video_id}/{run_id}/chunks/{chunk_id}/...
            if len(parts) >= 5:
                chunk_ids.add(parts[4])
    return sorted(chunk_ids)


def load_chunk_metadata(s3_client, video_id: str, run_id: str, chunk_id: str) -> Optional[ChunkMetadata]:
    key = f"runs/{video_id}/{run_id}/chunks/{chunk_id}/chunk_metadata.json"
    try:
        resp = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        data = json.loads(resp["Body"].read())
        start = float(data["chunk_time_offset_s"])
        dur = float(data["duration_s"])
        return ChunkMetadata(
            chunk_id=chunk_id,
            start_abs=start,
            end_abs=start + dur,
            duration_s=dur,
        )
    except Exception as e:
        logger.warning(f"Error loading chunk metadata {video_id}/{run_id}/{chunk_id}: {e}")
        return None


def load_diarization_segments(s3_client, video_id: str, run_id: str, chunk_id: str) -> List[DiarizationSegment]:
    key = f"runs/{video_id}/{run_id}/chunks/{chunk_id}/diarization_segments.json"
    try:
        resp = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        data = json.loads(resp["Body"].read())
        out: List[DiarizationSegment] = []
        for seg in data.get("segments", []):
            out.append(DiarizationSegment(
                t0_abs=float(seg["t0_abs"]),
                t1_abs=float(seg["t1_abs"]),
                spk=str(seg["spk"]),
            ))
        return out
    except Exception:
        return []


def load_words(s3_client, video_id: str, run_id: str, chunk_id: str) -> List[dict]:
    key = f"runs/{video_id}/{run_id}/chunks/{chunk_id}/words.json"
    try:
        resp = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        data = json.loads(resp["Body"].read())
        words = data.get("words", [])
        # Ensure we have numeric times for sorting.
        for w in words:
            if "t0_abs" in w:
                w["t0_abs"] = float(w["t0_abs"])
            if "t1_abs" in w:
                w["t1_abs"] = float(w["t1_abs"])
        return words
    except Exception:
        return []


def extract_creator_id(mp3_key: str) -> str:
    """Extract creator_id from mp3_key: audio/pretraining/{creator} - {video_id}.mp3"""
    filename = mp3_key.split("/")[-1]
    match = re.match(r"(.+?) - ", filename)
    if match:
        return match.group(1).strip()
    return filename.replace(".mp3", "")


def get_mp3_key_for_video(s3_client, video_id: str) -> Optional[str]:
    """Find the mp3 key for a video by listing audio/pretraining/."""
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix="audio/pretraining/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if video_id in key and key.endswith(".mp3"):
                return key
    return None


def get_mp3_duration_s(s3_client, mp3_key: str) -> Optional[float]:
    """Get MP3 duration using ffprobe (downloads mp3 to a temp file)."""
    import tempfile

    try:
        head = s3_client.head_object(Bucket=S3_BUCKET, Key=mp3_key)
        md = head.get("Metadata") or {}
        # boto3 exposes user metadata as lowercased keys in "Metadata" without the x-amz-meta- prefix.
        # We keep a couple fallbacks for older uploads / mismatched conventions.
        for key in ["duration-s", "duration_s", "x-amz-meta-duration-s"]:
            if key in md:
                try:
                    return float(md[key])
                except Exception:
                    break
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
    except FileNotFoundError:
        logger.warning("ffprobe not found; mp3_duration_s QC will be skipped")
        return None
    except Exception as e:
        logger.warning(f"Could not determine mp3 duration for {mp3_key}: {e}")
        return None


# =============================================================================
# Labels
# =============================================================================


def load_labels_json(path: Path) -> Dict[str, List[CallBoundary]]:
    with open(path) as f:
        data = json.load(f)
    labels: Dict[str, List[CallBoundary]] = {}
    for video in data:
        video_id = video["video_id"]
        labels[video_id] = [CallBoundary(start=c["start"], end=c["end"]) for c in video.get("calls", [])]
    return labels


def load_labels_csv(path: Path) -> Dict[str, List[CallBoundary]]:
    labels: Dict[str, List[CallBoundary]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row["video_id"]
            labels.setdefault(video_id, []).append(CallBoundary(start=float(row["start_s"]), end=float(row["end_s"])))
    for vid in labels:
        labels[vid].sort(key=lambda b: b.start)
    return labels


def load_labels_s3(s3_client, prefix: str = "labeling/corrected_boundaries/v1/") -> Dict[str, List[CallBoundary]]:
    labels: Dict[str, List[CallBoundary]] = {}
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            video_id = key[len(prefix):-5]
            if not video_id:
                continue
            try:
                resp = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
                doc = json.loads(resp["Body"].read())
                boundaries = [CallBoundary(start=float(b["start_s"]), end=float(b["end_s"])) for b in doc.get("boundaries", [])]
                boundaries.sort(key=lambda b: b.start)
                # Keep empty boundaries too: they are important "no-call" negatives.
                labels[video_id] = boundaries
            except Exception as e:
                logger.warning(f"Error loading boundaries for {video_id}: {e}")
    return labels


def validate_labels(labels: Dict[str, List[CallBoundary]]) -> List[str]:
    issues: List[str] = []
    for video_id, boundaries in labels.items():
        for i, b in enumerate(boundaries):
            if b.start >= b.end:
                issues.append(f"{video_id}: call {i} has start >= end ({b.start} >= {b.end})")
            if i < len(boundaries) - 1 and b.end > boundaries[i + 1].start:
                issues.append(f"{video_id}: calls {i} and {i+1} overlap ({b.end} > {boundaries[i+1].start})")
    return issues


# =============================================================================
# Core processing
# =============================================================================


def load_video_data(
    s3_client,
    video_id: str,
    labels: Dict[str, List[CallBoundary]],
    config: DatasetConfig,
) -> Optional[VideoData]:
    run_id = load_latest_run_id(s3_client, video_id)
    if not run_id:
        logger.error(f"No latest run for {video_id}")
        return None

    mp3_key = get_mp3_key_for_video(s3_client, video_id) or f"audio/pretraining/unknown - {video_id}.mp3"
    creator_id = extract_creator_id(mp3_key)

    mp3_duration_s = get_mp3_duration_s(s3_client, mp3_key) if mp3_key else None

    chunk_ids = list_chunks(s3_client, video_id, run_id)
    if not chunk_ids:
        logger.error(f"No chunks found for {video_id}/{run_id}")
        return None

    chunks: List[ChunkMetadata] = []
    coverage_ints: List[Tuple[float, float]] = []
    all_segments: List[DiarizationSegment] = []
    all_words: List[dict] = []

    for chunk_id in chunk_ids:
        meta = load_chunk_metadata(s3_client, video_id, run_id, chunk_id)
        if meta:
            chunks.append(meta)
            coverage_ints.append((meta.start_abs, meta.end_abs))

        all_segments.extend(load_diarization_segments(s3_client, video_id, run_id, chunk_id))
        all_words.extend(load_words(s3_client, video_id, run_id, chunk_id))

    chunks.sort(key=lambda c: c.start_abs)
    all_segments.sort(key=lambda s: s.t0_abs)

    if config.merge_gap_s > 0:
        all_segments = merge_adjacent_segments(all_segments, max_gap_s=config.merge_gap_s)

    processed_end_s = max((c.end_abs for c in chunks), default=0.0)
    chunk_coverage = merge_intervals(coverage_ints)

    return VideoData(
        video_id=video_id,
        run_id=run_id,
        mp3_key=mp3_key,
        creator_id=creator_id,
        chunks=chunks,
        segments=all_segments,
        words=sorted(all_words, key=lambda w: float(w.get("t0_abs", 0.0))),
        call_boundaries=labels.get(video_id, []),
        processed_end_s=processed_end_s,
        mp3_duration_s=mp3_duration_s,
        chunk_coverage=chunk_coverage,
    )


def process_video(
    s3_client,
    video_data: VideoData,
    config: DatasetConfig,
    vectorizer: Optional[HashingVectorizer],
) -> Tuple[pd.DataFrame, Optional[sparse.csr_matrix], Optional[np.ndarray], VideoReport]:
    """Generate window rows and optional text features for a video."""
    errors: List[str] = []
    warnings: List[str] = []

    # QC: label bounds vs mp3 + processed range
    eps = 1e-3
    if video_data.mp3_duration_s is not None:
        for i, b in enumerate(video_data.call_boundaries):
            if b.start < -eps or b.end > video_data.mp3_duration_s + eps:
                warnings.append(
                    f"label_outside_mp3: call {i} {b.start:.3f}-{b.end:.3f} (mp3={video_data.mp3_duration_s:.3f}s)"
                )

    for i, b in enumerate(video_data.call_boundaries):
        if b.start > video_data.processed_end_s + config.processed_end_tolerance_s:
            errors.append(
                f"label_after_processed_end: call {i} starts {b.start:.3f}s > processed_end {video_data.processed_end_s:.3f}s"
            )
        elif b.end > video_data.processed_end_s + config.processed_end_tolerance_s:
            errors.append(
                f"label_end_beyond_processed_end: call {i} ends {b.end:.3f}s > processed_end {video_data.processed_end_s:.3f}s"
            )

    # Timebase drift (QC)
    if video_data.mp3_duration_s is not None:
        timebase_drift_s = abs(video_data.processed_end_s - video_data.mp3_duration_s)
    else:
        timebase_drift_s = None

    # Exclude for drift if requested.
    if config.exclude_drift_above_s is not None and timebase_drift_s is not None:
        if timebase_drift_s > config.exclude_drift_above_s:
            errors.append(f"timebase_drift_excluded: drift={timebase_drift_s:.3f}s > {config.exclude_drift_above_s:.3f}s")

    # Generate window rows.
    window_config = WindowConfig(
        win_s=config.win_s,
        hop_s=config.hop_s,
        ignore_s=config.ignore_s,
        context_10s=config.context_10s,
        context_30s=config.context_30s,
    )

    timeline_end = video_data.processed_end_s
    if video_data.mp3_duration_s is not None:
        # processed_end can slightly exceed mp3 duration due to padding/rounding
        timeline_end = min(timeline_end, video_data.mp3_duration_s)

    rows: List[Dict] = []
    window_texts: List[str] = []

    dropped_uncovered = 0
    total_windows = 0

    text_extractor = (
        WindowTextExtractor(
            video_data.words,
            context_s=config.text_context_s,
            max_chars=config.text_max_chars,
        )
        if (config.include_text and vectorizer)
        else None
    )

    for t_start, t_end, t_mid in generate_window_times(0.0, timeline_end, window_config):
        total_windows += 1

        if not point_in_intervals(t_mid, video_data.chunk_coverage):
            dropped_uncovered += 1
            continue

        context_10s_start = max(0.0, t_end - window_config.context_10s)
        context_30s_start = max(0.0, t_end - window_config.context_30s)

        feats = compute_window_features(
            segments=video_data.segments,
            win_start=t_start,
            win_end=t_end,
            context_10s_start=context_10s_start,
            context_30s_start=context_30s_start,
        )

        y, ignore, dist = assign_labels(t_mid, video_data.call_boundaries, window_config.ignore_s)

        row = {
            "t_start": t_start,
            "t_end": t_end,
            "t_mid": t_mid,
            "y": y,
            "ignore": ignore,
            "dist_to_boundary_s": dist,
            **feats.to_dict(),
            "video_id": video_data.video_id,
            "run_id": video_data.run_id,
            "creator_id": video_data.creator_id,
            "mp3_key": video_data.mp3_key,
        }
        rows.append(row)

        if text_extractor is not None:
            window_texts.append(text_extractor.text_for_window(t_start, t_end))

    df = pd.DataFrame(rows)

    # Add audio-derived features (computed once per video).
    if config.include_audio_features and not df.empty:
        try:
            audio = load_mp3_audio_16k_from_s3(
                s3_client,
                video_data.mp3_key,
                max_duration_s=timeline_end,
                sr_hz=config.audio_sr_hz,
            )
            audio_cfg = AudioFeatureConfig(
                enabled=True,
                sr_hz=int(config.audio_sr_hz),
                filter_order=int(config.audio_filter_order),
                spectral_enabled=bool(config.audio_spectral_features),
                mfcc_enabled=bool(config.audio_mfcc),
                phone_low_hz=float(config.audio_phone_low_hz),
                phone_high_hz=float(config.audio_phone_high_hz),
                hf_low_hz=float(config.audio_hf_low_hz),
            )
            audio_cols = [
                "rms_energy",
                "log_rms",
                "log_rms_z",
                "zcr",
                "low_band_frac",
                "phone_band_frac",
                "hf_energy_frac",
                "mid_hf_band_frac",
                "ultra_hf_frac",
                "hi_ratio_3p5_7k",
            ]
            if config.audio_spectral_features:
                audio_cols.extend([
                    "spec_centroid_hz",
                    "spec_centroid_z",
                    "spec_rolloff_hz",
                    "spec_rolloff_z",
                ])
            if config.audio_mfcc:
                for k in range(13):
                    audio_cols.append(f"mfcc_{k:02d}")
                for k in range(13):
                    audio_cols.append(f"mfcc_z_{k:02d}")
            feats = compute_audio_features_for_windows(
                audio=audio,
                t_starts=df["t_start"].to_numpy(),
                t_ends=df["t_end"].to_numpy(),
                config=audio_cfg,
                required_columns=audio_cols,
            )
            for c in audio_cols:
                df[c] = feats[c]
        except Exception as e:
            errors.append(f"audio_features_failed: mp3_key={video_data.mp3_key} err={e}")
            # Without audio features, the dataset would have missing feature columns; skip this video.
            df = pd.DataFrame()

    if df.empty:
        warnings.append(f"no_emitted_windows: timeline_end={timeline_end:.3f}s")

    # Call sanity (objective; no role assumptions)
    call_sanity: List[Dict] = []
    for i, b in enumerate(video_data.call_boundaries):
        core_start = b.start + config.ignore_s
        core_end = b.end - config.ignore_s
        core_dur = max(0.0, core_end - core_start)

        if core_dur > 0 and not df.empty:
            core_window_count = int(((df["t_mid"] >= core_start) & (df["t_mid"] <= core_end)).sum())
        else:
            core_window_count = 0

        speech_dur = union_speech_duration_in_range(video_data.segments, core_start, core_end) if core_dur > 0 else 0.0
        speech_frac = (speech_dur / core_dur) if core_dur > 0 else 0.0

        coverage_dur = interval_overlap_duration(core_start, core_end, video_data.chunk_coverage) if core_dur > 0 else 0.0
        coverage_frac = (coverage_dur / core_dur) if core_dur > 0 else 0.0

        stat = {
            "call_index": i,
            "start": b.start,
            "end": b.end,
            "duration_s": b.end - b.start,
            "core_duration_s": core_dur,
            "core_windows_emitted": core_window_count,
            "core_chunk_coverage_frac": round(coverage_frac, 3) if core_dur > 0 else None,
            "core_speech_frac": round(speech_frac, 3) if core_dur > 0 else None,
        }
        call_sanity.append(stat)

        # Warning-only: a core with zero windows means we won't learn from the interior.
        if core_dur > 0 and core_window_count == 0:
            warnings.append(
                f"call_core_no_windows: call {i} core={core_start:.3f}-{core_end:.3f} (dur={core_dur:.2f}s)"
            )

    # Video stats
    in_call = int((df["y"] == 1).sum()) if not df.empty else 0
    ignored = int((df["ignore"] == 1).sum()) if not df.empty else 0

    report = VideoReport(
        video_id=video_data.video_id,
        run_id=video_data.run_id,
        creator_id=video_data.creator_id,
        mp3_key=video_data.mp3_key,
        windows_total=total_windows,
        windows_emitted=len(df),
        windows_dropped_uncovered=dropped_uncovered,
        calls=len(video_data.call_boundaries),
        class_balance_in_call=(in_call / len(df)) if len(df) > 0 else 0.0,
        ignored_windows=ignored,
        mp3_duration_s=video_data.mp3_duration_s,
        processed_end_s=video_data.processed_end_s,
        timebase_drift_s=timebase_drift_s,
        call_sanity=call_sanity,
        errors=errors,
        warnings=warnings,
    )

    # Build text sparse matrix (same order as df rows)
    X_text = None
    row_ids = None
    if config.include_text and vectorizer is not None:
        if len(window_texts) != len(df):
            # Should never happen; defensive.
            errors.append(f"text_row_mismatch: texts={len(window_texts)} rows={len(df)}")
        else:
            X_text = vectorizer.transform(window_texts).tocsr()

    return df, X_text, row_ids, report


def get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            check=True,
        )
        return result.stdout.strip()[:12]
    except Exception:
        return "unknown"


def build_dataset_metadata(config: DatasetConfig) -> Dict:
    audio_cfg = AudioFeatureConfig(
        enabled=bool(config.include_audio_features),
        sr_hz=int(config.audio_sr_hz),
        filter_order=int(config.audio_filter_order),
        spectral_enabled=bool(config.audio_spectral_features),
        mfcc_enabled=bool(config.audio_mfcc),
        phone_low_hz=float(config.audio_phone_low_hz),
        phone_high_hz=float(config.audio_phone_high_hz),
        hf_low_hz=float(config.audio_hf_low_hz),
    )
    meta = {
        "version": "v8",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_sha(),
        "config": {
            "win_s": config.win_s,
            "hop_s": config.hop_s,
            "ignore_s": config.ignore_s,
            "context_10s": config.context_10s,
            "context_30s": config.context_30s,
            "merge_gap_s": config.merge_gap_s,
            "processed_end_tolerance_s": config.processed_end_tolerance_s,
            "exclude_drift_above_s": config.exclude_drift_above_s,
            "audio_features": audio_cfg.to_meta(),
        },
        "non_feature_columns": [
            "row_id",
            "video_id",
            "run_id",
            "creator_id",
            "mp3_key",
            "t_start",
            "t_end",
            "t_mid",
            "y",
            "ignore",
            "dist_to_boundary_s",
        ],
        "feature_columns": [
            "speech_frac",
            "num_active_speakers",
            "speaker_switches_10s",
            "dominant_speech_frac",
            "speech_entropy",
            "unique_speakers_30s",
            "turns_30s",
            "avg_turn_len_30s",
        ],
        "group_split_column": "video_id",
        "text_features": None,
    }

    if config.include_audio_features:
        meta["feature_columns"].extend([
            "rms_energy",
            "log_rms",
            "log_rms_z",
            "zcr",
            "low_band_frac",
            "phone_band_frac",
            "hf_energy_frac",
            "mid_hf_band_frac",
            "ultra_hf_frac",
            "hi_ratio_3p5_7k",
        ])
        if config.audio_spectral_features:
            meta["feature_columns"].extend([
                "spec_centroid_hz",
                "spec_centroid_z",
                "spec_rolloff_hz",
                "spec_rolloff_z",
            ])
        if config.audio_mfcc:
            meta["feature_columns"].extend([f"mfcc_{k:02d}" for k in range(13)])
            meta["feature_columns"].extend([f"mfcc_z_{k:02d}" for k in range(13)])

    if config.include_text:
        meta["text_features"] = {
            "type": "hashing_char_ngrams",
            "n_features": config.text_n_features,
            "ngram_range": [config.text_ngram_min, config.text_ngram_max],
            "analyzer": "char_wb",
            "context_s": float(config.text_context_s),
            "max_chars": int(config.text_max_chars),
        }

    return meta


def build_report(video_reports: Dict[str, VideoReport], config: DatasetConfig) -> Dict:
    total_emitted = sum(r.windows_emitted for r in video_reports.values())
    total_total = sum(r.windows_total for r in video_reports.values())
    total_dropped = sum(r.windows_dropped_uncovered for r in video_reports.values())
    total_ignored = sum(r.ignored_windows for r in video_reports.values())

    # Aggregate class balance from per-video in-call fraction.
    total_in_call = sum(r.windows_emitted * r.class_balance_in_call for r in video_reports.values())

    anomalies: Dict[str, List[str]] = {
        "label_outside_mp3": [],
        "label_beyond_processed_end": [],
        "call_core_no_windows": [],
        "uncovered_window_rows_dropped": [],
        "timebase_drift": [],
    }

    for vid, rep in video_reports.items():
        for w in rep.warnings:
            if w.startswith("label_outside_mp3:"):
                anomalies["label_outside_mp3"].append(f"{vid}:{w}")
            if w.startswith("call_core_no_windows:"):
                anomalies["call_core_no_windows"].append(f"{vid}:{w}")

        for e in rep.errors:
            if e.startswith("label_after_processed_end:") or e.startswith("label_end_beyond_processed_end:"):
                anomalies["label_beyond_processed_end"].append(f"{vid}:{e}")
            if e.startswith("timebase_drift_excluded:"):
                anomalies["timebase_drift"].append(f"{vid}:{e}")

        if rep.windows_dropped_uncovered > 0:
            anomalies["uncovered_window_rows_dropped"].append(
                f"{vid}:{rep.windows_dropped_uncovered}/{rep.windows_total}"
            )

        if rep.timebase_drift_s is not None and config.exclude_drift_above_s is None:
            # Only report drift as an anomaly when not used as a hard filter.
            if rep.timebase_drift_s > 2.0:
                anomalies["timebase_drift"].append(f"{vid}:drift={rep.timebase_drift_s:.3f}s")

    return {
        "total_videos": len(video_reports),
        "total_windows_emitted": total_emitted,
        "total_windows_possible": total_total,
        "windows_dropped_uncovered": total_dropped,
        "ignored_windows": total_ignored,
        "class_balance": {
            "in_call": round(total_in_call / total_emitted, 4) if total_emitted else 0.0,
            "out_of_call": round(1.0 - (total_in_call / total_emitted), 4) if total_emitted else 0.0,
        },
        "videos": {
            vid: {
                "windows_total": r.windows_total,
                "windows_emitted": r.windows_emitted,
                "windows_dropped_uncovered": r.windows_dropped_uncovered,
                "calls": r.calls,
                "ignored_windows": r.ignored_windows,
                "mp3_duration_s": r.mp3_duration_s,
                "processed_end_s": r.processed_end_s,
                "timebase_drift_s": r.timebase_drift_s,
                "call_sanity": r.call_sanity,
                "errors": r.errors,
                "warnings": r.warnings,
            }
            for vid, r in video_reports.items()
        },
        "anomalies": anomalies,
    }


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build v2 call-segmenter training dataset (role-agnostic)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--labels", type=Path, help="Path to local labels file (labels.json or corrected_boundaries.csv)")
    parser.add_argument("--labels-s3", action="store_true", help="Load labels from S3 boundary store")
    parser.add_argument("--labels-s3-prefix", type=str, default="labeling/corrected_boundaries/v1/")
    parser.add_argument("--output-dir", type=Path, default=Path("data/call_segmenter/v2"))
    parser.add_argument("--win-s", type=float, default=1.0)
    parser.add_argument("--hop-s", type=float, default=0.5)
    parser.add_argument("--ignore-s", type=float, default=0.75)
    parser.add_argument("--exclude-drift-above", type=float, default=None, help="Exclude videos with |processed_end - mp3_duration| above this threshold")
    parser.add_argument("--processed-end-tolerance-s", type=float, default=2.0, help="Tolerance for label end beyond processed_end before hard error")
    video_sel = parser.add_mutually_exclusive_group()
    video_sel.add_argument("--video-ids", type=str, nargs="*", help="Process only these video IDs")
    video_sel.add_argument("--video-list", type=Path, default=None,
                           help="Process only video IDs listed in this file (one per line). "
                                "Recommended when IDs may start with '-'")
    parser.add_argument("--no-text", action="store_true", help="Skip hashed text feature extraction")
    parser.add_argument("--no-audio-features", action="store_true", help="Skip audio-derived band-energy features")
    parser.add_argument("--no-audio-spectral-features", action="store_true",
                        help="Disable spectral centroid/rolloff features (still computes band-energy features unless --no-audio-features)")
    parser.add_argument("--audio-mfcc", action="store_true",
                        help="Include MFCC(13)+z-score features (slower; requires band-energy audio decode)")
    parser.add_argument("--text-n-features", type=int, default=2**12)
    parser.add_argument("--text-ngram-min", type=int, default=2)
    parser.add_argument("--text-ngram-max", type=int, default=5)
    parser.add_argument("--text-context-s", type=float, default=0.0,
                        help="Extract transcript text from [t_start-context, t_end+context] for each window")
    parser.add_argument("--text-max-chars", type=int, default=300,
                        help="Cap concatenated transcript snippet length (chars) per window")
    parser.add_argument("--dry-run", action="store_true", help="Load/process but do not write outputs")

    args = parser.parse_args()

    cfg = DatasetConfig(
        win_s=args.win_s,
        hop_s=args.hop_s,
        ignore_s=args.ignore_s,
        exclude_drift_above_s=args.exclude_drift_above,
        processed_end_tolerance_s=args.processed_end_tolerance_s,
        include_text=not args.no_text,
        text_n_features=args.text_n_features,
        text_ngram_min=args.text_ngram_min,
        text_ngram_max=args.text_ngram_max,
        text_context_s=args.text_context_s,
        text_max_chars=args.text_max_chars,
        include_audio_features=not args.no_audio_features,
        audio_spectral_features=not args.no_audio_spectral_features,
        audio_mfcc=bool(args.audio_mfcc),
    )

    s3 = get_s3_client()

    # Load labels
    if args.labels_s3:
        logger.info(f"Loading labels from S3: s3://{S3_BUCKET}/{args.labels_s3_prefix}")
        labels = load_labels_s3(s3, prefix=args.labels_s3_prefix)
    elif args.labels:
        logger.info(f"Loading labels from {args.labels}")
        if args.labels.suffix == ".json":
            labels = load_labels_json(args.labels)
        elif args.labels.suffix == ".csv":
            labels = load_labels_csv(args.labels)
        else:
            raise ValueError(f"Unsupported label format: {args.labels}")
    else:
        logger.error("Must provide --labels or --labels-s3")
        return 2

    logger.info(f"Loaded labels for {len(labels)} videos")

    issues = validate_labels(labels)
    if issues:
        logger.warning(f"Found {len(issues)} label issues (showing first 10):")
        for i in issues[:10]:
            logger.warning(f"  {i}")

    if args.video_list:
        if not args.video_list.exists():
            raise FileNotFoundError(f"--video-list not found: {args.video_list}")
        video_ids = []
        for line in args.video_list.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            video_ids.append(line)
    else:
        video_ids = args.video_ids if args.video_ids else sorted(labels.keys())
    logger.info(f"Processing {len(video_ids)} videos")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    vectorizer = None
    if cfg.include_text:
        vectorizer = HashingVectorizer(
            n_features=cfg.text_n_features,
            analyzer="char_wb",
            ngram_range=(cfg.text_ngram_min, cfg.text_ngram_max),
            alternate_sign=False,
            norm=None,
            lowercase=True,
        )

    all_dfs: List[pd.DataFrame] = []
    text_mats: List[sparse.csr_matrix] = []
    text_row_ids: List[np.ndarray] = []
    video_reports: Dict[str, VideoReport] = {}
    index_entries: List[Dict] = []

    next_row_id = 0

    for idx, video_id in enumerate(video_ids):
        logger.info(f"[{idx+1}/{len(video_ids)}] {video_id}")

        vd = load_video_data(s3, video_id, labels, cfg)
        if vd is None:
            logger.error(f"  Failed to load video data for {video_id}")
            continue

        logger.info(
            f"  Loaded chunks={len(vd.chunks)} segs={len(vd.segments)} words={len(vd.words)} calls={len(vd.call_boundaries)}"
        )

        df, X_text, _, rep = process_video(s3, vd, cfg, vectorizer)

        # Hard exclude on errors (timebase/label beyond processed end, etc.)
        if rep.errors:
            logger.warning(f"  EXCLUDING video due to errors: {rep.errors[:3]}")
            video_reports[video_id] = rep
            continue

        # Assign row_id and append.
        if not df.empty:
            df.insert(0, "row_id", np.arange(next_row_id, next_row_id + len(df), dtype=np.int64))
            row_ids = df["row_id"].to_numpy()
            next_row_id += len(df)

            if X_text is not None:
                text_mats.append(X_text)
                text_row_ids.append(row_ids)

        all_dfs.append(df)
        video_reports[video_id] = rep

        index_entries.append({
            "video_id": vd.video_id,
            "run_id": vd.run_id,
            "creator_id": vd.creator_id,
            "mp3_key": vd.mp3_key,
            "windows_emitted": int(len(df)),
            "calls": int(len(vd.call_boundaries)),
        })

        logger.info(
            f"  windows_emitted={len(df)} dropped_uncovered={rep.windows_dropped_uncovered}/{rep.windows_total} "
            f"in_call={rep.class_balance_in_call:.1%} ignored={rep.ignored_windows}"
        )

        if not args.dry_run and not df.empty:
            out_path = args.output_dir / f"windows_{video_id}.parquet"
            df.to_parquet(out_path, index=False)

    if not all_dfs:
        logger.error("No data generated")
        return 1

    combined_df = pd.concat(all_dfs, ignore_index=True)

    if not args.dry_run:
        combined_path = args.output_dir / "windows_all.parquet"
        combined_df.to_parquet(combined_path, index=False)
        logger.info(f"Wrote {combined_path}")

        index_path = args.output_dir / "index.jsonl"
        with open(index_path, "w") as f:
            for entry in index_entries:
                f.write(json.dumps(entry) + "\n")
        logger.info(f"Wrote {index_path}")

        report = build_report(video_reports, cfg)
        report_path = args.output_dir / "report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        logger.info(f"Wrote {report_path}")

        meta = build_dataset_metadata(cfg)
        meta_path = args.output_dir / "dataset_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, cls=NumpyEncoder)
        logger.info(f"Wrote {meta_path}")

        # Save text features (sparse) if requested.
        if cfg.include_text and text_mats:
            X_all = sparse.vstack(text_mats).tocsr()
            row_ids_all = np.concatenate(text_row_ids) if text_row_ids else np.array([], dtype=np.int64)

            # Sort by row_id to make join deterministic.
            order = np.argsort(row_ids_all)
            X_all = X_all[order]
            row_ids_all = row_ids_all[order]

            sparse.save_npz(args.output_dir / "text_features.npz", X_all)
            np.save(args.output_dir / "text_row_ids.npy", row_ids_all)
            logger.info(f"Wrote {args.output_dir / 'text_features.npz'} and {args.output_dir / 'text_row_ids.npy'}")

    # Summary
    logger.info("=" * 60)
    logger.info("Dataset Build Complete (v2)")
    logger.info(f"  Total videos (included): {sum(1 for r in video_reports.values() if not r.errors)}")
    logger.info(f"  Total windows (emitted): {len(combined_df)}")
    if len(combined_df) > 0:
        in_call = int((combined_df["y"] == 1).sum())
        ignored = int((combined_df["ignore"] == 1).sum())
        logger.info(f"  Class balance: {in_call/len(combined_df):.1%} in-call")
        logger.info(f"  Ignored windows: {ignored} ({ignored/len(combined_df):.1%})")

    # Anomaly summary
    rep = build_report(video_reports, cfg)
    for k, items in rep["anomalies"].items():
        if items:
            logger.warning(f"  {k}: {len(items)} items")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
