from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .decode_viterbi import ViterbiConfig, on_off_to_segments, viterbi_decode_on_off


@dataclass(frozen=True)
class DecodeConfig:
    mode: str = "peaks"  # "peaks" | "viterbi"

    start_peak_threshold: float = 0.70
    end_peak_threshold: float = 0.70
    in_call_mean_min: float = 0.60

    nms_min_sep_s: float = 0.35
    min_duration_s: float = 1.0
    max_duration_s: float = 4 * 60 * 60  # 4 hours

    internal_peak_drop_threshold: float = 0.80
    boundary_join_tolerance_s: float = 1e-3  # allow next start at exactly previous end

    # Viterbi/HMM decode params (used when mode == "viterbi")
    viterbi_off_to_on_penalty: float = 6.0
    viterbi_on_to_off_penalty: float = 6.0
    viterbi_start_scale: float = 0.0
    viterbi_end_scale: float = 0.0
    viterbi_min_on_s: float = 2.0
    viterbi_min_off_s: float = 0.0
    viterbi_smooth_win_s: float = 0.0


def _local_peak_indices(probs: np.ndarray, *, threshold: float) -> np.ndarray:
    p = probs.astype(np.float32, copy=False)
    if p.size == 0:
        return np.zeros((0,), dtype=np.int64)

    # Use *strict* local maxima to avoid generating huge numbers of peaks on nearly-flat probabilities.
    # (WavLM frame heads can output low-amplitude signals early in training.)
    is_peak = np.zeros_like(p, dtype=bool)
    thr = float(threshold)
    if p.size == 1:
        is_peak[0] = p[0] >= thr
        return np.flatnonzero(is_peak).astype(np.int64)

    # Edge peaks: allow only strict dominance over the single neighbor.
    is_peak[0] = (p[0] >= thr) and (p[0] > p[1])
    is_peak[-1] = (p[-1] >= thr) and (p[-1] > p[-2])

    mid = (p[1:-1] >= thr) & (p[1:-1] > p[:-2]) & (p[1:-1] > p[2:])
    is_peak[1:-1] = mid
    return np.flatnonzero(is_peak).astype(np.int64)


def _nms_time(
    peak_idxs: np.ndarray,
    *,
    times_s: np.ndarray,
    probs: np.ndarray,
    min_sep_s: float,
) -> np.ndarray:
    if peak_idxs.size == 0:
        return peak_idxs

    # Greedy NMS by peak probability, using time-index suppression on the (sorted) frame timeline.
    # This avoids O(N^2) behavior when the start/end heads are near-flat and produce many local maxima.
    order = peak_idxs[np.argsort(probs[peak_idxs])[::-1]]
    suppressed = np.zeros((times_s.shape[0],), dtype=bool)
    kept: List[int] = []

    sep = float(min_sep_s)
    for idx in order.tolist():
        idx_i = int(idx)
        if suppressed[idx_i]:
            continue
        kept.append(idx_i)

        t0 = float(times_s[idx_i])
        left = int(np.searchsorted(times_s, t0 - sep, side="left"))
        right = int(np.searchsorted(times_s, t0 + sep, side="right"))
        suppressed[left:right] = True

    kept.sort(key=lambda i: float(times_s[i]))
    return np.asarray(kept, dtype=np.int64)


def probabilities_to_segments(
    *,
    times_s: np.ndarray,
    in_call_p: np.ndarray,
    start_p: np.ndarray,
    end_p: np.ndarray,
    cfg: DecodeConfig,
) -> list[dict]:
    """Conservative start/end pairing decoder.

    Drops ambiguous patterns rather than risking merges or over-splits.
    """
    if not (times_s.shape == in_call_p.shape == start_p.shape == end_p.shape):
        raise ValueError("All arrays must have the same shape")

    if str(cfg.mode).lower() == "viterbi":
        on = viterbi_decode_on_off(
            times_s=times_s,
            in_call_p=in_call_p,
            start_p=start_p,
            end_p=end_p,
            cfg=ViterbiConfig(
                off_to_on_penalty=float(cfg.viterbi_off_to_on_penalty),
                on_to_off_penalty=float(cfg.viterbi_on_to_off_penalty),
                start_scale=float(cfg.viterbi_start_scale),
                end_scale=float(cfg.viterbi_end_scale),
                min_on_s=float(cfg.viterbi_min_on_s),
                min_off_s=float(cfg.viterbi_min_off_s),
                smooth_win_s=float(cfg.viterbi_smooth_win_s),
            ),
        )
        seg_bounds = on_off_to_segments(times_s=times_s, on=on)
        segments: list[dict] = []
        for s_t, e_t in seg_bounds:
            if e_t - s_t < float(cfg.min_duration_s) or e_t - s_t > float(cfg.max_duration_s):
                continue
            inside = (times_s >= float(s_t)) & (times_s <= float(e_t))
            if inside.sum() == 0:
                continue
            mean_in_call = float(in_call_p[inside].mean())
            if mean_in_call < float(cfg.in_call_mean_min):
                continue
            segments.append(
                {
                    "start_s": float(s_t),
                    "end_s": float(e_t),
                    "score": float(mean_in_call),
                    "mean_in_call": float(mean_in_call),
                }
            )
        return segments

    start_peaks = _local_peak_indices(start_p, threshold=float(cfg.start_peak_threshold))
    end_peaks = _local_peak_indices(end_p, threshold=float(cfg.end_peak_threshold))

    start_peaks = _nms_time(start_peaks, times_s=times_s, probs=start_p, min_sep_s=float(cfg.nms_min_sep_s))
    end_peaks = _nms_time(end_peaks, times_s=times_s, probs=end_p, min_sep_s=float(cfg.nms_min_sep_s))

    starts = start_peaks.tolist()
    ends = end_peaks.tolist()

    segments: list[dict] = []
    i = 0
    j = 0
    while i < len(starts):
        s_idx = int(starts[i])
        s_t = float(times_s[s_idx])

        while j < len(ends) and float(times_s[int(ends[j])]) <= s_t + float(cfg.min_duration_s):
            j += 1
        if j >= len(ends):
            break

        e_idx = int(ends[j])
        e_t = float(times_s[e_idx])

        if e_t - s_t > float(cfg.max_duration_s):
            i += 1
            continue

        if i + 1 < len(starts) and float(times_s[int(starts[i + 1])]) < e_t:
            # Multiple starts before an end -> ambiguous. Drop the whole region up to this end.
            while i < len(starts) and float(times_s[int(starts[i])]) < e_t:
                i += 1
            j += 1
            continue

        inside = (times_s >= s_t) & (times_s <= e_t)
        if inside.sum() == 0:
            i += 1
            continue

        mean_in_call = float(in_call_p[inside].mean())
        if mean_in_call < float(cfg.in_call_mean_min):
            i += 1
            continue

        # If we see another strong boundary peak inside the segment, drop it.
        internal = (times_s > s_t) & (times_s < e_t)
        if internal.any():
            if float(start_p[internal].max(initial=0.0)) >= float(cfg.internal_peak_drop_threshold):
                i += 1
                continue
            if float(end_p[internal].max(initial=0.0)) >= float(cfg.internal_peak_drop_threshold):
                i += 1
                continue

        segments.append(
            {
                "start_s": float(s_t),
                "end_s": float(e_t),
                "score": float(min(float(start_p[s_idx]), float(end_p[e_idx]), mean_in_call)),
                "mean_in_call": float(mean_in_call),
                "start_p": float(start_p[s_idx]),
                "end_p": float(end_p[e_idx]),
            }
        )

        # Move to the next start at/after this end (allow exact equality for 0-gap calls).
        i += 1
        while i < len(starts) and float(times_s[int(starts[i])]) < e_t - float(cfg.boundary_join_tolerance_s):
            i += 1
        j += 1

    return segments
