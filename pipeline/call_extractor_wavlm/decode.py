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

    starts = [int(i) for i in start_peaks.tolist()]
    ends = [int(i) for i in end_peaks.tolist()]

    # Process a combined timeline of (end, start) peaks to allow exact 0-gap boundaries.
    # Tie-break: process END before START when times are equal.
    events: list[tuple[float, int, int]] = []
    for i in starts:
        events.append((float(times_s[int(i)]), 1, int(i)))  # 1=start
    for j in ends:
        events.append((float(times_s[int(j)]), 0, int(j)))  # 0=end
    events.sort(key=lambda x: (float(x[0]), int(x[1])))

    min_dur = float(cfg.min_duration_s)
    max_dur = float(cfg.max_duration_s)
    in_call_min = float(cfg.in_call_mean_min)
    internal_thr = float(cfg.internal_peak_drop_threshold)
    join_tol = float(cfg.boundary_join_tolerance_s)

    def slice_bounds(s_t: float, e_t: float) -> tuple[int, int]:
        l = int(np.searchsorted(times_s, float(s_t), side="left"))
        r = int(np.searchsorted(times_s, float(e_t), side="right"))
        return l, r

    def slice_internal(s_t: float, e_t: float) -> tuple[int, int]:
        # Exclude a small neighborhood around boundaries to avoid dropping 0-gap segments
        # due to frame discretization (peaks can land slightly inside the interval).
        l = int(np.searchsorted(times_s, float(s_t) + join_tol, side="right"))
        r = int(np.searchsorted(times_s, float(e_t) - join_tol, side="left"))
        return l, r

    segments: list[dict] = []
    candidate_starts: list[int] = []
    min_start_t = float("-inf")
    for _t, kind, idx in events:
        if kind == 1:
            candidate_starts.append(int(idx))
            continue

        if not candidate_starts:
            continue

        e_idx = int(idx)
        e_t = float(times_s[e_idx])

        # Drop candidates that are already too old (cannot match any later end either).
        candidate_starts = [s for s in candidate_starts if (e_t - float(times_s[int(s)])) <= max_dur]
        if not candidate_starts:
            continue

        # Starts very close to this end can belong to the *next* call in a 0-gap boundary;
        # keep them for the next segment instead of treating them as ambiguous.
        eligible_starts = [s for s in candidate_starts if float(times_s[int(s)]) < (e_t - join_tol)]
        future_starts = [s for s in candidate_starts if float(times_s[int(s)]) >= (e_t - join_tol)]
        if not eligible_starts:
            candidate_starts = future_starts
            continue

        # If we have multiple *strong* starts before an end, treat as ambiguous and drop.
        strong_starts = [s for s in eligible_starts if float(start_p[int(s)]) >= internal_thr]
        if len(strong_starts) > 1:
            candidate_starts.clear()
            continue

        best: tuple[float, int, float, float] | None = None  # (score, s_idx, s_t_eff, mean_in_call)
        for s_idx in eligible_starts:
            s_idx = int(s_idx)
            s_t_raw = float(times_s[s_idx])
            s_t = max(float(s_t_raw), float(min_start_t))
            dur = float(e_t - s_t)
            if dur < min_dur or dur > max_dur:
                continue

            l, r = slice_bounds(s_t, e_t)
            if r <= l:
                continue
            mean_in_call = float(in_call_p[l:r].mean())
            if mean_in_call < in_call_min:
                continue

            li, ri = slice_internal(s_t, e_t)
            if ri > li:
                if float(start_p[li:ri].max(initial=0.0)) >= internal_thr:
                    continue
                if float(end_p[li:ri].max(initial=0.0)) >= internal_thr:
                    continue

            score = float(min(float(start_p[s_idx]), float(end_p[e_idx]), mean_in_call))
            if best is None or score > best[0]:
                best = (score, s_idx, s_t, mean_in_call)

        if best is None:
            candidate_starts = future_starts
            continue

        score, s_idx, s_t, mean_in_call = best

        segments.append(
            {
                "start_s": float(s_t),
                "end_s": float(e_t),
                "score": float(score),
                "mean_in_call": float(mean_in_call),
                "start_p": float(start_p[int(s_idx)]),
                "end_p": float(end_p[int(e_idx)]),
            }
        )

        # Reset for the next segment, but carry forward any start peaks that were
        # within the join tolerance of this end (0-gap boundaries).
        candidate_starts = [s for s in future_starts if int(s) != int(s_idx)]
        min_start_t = float(e_t)

    return segments
