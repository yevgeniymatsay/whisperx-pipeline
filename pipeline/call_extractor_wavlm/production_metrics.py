from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def _intersection_exact_s(a0: float, a1: float, b0: float, b1: float) -> float:
    s = max(float(a0), float(b0))
    e = min(float(a1), float(b1))
    return max(0.0, e - s)


def merge_intervals(
    intervals: Sequence[tuple[float, float]],
    *,
    join_tolerance_s: float = 0.0,
) -> list[tuple[float, float]]:
    """Return a sorted, disjoint union of intervals.

    join_tolerance_s merges intervals that are within tolerance (for numeric jitter).
    """
    tol = float(join_tolerance_s)
    if tol < 0.0:
        raise ValueError(f"join_tolerance_s must be >= 0 (got {join_tolerance_s})")

    cleaned: list[tuple[float, float]] = []
    for s, e in intervals:
        s_f = float(s)
        e_f = float(e)
        if e_f <= s_f:
            continue
        cleaned.append((s_f, e_f))

    if not cleaned:
        return []

    cleaned.sort(key=lambda x: (float(x[0]), float(x[1])))

    out: list[tuple[float, float]] = []
    cur_s, cur_e = cleaned[0]
    for s, e in cleaned[1:]:
        if float(s) <= float(cur_e) + tol:
            cur_e = max(float(cur_e), float(e))
        else:
            out.append((float(cur_s), float(cur_e)))
            cur_s, cur_e = float(s), float(e)
    out.append((float(cur_s), float(cur_e)))
    return out


def overlap_seconds_with_union(
    interval: tuple[float, float],
    union_intervals: Sequence[tuple[float, float]],
) -> float:
    """Compute exact overlap seconds between an interval and a disjoint union list."""
    a0, a1 = float(interval[0]), float(interval[1])
    if a1 <= a0 or not union_intervals:
        return 0.0

    total = 0.0
    # Two-pointer sweep. union_intervals must be sorted disjoint (merge_intervals output).
    for b0, b1 in union_intervals:
        if float(b1) <= a0:
            continue
        if float(b0) >= a1:
            break
        total += _intersection_exact_s(a0, a1, float(b0), float(b1))
    return float(total)


def total_seconds(intervals: Sequence[tuple[float, float]]) -> float:
    """Sum of interval durations (expects non-overlapping intervals if used as a 'union' length)."""
    return float(sum(max(0.0, float(e) - float(s)) for s, e in intervals))


def overlap_seconds_between_unions(
    union_a: Sequence[tuple[float, float]],
    union_b: Sequence[tuple[float, float]],
) -> float:
    """Exact overlap seconds between two disjoint union lists."""
    total = 0.0
    for s, e in union_a:
        total += overlap_seconds_with_union((float(s), float(e)), union_b)
    return float(total)


def per_call_coverages(
    *,
    gt_calls: Sequence[tuple[float, float]],
    pred_intervals: Sequence[tuple[float, float]],
) -> list[float]:
    """Per-call coverage fraction: overlap(pred_union, gt_call) / duration(gt_call)."""
    pred_union = merge_intervals(pred_intervals, join_tolerance_s=0.0)
    coverages: list[float] = []
    for gs, ge in gt_calls:
        gs_f = float(gs)
        ge_f = float(ge)
        dur = max(0.0, ge_f - gs_f)
        if dur <= 0.0:
            continue
        ov = overlap_seconds_with_union((gs_f, ge_f), pred_union)
        ov = min(float(ov), float(dur))
        coverages.append(float(ov) / float(dur))
    return coverages


@dataclass(frozen=True)
class PurityAgg:
    pred_seconds: float
    overlap_seconds: float
    non_call_seconds: float
    segment_purities: list[float]


def purity_vs_union(
    *,
    pred_intervals: Sequence[tuple[float, float]],
    gt_union_intervals: Sequence[tuple[float, float]],
) -> PurityAgg:
    pred_seconds = 0.0
    overlap_seconds = 0.0
    non_call_seconds = 0.0
    purities: list[float] = []

    for s, e in pred_intervals:
        s_f = float(s)
        e_f = float(e)
        dur = max(0.0, e_f - s_f)
        if dur <= 0.0:
            continue
        ov = overlap_seconds_with_union((s_f, e_f), gt_union_intervals)
        ov = min(float(ov), float(dur))
        pred_seconds += float(dur)
        overlap_seconds += float(ov)
        nc = float(dur) - float(ov)
        non_call_seconds += float(nc)
        purities.append(float(ov) / float(dur))

    return PurityAgg(
        pred_seconds=float(pred_seconds),
        overlap_seconds=float(overlap_seconds),
        non_call_seconds=float(non_call_seconds),
        segment_purities=purities,
    )


def quantiles(values: Iterable[float], qs: Sequence[float]) -> list[float | None]:
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    if arr.size == 0:
        return [None for _ in qs]
    return [float(np.quantile(arr, float(q))) for q in qs]
