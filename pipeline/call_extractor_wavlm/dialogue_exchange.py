from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Turn:
    start_s: float
    end_s: float
    speaker: str

    @property
    def dur_s(self) -> float:
        return float(max(0.0, self.end_s - self.start_s))


def merge_consecutive_turns(turns: Sequence[Turn], *, eps_s: float = 1e-6) -> list[Turn]:
    """Merge overlapping/touching turns for the same speaker.

    This expects turns are roughly time-ordered; it sorts defensively.
    """
    if not turns:
        return []
    items = sorted(list(turns), key=lambda t: (t.start_s, t.end_s, t.speaker))
    out: list[Turn] = []
    for t in items:
        if t.end_s <= t.start_s:
            continue
        if not out:
            out.append(t)
            continue
        prev = out[-1]
        if t.speaker == prev.speaker and t.start_s <= prev.end_s + float(eps_s):
            out[-1] = Turn(start_s=prev.start_s, end_s=max(prev.end_s, t.end_s), speaker=prev.speaker)
        else:
            out.append(t)
    return out


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, float(q)))


def compute_t_bridge_s(
    turns: Sequence[Turn],
    *,
    gap_pctl: float = 90.0,
    min_s: float = 2.0,
    max_s: float = 15.0,
) -> float:
    """Compute a per-clip bridge time from alternating-speaker gaps (pctl-clipped)."""
    merged = merge_consecutive_turns(turns)
    gaps: list[float] = []
    for a, b in zip(merged, merged[1:]):
        if a.speaker == b.speaker:
            continue
        gap = float(max(0.0, b.start_s - a.end_s))
        gaps.append(gap)
    raw = _percentile(gaps, float(gap_pctl))
    if not math.isfinite(raw):
        raw = float(min_s)
    return float(min(max(raw, float(min_s)), float(max_s)))


def dialogue_exchange_mask(
    turns: Sequence[Turn],
    *,
    clip_duration_s: float,
    dt_s: float,
    t_bridge_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (times_s, mask) for DIALOGUE_EXCHANGE definition.

    mask[t] is True if:
    - both speakers have spoken within last t_bridge_s, and
    - at least one alternation occurred within last t_bridge_s.
    """
    duration = float(clip_duration_s)
    if duration <= 0.0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=bool)
    dt = float(dt_s)
    if dt <= 0.0:
        raise ValueError(f"dt_s must be > 0 (got {dt_s})")
    tb = float(t_bridge_s)
    if tb <= 0.0:
        raise ValueError(f"t_bridge_s must be > 0 (got {t_bridge_s})")

    merged = merge_consecutive_turns(turns)
    if not merged:
        times = np.arange(0.0, duration, dt, dtype=np.float32)
        return times, np.zeros((times.shape[0],), dtype=bool)

    times = np.arange(0.0, duration, dt, dtype=np.float32)
    active = np.full((times.shape[0],), -1, dtype=np.int8)  # -1 silence, 0 S0, 1 S1

    # Fill active speaker per time step.
    for t in merged:
        spk = str(t.speaker)
        if spk not in ("S0", "S1"):
            continue
        s = float(max(0.0, min(duration, t.start_s)))
        e = float(max(0.0, min(duration, t.end_s)))
        if e <= s:
            continue
        i0 = int(np.searchsorted(times, s, side="left"))
        i1 = int(np.searchsorted(times, e, side="left"))
        active[i0:i1] = 0 if spk == "S0" else 1

    # Alternation events at turn starts where speaker differs from previous.
    alt_event = np.zeros((times.shape[0],), dtype=bool)
    for prev, cur in zip(merged, merged[1:]):
        if prev.speaker == cur.speaker:
            continue
        t_alt = float(cur.start_s)
        idx = int(np.searchsorted(times, t_alt, side="left"))
        if 0 <= idx < alt_event.shape[0]:
            alt_event[idx] = True

    last_s0 = -1e9
    last_s1 = -1e9
    last_alt = -1e9
    out = np.zeros((times.shape[0],), dtype=bool)
    for i, t in enumerate(times.tolist()):
        if alt_event[i]:
            last_alt = float(t)
        a = int(active[i])
        if a == 0:
            last_s0 = float(t)
        elif a == 1:
            last_s1 = float(t)
        ok = (float(t) - last_s0 <= tb) and (float(t) - last_s1 <= tb) and (float(t) - last_alt <= tb)
        out[i] = bool(ok)

    return times, out


def mask_to_intervals(
    times_s: np.ndarray,
    mask: np.ndarray,
    *,
    dt_s: float,
) -> list[tuple[float, float]]:
    if times_s.size == 0 or mask.size == 0:
        return []
    if times_s.shape[0] != mask.shape[0]:
        raise ValueError("times_s and mask must have the same length")
    dt = float(dt_s)
    if dt <= 0.0:
        raise ValueError(f"dt_s must be > 0 (got {dt_s})")

    intervals: list[tuple[float, float]] = []
    in_run = False
    start_idx = 0
    for i, v in enumerate(mask.tolist()):
        if bool(v) and not in_run:
            in_run = True
            start_idx = i
        elif (not bool(v)) and in_run:
            in_run = False
            start = float(times_s[start_idx])
            end = float(times_s[i - 1]) + dt
            if end > start:
                intervals.append((start, end))
    if in_run:
        start = float(times_s[start_idx])
        end = float(times_s[-1]) + dt
        if end > start:
            intervals.append((start, end))
    return intervals


def dialogue_exchange_intervals(
    turns: Sequence[Turn],
    *,
    clip_duration_s: float,
    dt_s: float = 0.1,
    gap_pctl: float = 90.0,
    t_bridge_min_s: float = 2.0,
    t_bridge_max_s: float = 15.0,
) -> tuple[float, list[tuple[float, float]]]:
    """Return (t_bridge_s, relative intervals) for DIALOGUE_EXCHANGE."""
    tb = compute_t_bridge_s(turns, gap_pctl=float(gap_pctl), min_s=float(t_bridge_min_s), max_s=float(t_bridge_max_s))
    times, mask = dialogue_exchange_mask(turns, clip_duration_s=float(clip_duration_s), dt_s=float(dt_s), t_bridge_s=float(tb))
    intervals = mask_to_intervals(times, mask, dt_s=float(dt_s))
    return float(tb), intervals

