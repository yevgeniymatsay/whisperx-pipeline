from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ViterbiConfig:
    off_to_on_penalty: float = 6.0
    on_to_off_penalty: float = 6.0
    start_scale: float = 0.0
    end_scale: float = 0.0

    min_on_s: float = 2.0
    min_off_s: float = 0.0
    smooth_win_s: float = 0.0


def _median_dt_s(times_s: np.ndarray) -> float:
    if times_s.size < 2:
        return 0.0
    d = np.diff(times_s.astype(np.float64, copy=False))
    d = d[d > 0]
    if d.size == 0:
        return 0.0
    return float(np.median(d))


def _smooth(p: np.ndarray, *, win_steps: int) -> np.ndarray:
    if win_steps <= 1:
        return p
    k = np.ones((int(win_steps),), dtype=np.float32) / float(win_steps)
    return np.convolve(p.astype(np.float32, copy=False), k, mode="same")


def _apply_min_run_lengths(on: np.ndarray, *, min_on_steps: int, min_off_steps: int) -> np.ndarray:
    """Flip short runs to enforce minimum ON/OFF durations.

    Note: Filling short OFF gaps can merge adjacent calls; decode sweeps must enforce merges==0.
    """
    on = on.astype(bool, copy=True)
    n = int(on.size)
    if n == 0:
        return on

    def iter_runs(vals: np.ndarray):
        i = 0
        while i < n:
            v = bool(vals[i])
            j = i + 1
            while j < n and bool(vals[j]) == v:
                j += 1
            yield v, i, j
            i = j

    # First remove short ON runs.
    if int(min_on_steps) > 1:
        for v, i, j in list(iter_runs(on)):
            if v and (j - i) < int(min_on_steps):
                on[i:j] = False

    # Then fill short OFF gaps.
    if int(min_off_steps) > 1:
        for v, i, j in list(iter_runs(on)):
            if (not v) and (j - i) < int(min_off_steps):
                on[i:j] = True

    return on


def viterbi_decode_on_off(
    *,
    times_s: np.ndarray,
    in_call_p: np.ndarray,
    start_p: np.ndarray,
    end_p: np.ndarray,
    cfg: ViterbiConfig,
) -> np.ndarray:
    """Two-state Viterbi decode (OFF/ON) with time-varying transition bonuses from start/end probabilities."""
    if not (times_s.shape == in_call_p.shape == start_p.shape == end_p.shape):
        raise ValueError("All arrays must have the same shape")
    T = int(times_s.size)
    if T == 0:
        return np.zeros((0,), dtype=bool)

    eps = 1e-6
    p = np.clip(in_call_p.astype(np.float32, copy=False), eps, 1.0 - eps)

    dt_s = _median_dt_s(times_s)
    win_steps = 0
    if float(cfg.smooth_win_s) > 0.0 and dt_s > 0.0:
        win_steps = int(max(1, round(float(cfg.smooth_win_s) / float(dt_s))))
    if win_steps > 1:
        p = _smooth(p, win_steps=win_steps)
        p = np.clip(p, eps, 1.0 - eps)

    emit_on = np.log(p)
    emit_off = np.log(1.0 - p)

    dp_on = np.empty((T,), dtype=np.float32)
    dp_off = np.empty((T,), dtype=np.float32)
    bp_on = np.empty((T,), dtype=np.int8)  # 0=from_on, 1=from_off
    bp_off = np.empty((T,), dtype=np.int8)  # 0=from_off, 1=from_on

    dp_on[0] = emit_on[0]
    dp_off[0] = emit_off[0]
    bp_on[0] = 0
    bp_off[0] = 0

    off_to_on = float(cfg.off_to_on_penalty)
    on_to_off = float(cfg.on_to_off_penalty)
    start_scale = float(cfg.start_scale)
    end_scale = float(cfg.end_scale)

    start_p = start_p.astype(np.float32, copy=False)
    end_p = end_p.astype(np.float32, copy=False)

    for t in range(1, T):
        # OFF -> ON transition bonus uses start_p at the transition time step.
        bonus_on = start_scale * float(start_p[t])
        s_on_from_on = float(dp_on[t - 1])
        s_on_from_off = float(dp_off[t - 1]) - off_to_on + bonus_on
        if s_on_from_on >= s_on_from_off:
            dp_on[t] = emit_on[t] + s_on_from_on
            bp_on[t] = 0
        else:
            dp_on[t] = emit_on[t] + s_on_from_off
            bp_on[t] = 1

        # ON -> OFF transition bonus uses end_p at the transition time step.
        bonus_off = end_scale * float(end_p[t])
        s_off_from_off = float(dp_off[t - 1])
        s_off_from_on = float(dp_on[t - 1]) - on_to_off + bonus_off
        if s_off_from_off >= s_off_from_on:
            dp_off[t] = emit_off[t] + s_off_from_off
            bp_off[t] = 0
        else:
            dp_off[t] = emit_off[t] + s_off_from_on
            bp_off[t] = 1

    on = np.empty((T,), dtype=bool)
    state_on = bool(dp_on[-1] >= dp_off[-1])
    on[-1] = state_on

    for t in range(T - 1, 0, -1):
        if state_on:
            state_on = bool(bp_on[t] == 0)
        else:
            state_on = bool(bp_off[t] == 1)
        on[t - 1] = state_on

    if dt_s > 0.0:
        min_on_steps = int(max(1, math.ceil(float(cfg.min_on_s) / float(dt_s)))) if float(cfg.min_on_s) > 0 else 1
        min_off_steps = int(max(1, math.ceil(float(cfg.min_off_s) / float(dt_s)))) if float(cfg.min_off_s) > 0 else 1
        on = _apply_min_run_lengths(on, min_on_steps=min_on_steps, min_off_steps=min_off_steps)

    return on


def on_off_to_segments(
    *,
    times_s: np.ndarray,
    on: np.ndarray,
) -> list[tuple[float, float]]:
    if times_s.size == 0:
        return []
    if times_s.shape != on.shape:
        raise ValueError("times_s and on must have the same shape")

    segs: list[tuple[float, float]] = []
    n = int(on.size)
    i = 0
    while i < n:
        if not bool(on[i]):
            i += 1
            continue
        j = i + 1
        while j < n and bool(on[j]):
            j += 1
        segs.append((float(times_s[i]), float(times_s[j - 1])))
        i = j
    return segs

