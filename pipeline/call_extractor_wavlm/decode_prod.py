from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .decode_viterbi import ViterbiConfig, on_off_to_segments, viterbi_decode_on_off


@dataclass(frozen=True)
class ProdDecodeConfig:
    """Production-oriented decoder config (no merges, low no-call FP).

    Notes:
    - Base segmentation uses two-state Viterbi on `in_call_p`.
    - start/end heads are optional; default to disabled (start_scale=end_scale=0).
    - Post-processing prefers over-splitting over merging.
    """

    # Viterbi base
    viterbi_off_to_on_penalty: float = 9.4596
    viterbi_on_to_off_penalty: float = 9.4478
    viterbi_cost_mult: float = 1.0
    viterbi_start_scale: float = 0.0
    viterbi_end_scale: float = 0.0
    viterbi_min_on_s: float = 1.0
    viterbi_min_off_s: float = 0.0
    viterbi_smooth_win_s: float = 0.10

    # Anti-merge splitter (scan for sustained low `in_call` valleys inside predicted segments)
    split_lo: float = 0.35
    split_min_off_s: float = 0.50

    # Boundary-cue splitter (uses start/end heads to split even when `in_call` never dips)
    use_boundary_cues: bool = False
    boundary_mode: str = "pair_end_start"  # "pair_end_start" or "max_score"
    boundary_start_thr: float = 0.70
    boundary_end_thr: float = 0.70
    boundary_score_thr: float = 0.70  # used only for max_score mode
    boundary_smooth_win_s: float = 0.10
    boundary_nms_sep_s: float = 0.50
    boundary_pair_max_gap_s: float = 5.00
    boundary_split_margin_s: float = 2.00
    boundary_split_gap_s: float = 0.20  # avoid tol-overlap merges on tiny/zero-gap GT

    # Segment constraints
    min_duration_s: float = 2.0
    max_duration_s: float = 4 * 60 * 60  # 4 hours
    max_segment_s: float = 900.0  # failsafe: force split long segments (merge-risk guard)

    # Cheap confidence filter before judge LLM
    mean_in_call_min: float = 0.45
    max_in_call_min: float = 0.70

    # Optional contamination trim
    trim_s: float = 0.0


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
        return p.astype(np.float32, copy=False)
    k = np.ones((int(win_steps),), dtype=np.float32) / float(win_steps)
    return np.convolve(p.astype(np.float32, copy=False), k, mode="same").astype(np.float32, copy=False)


def _smooth_in_call(
    *,
    times_s: np.ndarray,
    in_call_p: np.ndarray,
    smooth_win_s: float,
) -> np.ndarray:
    eps = 1e-6
    p = np.clip(in_call_p.astype(np.float32, copy=False), eps, 1.0 - eps)
    dt_s = _median_dt_s(times_s)
    if float(smooth_win_s) <= 0.0 or dt_s <= 0.0:
        return p
    win_steps = int(max(1, round(float(smooth_win_s) / float(dt_s))))
    if win_steps <= 1:
        return p
    p = _smooth(p, win_steps=win_steps)
    return np.clip(p, eps, 1.0 - eps)


def _smooth_generic(
    *,
    times_s: np.ndarray,
    p: np.ndarray,
    smooth_win_s: float,
) -> np.ndarray:
    p = p.astype(np.float32, copy=False)
    dt_s = _median_dt_s(times_s)
    if float(smooth_win_s) <= 0.0 or dt_s <= 0.0:
        return p
    win_steps = int(max(1, round(float(smooth_win_s) / float(dt_s))))
    if win_steps <= 1:
        return p
    return _smooth(p, win_steps=win_steps)


def _run_duration_s(*, times_s: np.ndarray, i: int, j: int, dt_s: float) -> float:
    if j <= i:
        return 0.0
    # times_s are frame centers; add one frame to approximate span.
    base = float(times_s[int(j - 1)]) - float(times_s[int(i)])
    if dt_s > 0.0:
        base += float(dt_s)
    return max(0.0, float(base))


def _split_one_segment_on_valley(
    *,
    times_s: np.ndarray,
    p_sm: np.ndarray,
    seg: tuple[float, float],
    split_lo: float,
    split_min_off_s: float,
) -> tuple[list[tuple[float, float]], bool]:
    """Split a segment once if it contains a sustained low-probability valley.

    Returns (segments, did_split).
    """
    s_t, e_t = float(seg[0]), float(seg[1])
    if e_t <= s_t:
        return [], False

    inside = (times_s >= s_t) & (times_s <= e_t)
    idxs = np.flatnonzero(inside).astype(np.int64)
    if idxs.size < 3:
        return [seg], False

    dt_s = _median_dt_s(times_s[idxs])
    low = p_sm[idxs] < float(split_lo)

    best_split_t: float | None = None
    i = 0
    n = int(idxs.size)
    while i < n:
        if not bool(low[i]):
            i += 1
            continue
        j = i + 1
        while j < n and bool(low[j]):
            j += 1
        dur = _run_duration_s(times_s=times_s[idxs], i=int(i), j=int(j), dt_s=float(dt_s))
        if dur >= float(split_min_off_s):
            # Split at the deepest point of the valley.
            local = p_sm[idxs[int(i) : int(j)]]
            k = int(i) + int(np.argmin(local))
            best_split_t = float(times_s[int(idxs[int(k)])])
            break
        i = j

    if best_split_t is None:
        return [seg], False

    # Avoid degenerate splits.
    join_tol = 1e-3
    if best_split_t <= (s_t + join_tol) or best_split_t >= (e_t - join_tol):
        return [seg], False

    return [(s_t, best_split_t), (best_split_t, e_t)], True


def _split_segments_on_valleys(
    *,
    times_s: np.ndarray,
    p_sm: np.ndarray,
    segments: list[tuple[float, float]],
    split_lo: float,
    split_min_off_s: float,
) -> tuple[list[tuple[float, float]], bool]:
    out: list[tuple[float, float]] = []
    did_any = False
    for seg in segments:
        pending = [seg]
        while pending:
            cur = pending.pop()
            parts, did = _split_one_segment_on_valley(
                times_s=times_s,
                p_sm=p_sm,
                seg=cur,
                split_lo=float(split_lo),
                split_min_off_s=float(split_min_off_s),
            )
            if did:
                did_any = True
                pending.extend(parts)
            else:
                out.extend(parts)
    out.sort(key=lambda x: (float(x[0]), float(x[1])))
    return out, did_any


def _find_local_peaks(
    *,
    times_s: np.ndarray,
    values: np.ndarray,
    thr: float,
    start_s: float,
    end_s: float,
    margin_s: float,
) -> list[tuple[float, float]]:
    """Return (t, value) peaks above threshold within [start+margin, end-margin]."""
    t0 = float(start_s) + float(margin_s)
    t1 = float(end_s) - float(margin_s)
    if t1 <= t0:
        return []

    inside = (times_s >= t0) & (times_s <= t1)
    idxs = np.flatnonzero(inside).astype(np.int64)
    if idxs.size < 3:
        return []

    v = values[idxs].astype(np.float32, copy=False)
    peaks: list[tuple[float, float]] = []
    thr_f = float(thr)
    for i in range(1, int(v.size) - 1):
        cur = float(v[int(i)])
        if cur < thr_f:
            continue
        prev = float(v[int(i - 1)])
        nxt = float(v[int(i + 1)])
        if cur >= prev and cur >= nxt and (cur > prev or cur > nxt):
            t = float(times_s[int(idxs[int(i)])])
            peaks.append((t, cur))
    return peaks


def _nms_time(
    peaks: list[tuple[float, float]],
    *,
    min_sep_s: float,
) -> list[tuple[float, float]]:
    if not peaks:
        return []
    sep = float(min_sep_s)
    if sep <= 0.0:
        return sorted(peaks, key=lambda x: float(x[0]))

    order = sorted(peaks, key=lambda x: float(x[1]), reverse=True)
    kept: list[tuple[float, float]] = []
    for t, v in order:
        if all(abs(float(t) - float(kt)) >= sep for kt, _ in kept):
            kept.append((float(t), float(v)))
    kept.sort(key=lambda x: float(x[0]))
    return kept


def _split_one_segment_on_boundary_cues(
    *,
    times_s: np.ndarray,
    p_sm: np.ndarray,
    start_sm: np.ndarray,
    end_sm: np.ndarray,
    seg: tuple[float, float],
    cfg: ProdDecodeConfig,
) -> tuple[list[tuple[float, float]], bool]:
    """Split a segment once using boundary cues (start/end peaks)."""
    s_t, e_t = float(seg[0]), float(seg[1])
    if e_t <= s_t:
        return [], False

    mode = str(cfg.boundary_mode)
    margin_s = float(cfg.boundary_split_margin_s)
    gap_s = float(cfg.boundary_split_gap_s)

    split_times: list[float] = []

    if mode == "pair_end_start":
        end_peaks = _nms_time(
            _find_local_peaks(
                times_s=times_s,
                values=end_sm,
                thr=float(cfg.boundary_end_thr),
                start_s=s_t,
                end_s=e_t,
                margin_s=margin_s,
            ),
            min_sep_s=float(cfg.boundary_nms_sep_s),
        )
        start_peaks = _nms_time(
            _find_local_peaks(
                times_s=times_s,
                values=start_sm,
                thr=float(cfg.boundary_start_thr),
                start_s=s_t,
                end_s=e_t,
                margin_s=margin_s,
            ),
            min_sep_s=float(cfg.boundary_nms_sep_s),
        )
        if not end_peaks or not start_peaks:
            return [seg], False

        starts_only = [float(t) for t, _ in start_peaks]
        max_gap = float(cfg.boundary_pair_max_gap_s)

        for t_end, _ in end_peaks:
            t_start = None
            for ts in starts_only:
                if float(ts) <= float(t_end):
                    continue
                if float(ts) - float(t_end) <= max_gap:
                    t_start = float(ts)
                    break
            if t_start is None:
                continue

            mid_mask = (times_s >= float(t_end)) & (times_s <= float(t_start))
            idxs = np.flatnonzero(mid_mask).astype(np.int64)
            if idxs.size > 0:
                k = int(idxs[int(np.argmin(p_sm[idxs]))])
                split_times.append(float(times_s[int(k)]))
            else:
                split_times.append(float(t_end + (t_start - t_end) / 2.0))

    elif mode == "max_score":
        score = np.maximum(start_sm, end_sm)
        peaks = _nms_time(
            _find_local_peaks(
                times_s=times_s,
                values=score,
                thr=float(cfg.boundary_score_thr),
                start_s=s_t,
                end_s=e_t,
                margin_s=margin_s,
            ),
            min_sep_s=float(cfg.boundary_nms_sep_s),
        )
        split_times = [float(t) for t, _ in peaks]
    else:
        raise ValueError(f"Unsupported boundary_mode: {mode}")

    if not split_times:
        return [seg], False

    split_times = sorted(set(float(t) for t in split_times))
    out: list[tuple[float, float]] = []
    cur_s = float(s_t)
    join_tol = 1e-3
    for t in split_times:
        left_e = float(t) - gap_s
        right_s = float(t) + gap_s
        if left_e <= cur_s + join_tol:
            cur_s = max(cur_s, right_s)
            continue
        out.append((float(cur_s), float(left_e)))
        cur_s = float(right_s)
    if float(e_t) > cur_s + join_tol:
        out.append((float(cur_s), float(e_t)))

    if len(out) <= 1:
        return [seg], False
    return out, True


def _split_segments_on_boundary_cues(
    *,
    times_s: np.ndarray,
    p_sm: np.ndarray,
    start_sm: np.ndarray,
    end_sm: np.ndarray,
    segments: list[tuple[float, float]],
    cfg: ProdDecodeConfig,
) -> tuple[list[tuple[float, float]], bool]:
    out: list[tuple[float, float]] = []
    did_any = False
    for seg in segments:
        pending = [seg]
        while pending:
            cur = pending.pop()
            parts, did = _split_one_segment_on_boundary_cues(
                times_s=times_s,
                p_sm=p_sm,
                start_sm=start_sm,
                end_sm=end_sm,
                seg=cur,
                cfg=cfg,
            )
            if did:
                did_any = True
                pending.extend(parts)
            else:
                out.extend(parts)
    out.sort(key=lambda x: (float(x[0]), float(x[1])))
    return out, did_any


def _split_long_segments(
    *,
    segments: list[tuple[float, float]],
    max_segment_s: float,
) -> tuple[list[tuple[float, float]], bool]:
    max_s = float(max_segment_s)
    if max_s <= 0.0:
        return segments, False
    out: list[tuple[float, float]] = []
    did = False
    for s_t, e_t in segments:
        s_t = float(s_t)
        e_t = float(e_t)
        if e_t <= s_t:
            continue
        while (e_t - s_t) > max_s:
            did = True
            cut = s_t + max_s
            out.append((float(s_t), float(cut)))
            s_t = float(cut)
        out.append((float(s_t), float(e_t)))
    out.sort(key=lambda x: (float(x[0]), float(x[1])))
    return out, did


def decode_production_segments(
    *,
    times_s: np.ndarray,
    in_call_p: np.ndarray,
    start_p: np.ndarray | None = None,
    end_p: np.ndarray | None = None,
    cfg: ProdDecodeConfig,
) -> list[dict]:
    """Decode segments suitable for production (judge LLM downstream).

    Guarantees:
    - Never relies on start/end peaks being present (start/end default to disabled).
    - Prefers splitting/dropping over merge-risk output via valley splitting + max_segment_s.
    """
    if times_s.shape != in_call_p.shape:
        raise ValueError("times_s and in_call_p must have same shape")
    if times_s.size == 0:
        return []

    # Optional arrays for viterbi transition bonuses (disabled by default).
    if start_p is None:
        start_p = np.zeros_like(in_call_p, dtype=np.float32)
    if end_p is None:
        end_p = np.zeros_like(in_call_p, dtype=np.float32)
    if not (start_p.shape == end_p.shape == times_s.shape):
        raise ValueError("start_p/end_p must match times_s shape when provided")

    # Smooth in_call once; reuse for splitter + filters (Viterbi has its own internal smoothing too).
    p_sm = _smooth_in_call(times_s=times_s, in_call_p=in_call_p, smooth_win_s=float(cfg.viterbi_smooth_win_s))

    vcfg = ViterbiConfig(
        off_to_on_penalty=float(cfg.viterbi_off_to_on_penalty) * float(cfg.viterbi_cost_mult),
        on_to_off_penalty=float(cfg.viterbi_on_to_off_penalty) * float(cfg.viterbi_cost_mult),
        start_scale=float(cfg.viterbi_start_scale),
        end_scale=float(cfg.viterbi_end_scale),
        min_on_s=float(cfg.viterbi_min_on_s),
        min_off_s=float(cfg.viterbi_min_off_s),
        smooth_win_s=float(cfg.viterbi_smooth_win_s),
    )
    on = viterbi_decode_on_off(times_s=times_s, in_call_p=in_call_p, start_p=start_p, end_p=end_p, cfg=vcfg)
    base = on_off_to_segments(times_s=times_s, on=on)

    # Viterbi returns end as the last ON frame; extend by dt/2 would be nicer but keep consistent with existing decoder.
    segments = [(float(s), float(e)) for s, e in base if float(e) > float(s)]
    if not segments:
        return []

    segments, did_max_split = _split_long_segments(segments=segments, max_segment_s=float(cfg.max_segment_s))

    did_boundary_split = False
    if bool(cfg.use_boundary_cues):
        start_sm = _smooth_generic(
            times_s=times_s,
            p=start_p,
            smooth_win_s=float(cfg.boundary_smooth_win_s),
        )
        end_sm = _smooth_generic(
            times_s=times_s,
            p=end_p,
            smooth_win_s=float(cfg.boundary_smooth_win_s),
        )
        segments, did_boundary_split = _split_segments_on_boundary_cues(
            times_s=times_s,
            p_sm=p_sm,
            start_sm=start_sm,
            end_sm=end_sm,
            segments=segments,
            cfg=cfg,
        )

    segments, did_valley_split = _split_segments_on_valleys(
        times_s=times_s,
        p_sm=p_sm,
        segments=segments,
        split_lo=float(cfg.split_lo),
        split_min_off_s=float(cfg.split_min_off_s),
    )

    reasons: list[str] = []
    if did_max_split:
        reasons.append("max_segment")
    if did_boundary_split:
        reasons.append("boundary")
    if did_valley_split:
        reasons.append("valley")
    split_reason = "+".join(reasons) if reasons else None

    min_dur = float(cfg.min_duration_s)
    max_dur = float(cfg.max_duration_s)
    trim = float(cfg.trim_s)

    out: list[dict] = []
    for s_t, e_t in segments:
        s_t = float(s_t)
        e_t = float(e_t)

        if trim > 0.0:
            s_t = float(s_t + trim)
            e_t = float(e_t - trim)
        if e_t <= s_t:
            continue

        dur = float(e_t - s_t)
        if dur < min_dur or dur > max_dur:
            continue

        inside = (times_s >= s_t) & (times_s <= e_t)
        if int(inside.sum()) <= 0:
            continue
        mean_in_call = float(p_sm[inside].mean())
        max_in_call = float(p_sm[inside].max(initial=0.0))

        if mean_in_call < float(cfg.mean_in_call_min) or max_in_call < float(cfg.max_in_call_min):
            continue

        out.append(
            {
                "start_s": float(s_t),
                "end_s": float(e_t),
                "duration_s": float(dur),
                "score": float(mean_in_call),
                "mean_in_call": float(mean_in_call),
                "max_in_call": float(max_in_call),
                "was_split": bool(split_reason is not None),
                "split_reason": split_reason,
                "trim_s": float(trim),
            }
        )

    out.sort(key=lambda s: (float(s["start_s"]), float(s["end_s"])))
    return out
