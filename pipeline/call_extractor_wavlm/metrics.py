from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .types import CallBoundary

METRICS_VERSION = "gate_metrics_v1"
GATE_POLICY_VERSION = "strict_gates_v1_fp_total"


def _intersection_exact_s(a0: float, a1: float, b0: float, b1: float) -> float:
    s = max(float(a0), float(b0))
    e = min(float(a1), float(b1))
    return max(0.0, e - s)


def _intersection_tol_s(a0: float, a1: float, b0: float, b1: float, tol_s: float) -> float:
    tol = float(tol_s)
    s = max(float(a0), float(b0) - tol)
    e = min(float(a1), float(b1) + tol)
    return max(0.0, e - s)


@dataclass(frozen=True)
class GateMetrics:
    merges: int
    oversplits: int
    gt_calls: int
    pred_calls: int
    matched_calls: int
    keep_rate: float

    false_positive_segments: int

    kept_calls_coverage: int
    keep_rate_coverage: float
    mean_coverage_exact: float | None

    mean_start_abs_err_s: float | None
    mean_end_abs_err_s: float | None

    mean_iou: float | None
    kept_calls_iou_0_5: int
    keep_rate_iou_0_5: float
    kept_calls_iou_0_8: int
    keep_rate_iou_0_8: float


def compute_gate_metrics(
    *,
    gt: Sequence[CallBoundary],
    pred: Sequence[CallBoundary],
    match_tol_s: float = 0.0,
    overlap_eps_s: float = 0.10,
    min_coverage: float = 0.0,
) -> GateMetrics:
    gt_n = len(gt)
    pred_n = len(pred)

    tol_s = float(match_tol_s)
    if tol_s < 0.0:
        raise ValueError(f"match_tol_s must be >= 0 (got {match_tol_s})")

    eps_s = float(overlap_eps_s)
    if eps_s < 0.0:
        raise ValueError(f"overlap_eps_s must be >= 0 (got {overlap_eps_s})")

    cov_thr = float(min_coverage)
    if not (0.0 <= cov_thr <= 1.0):
        raise ValueError(f"min_coverage must be in [0, 1] (got {min_coverage})")

    pred_to_gt: list[list[int]] = [[] for _ in range(pred_n)]
    gt_to_pred: list[list[int]] = [[] for _ in range(gt_n)]

    for pi, p in enumerate(pred):
        for gi, g in enumerate(gt):
            if _intersection_tol_s(p.start_s, p.end_s, g.start_s, g.end_s, tol_s) >= eps_s:
                pred_to_gt[pi].append(gi)
                gt_to_pred[gi].append(pi)

    merges = sum(1 for lst in pred_to_gt if len(lst) > 1)
    oversplits = sum(1 for lst in gt_to_pred if len(lst) > 1)

    matched: list[tuple[int, int]] = []
    for gi, preds_for_gt in enumerate(gt_to_pred):
        if len(preds_for_gt) != 1:
            continue
        pi = preds_for_gt[0]
        if len(pred_to_gt[pi]) == 1 and pred_to_gt[pi][0] == gi:
            matched.append((gi, pi))

    start_errs: list[float] = []
    end_errs: list[float] = []
    coverages_exact: list[float] = []
    for gi, pi in matched:
        p = pred[int(pi)]
        g = gt[int(gi)]
        start_errs.append(abs(float(p.start_s) - float(g.start_s)))
        end_errs.append(abs(float(p.end_s) - float(g.end_s)))

        ov = _intersection_exact_s(p.start_s, p.end_s, g.start_s, g.end_s)
        gt_dur = max(0.0, float(g.end_s) - float(g.start_s))
        cov = (float(ov) / float(gt_dur)) if gt_dur > 0.0 else 0.0
        coverages_exact.append(float(cov))

    mean_start = float(np.mean(start_errs)) if start_errs else None
    mean_end = float(np.mean(end_errs)) if end_errs else None

    ious: list[float] = []
    for gi, pi in matched:
        g = gt[int(gi)]
        p = pred[int(pi)]
        ov = _intersection_exact_s(p.start_s, p.end_s, g.start_s, g.end_s)
        gt_dur = max(0.0, float(g.end_s) - float(g.start_s))
        pred_dur = max(0.0, float(p.end_s) - float(p.start_s))
        union = gt_dur + pred_dur - float(ov)
        if union <= 0.0:
            ious.append(0.0)
        else:
            ious.append(float(ov) / float(union))
    mean_iou = float(np.mean(ious)) if ious else None
    mean_cov_exact = float(np.mean(coverages_exact)) if coverages_exact else None

    kept_cov = int(sum(1 for c in coverages_exact if float(c) >= cov_thr))
    kept_iou_0_5 = int(sum(1 for iou, cov in zip(ious, coverages_exact) if float(cov) >= cov_thr and float(iou) >= 0.5))
    kept_iou_0_8 = int(sum(1 for iou, cov in zip(ious, coverages_exact) if float(cov) >= cov_thr and float(iou) >= 0.8))

    keep_rate = (len(matched) / gt_n) if gt_n > 0 else 0.0
    keep_rate_cov = (kept_cov / gt_n) if gt_n > 0 else 0.0
    keep_rate_iou_0_5 = (kept_iou_0_5 / gt_n) if gt_n > 0 else 0.0
    keep_rate_iou_0_8 = (kept_iou_0_8 / gt_n) if gt_n > 0 else 0.0

    # A "false positive" is any predicted segment that overlaps no ground-truth call.
    # This must be counted even on videos that contain some calls, otherwise strict gating can
    # accidentally allow extra non-call segments.
    fp_segments = sum(1 for lst in pred_to_gt if len(lst) == 0)

    return GateMetrics(
        merges=int(merges),
        oversplits=int(oversplits),
        gt_calls=int(gt_n),
        pred_calls=int(pred_n),
        matched_calls=int(len(matched)),
        keep_rate=float(keep_rate),
        false_positive_segments=int(fp_segments),
        kept_calls_coverage=int(kept_cov),
        keep_rate_coverage=float(keep_rate_cov),
        mean_coverage_exact=mean_cov_exact,
        mean_start_abs_err_s=mean_start,
        mean_end_abs_err_s=mean_end,
        mean_iou=mean_iou,
        kept_calls_iou_0_5=int(kept_iou_0_5),
        keep_rate_iou_0_5=float(keep_rate_iou_0_5),
        kept_calls_iou_0_8=int(kept_iou_0_8),
        keep_rate_iou_0_8=float(keep_rate_iou_0_8),
    )


def boundaries_from_json_segments(segments: Sequence[dict]) -> list[CallBoundary]:
    out: list[CallBoundary] = []
    for s in segments:
        out.append(CallBoundary(start_s=float(s["start_s"]), end_s=float(s["end_s"])))
    out.sort(key=lambda b: (b.start_s, b.end_s))
    return out
