from __future__ import annotations

import numpy as np

from pipeline.call_extractor_wavlm.decode import DecodeConfig, probabilities_to_segments
from pipeline.call_extractor_wavlm.labels import TargetConfig, make_targets_for_frames
from pipeline.call_extractor_wavlm.metrics import compute_gate_metrics
from pipeline.call_extractor_wavlm.types import CallBoundary


def test_targets_allow_zero_gap_boundary_events() -> None:
    boundaries = [
        CallBoundary(start_s=10.0, end_s=20.0),
        CallBoundary(start_s=20.0, end_s=30.0),
    ]
    t = np.arange(19.7, 20.3, 0.02, dtype=np.float32)
    in_call, start, end = make_targets_for_frames(t, boundaries=boundaries, cfg=TargetConfig(0.20, 0.20))

    assert in_call.max() == 1.0
    assert start.max() == 1.0
    assert end.max() == 1.0


def test_decoder_drops_multiple_starts_before_end() -> None:
    times = np.arange(0.0, 10.0, 1.0, dtype=np.float32)
    in_call = np.ones_like(times, dtype=np.float32)
    start = np.zeros_like(times, dtype=np.float32)
    end = np.zeros_like(times, dtype=np.float32)

    start[1] = 0.95
    start[3] = 0.90
    end[8] = 0.95

    cfg = DecodeConfig(start_peak_threshold=0.8, end_peak_threshold=0.8, in_call_mean_min=0.5, min_duration_s=1.0)
    segs = probabilities_to_segments(times_s=times, in_call_p=in_call, start_p=start, end_p=end, cfg=cfg)
    assert segs == []


def test_decoder_allows_zero_gap_adjacent_calls() -> None:
    times = np.arange(0.0, 12.0, 1.0, dtype=np.float32)
    in_call = np.ones_like(times, dtype=np.float32)
    start = np.zeros_like(times, dtype=np.float32)
    end = np.zeros_like(times, dtype=np.float32)

    start[0] = 0.95
    end[5] = 0.95
    start[5] = 0.95
    end[10] = 0.95

    cfg = DecodeConfig(start_peak_threshold=0.8, end_peak_threshold=0.8, in_call_mean_min=0.5, min_duration_s=1.0)
    segs = probabilities_to_segments(times_s=times, in_call_p=in_call, start_p=start, end_p=end, cfg=cfg)
    assert len(segs) == 2
    assert segs[0]["start_s"] == 0.0
    assert segs[0]["end_s"] == 5.0
    assert segs[1]["start_s"] == 5.0
    assert segs[1]["end_s"] == 10.0


def test_decoder_keeps_single_strong_start_when_multiple_starts_exist() -> None:
    times = np.arange(0.0, 10.0, 1.0, dtype=np.float32)
    in_call = np.ones_like(times, dtype=np.float32)
    start = np.zeros_like(times, dtype=np.float32)
    end = np.zeros_like(times, dtype=np.float32)

    start[1] = 0.95
    start[3] = 0.65  # below internal_peak_drop_threshold => treated as weak
    end[8] = 0.95

    cfg = DecodeConfig(
        start_peak_threshold=0.6,
        end_peak_threshold=0.8,
        in_call_mean_min=0.5,
        min_duration_s=1.0,
        internal_peak_drop_threshold=0.9,
    )
    segs = probabilities_to_segments(times_s=times, in_call_p=in_call, start_p=start, end_p=end, cfg=cfg)
    assert len(segs) == 1
    assert segs[0]["start_s"] == 1.0
    assert segs[0]["end_s"] == 8.0


def test_transitions_decoder_ignores_internal_start_peaks_without_in_call_transition() -> None:
    # Start peaks inside an active call are ignored if in_call is already high before the peak.
    times = np.arange(0.0, 6.0, 1.0, dtype=np.float32)
    in_call = np.array([0.1, 0.9, 0.9, 0.9, 0.1, 0.1], dtype=np.float32)
    start = np.zeros_like(times, dtype=np.float32)
    end = np.zeros_like(times, dtype=np.float32)

    start[1] = 0.95  # valid boundary-like start (off->on)
    start[3] = 0.95  # internal start peak; should be ignored (on->on)
    end[4] = 0.95  # boundary-like end (on->off)

    cfg = DecodeConfig(
        mode="transitions",
        start_peak_threshold=0.8,
        end_peak_threshold=0.8,
        in_call_threshold=0.5,
        transition_win_s=1.0,
        transition_margin=0.0,
        in_call_smooth_win_s=0.0,
        in_call_mean_min=0.0,
        min_duration_s=1.0,
    )
    segs = probabilities_to_segments(times_s=times, in_call_p=in_call, start_p=start, end_p=end, cfg=cfg)
    assert len(segs) == 1
    assert segs[0]["start_s"] == 1.0
    assert segs[0]["end_s"] == 4.0


def test_transitions_decoder_restarts_on_new_start_when_end_missing() -> None:
    # If a new boundary-like start arrives before any end, restart (drop the earlier segment) to avoid merges.
    times = np.arange(0.0, 7.0, 1.0, dtype=np.float32)
    in_call = np.array([0.1, 0.9, 0.1, 0.9, 0.9, 0.9, 0.1], dtype=np.float32)
    start = np.zeros_like(times, dtype=np.float32)
    end = np.zeros_like(times, dtype=np.float32)

    start[1] = 0.95  # boundary-like start (off->on)
    start[3] = 0.95  # boundary-like start (off->on) again
    end[6] = 0.95  # only one end peak (the earlier end is missing)

    cfg = DecodeConfig(
        mode="transitions",
        start_peak_threshold=0.8,
        end_peak_threshold=0.8,
        in_call_threshold=0.5,
        transition_win_s=1.0,
        transition_margin=0.0,
        in_call_smooth_win_s=0.0,
        transition_restart_on_new_start=True,
        in_call_mean_min=0.0,
        min_duration_s=1.0,
    )
    segs = probabilities_to_segments(times_s=times, in_call_p=in_call, start_p=start, end_p=end, cfg=cfg)
    assert len(segs) == 1
    assert segs[0]["start_s"] == 3.0
    assert segs[0]["end_s"] == 6.0


def test_metrics_merge_and_oversplit_detection() -> None:
    gt = [CallBoundary(0.0, 5.0), CallBoundary(5.0, 10.0)]
    pred_merge = [CallBoundary(0.0, 10.0)]
    m1 = compute_gate_metrics(gt=gt, pred=pred_merge)
    assert m1.merges == 1
    assert m1.oversplits == 0

    pred_oversplit = [CallBoundary(0.0, 3.0), CallBoundary(3.0, 5.0)]
    m2 = compute_gate_metrics(gt=[CallBoundary(0.0, 5.0)], pred=pred_oversplit)
    assert m2.merges == 0
    assert m2.oversplits == 1


def test_metrics_counts_false_positive_segments_on_call_videos() -> None:
    # GT has a single call, but pred includes an extra non-overlapping segment.
    gt = [CallBoundary(10.0, 20.0)]
    pred = [CallBoundary(10.0, 20.0), CallBoundary(30.0, 40.0)]
    m = compute_gate_metrics(gt=gt, pred=pred)
    assert m.merges == 0
    assert m.oversplits == 0
    assert m.false_positive_segments == 1


def test_metrics_counts_false_positive_segments_on_no_call_videos() -> None:
    gt: list[CallBoundary] = []
    pred = [CallBoundary(0.0, 5.0), CallBoundary(10.0, 12.0)]
    m = compute_gate_metrics(gt=gt, pred=pred)
    assert m.gt_calls == 0
    assert m.pred_calls == 2
    assert m.false_positive_segments == 2
