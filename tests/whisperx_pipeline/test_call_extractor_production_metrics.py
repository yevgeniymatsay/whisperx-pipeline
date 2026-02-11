from __future__ import annotations

from pipeline.call_extractor_wavlm.production_metrics import merge_intervals, overlap_seconds_with_union, purity_vs_union, quantiles


def test_merge_intervals_merges_overlaps_and_touches() -> None:
    intervals = [(0.0, 1.0), (0.5, 2.0), (2.0, 3.0), (10.0, 11.0)]
    out = merge_intervals(intervals, join_tolerance_s=0.0)
    assert out == [(0.0, 3.0), (10.0, 11.0)]


def test_overlap_seconds_with_union() -> None:
    union = [(0.0, 1.0), (3.0, 5.0)]
    assert overlap_seconds_with_union((0.5, 3.5), union) == 1.0  # [0.5,1.0] + [3.0,3.5]


def test_purity_vs_union_counts_overlap_and_non_call() -> None:
    gt = merge_intervals([(0.0, 10.0), (20.0, 30.0)])
    pred = [(5.0, 15.0), (25.0, 27.0)]
    agg = purity_vs_union(pred_intervals=pred, gt_union_intervals=gt)

    # pred seconds = 10 + 2 = 12
    assert agg.pred_seconds == 12.0
    # overlap = [5,10] (5s) + [25,27] (2s) = 7s
    assert agg.overlap_seconds == 7.0
    assert agg.non_call_seconds == 5.0
    assert agg.segment_purities == [0.5, 1.0]


def test_quantiles_empty() -> None:
    assert quantiles([], qs=[0.1, 0.5]) == [None, None]

