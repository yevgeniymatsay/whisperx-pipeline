from __future__ import annotations

from pipeline.call_extractor_wavlm.production_metrics import (
    merge_intervals,
    overlap_seconds_between_unions,
    overlap_seconds_with_union,
    per_call_coverages,
    purity_vs_union,
    quantiles,
    total_seconds,
)


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


def test_per_call_coverages_uses_pred_union() -> None:
    # GT has two calls; pred has two segments that overlap the first call, and one that overlaps the second.
    gt_calls = [(0.0, 10.0), (20.0, 30.0)]
    pred = [(0.0, 4.0), (3.0, 10.0), (22.0, 28.0)]
    cov = per_call_coverages(gt_calls=gt_calls, pred_intervals=pred)
    assert cov == [1.0, 0.6]  # [0,10] fully covered by union; [20,30] has 6s of overlap.


def test_overlap_seconds_between_unions() -> None:
    a = merge_intervals([(0.0, 2.0), (10.0, 11.0)])
    b = merge_intervals([(1.0, 3.0), (10.5, 12.0)])
    assert overlap_seconds_between_unions(a, b) == 1.5  # [1,2] + [10.5,11]
    assert total_seconds(a) == 3.0
