from __future__ import annotations

import numpy as np

from pipeline.call_extractor_wavlm.decode_prod import ProdDecodeConfig, decode_production_segments


def test_prod_decoder_splits_on_valley_when_viterbi_stays_on() -> None:
    # Force Viterbi to stay ON by making switching extremely expensive.
    times = np.arange(0.0, 10.0, 0.5, dtype=np.float32)
    in_call = np.ones_like(times, dtype=np.float32) * 0.99
    # Insert a sustained low valley (~2s) that should trigger the valley splitter.
    valley = (times >= 4.0) & (times <= 5.5)
    in_call[valley] = 0.20

    cfg = ProdDecodeConfig(
        viterbi_off_to_on_penalty=999.0,
        viterbi_on_to_off_penalty=999.0,
        viterbi_cost_mult=1.0,
        viterbi_min_on_s=0.0,
        viterbi_min_off_s=0.0,
        viterbi_smooth_win_s=0.0,
        split_lo=0.35,
        split_min_off_s=1.0,
        min_duration_s=0.5,
        max_segment_s=10_000.0,
        mean_in_call_min=0.0,
        max_in_call_min=0.0,
        trim_s=0.0,
    )
    segs = decode_production_segments(times_s=times, in_call_p=in_call, cfg=cfg)
    assert len(segs) == 2
    assert segs[0]["was_split"] is True
    assert segs[0]["split_reason"] in ("valley", "max_segment+valley", "max_segment")
    assert segs[0]["end_s"] <= segs[1]["start_s"]


def test_prod_decoder_confidence_filter_drops_low_mean_segments() -> None:
    times = np.arange(0.0, 6.0, 1.0, dtype=np.float32)
    in_call = np.ones_like(times, dtype=np.float32) * 0.40

    cfg = ProdDecodeConfig(
        viterbi_off_to_on_penalty=0.0,
        viterbi_on_to_off_penalty=0.0,
        viterbi_cost_mult=1.0,
        viterbi_min_on_s=0.0,
        viterbi_min_off_s=0.0,
        viterbi_smooth_win_s=0.0,
        split_lo=0.2,
        split_min_off_s=10.0,
        min_duration_s=0.5,
        max_segment_s=10_000.0,
        mean_in_call_min=0.50,
        max_in_call_min=0.0,
        trim_s=0.0,
    )
    segs = decode_production_segments(times_s=times, in_call_p=in_call, cfg=cfg)
    assert segs == []


def test_prod_decoder_trim_shrinks_segments() -> None:
    times = np.arange(0.0, 10.0, 1.0, dtype=np.float32)
    in_call = np.ones_like(times, dtype=np.float32) * 0.99

    cfg = ProdDecodeConfig(
        viterbi_off_to_on_penalty=999.0,
        viterbi_on_to_off_penalty=999.0,
        viterbi_cost_mult=1.0,
        viterbi_min_on_s=0.0,
        viterbi_min_off_s=0.0,
        viterbi_smooth_win_s=0.0,
        split_lo=0.0,
        split_min_off_s=10.0,
        min_duration_s=0.5,
        max_segment_s=10_000.0,
        mean_in_call_min=0.0,
        max_in_call_min=0.0,
        trim_s=1.0,
    )
    segs = decode_production_segments(times_s=times, in_call_p=in_call, cfg=cfg)
    assert len(segs) == 1
    assert segs[0]["start_s"] == 1.0
    assert segs[0]["end_s"] == 8.0


def test_prod_decoder_boundary_cues_split_when_in_call_never_dips_low() -> None:
    times = np.arange(0.0, 20.0, 0.5, dtype=np.float32)
    in_call = np.ones_like(times, dtype=np.float32) * 0.99
    # Add a shallow dip between end->start peaks so the boundary splitter has an argmin.
    dip = (times >= 10.0) & (times <= 10.5)
    in_call[dip] = 0.80

    start = np.zeros_like(times, dtype=np.float32)
    end = np.zeros_like(times, dtype=np.float32)
    end[int(np.argmin(np.abs(times - 9.0)))] = 0.90
    start[int(np.argmin(np.abs(times - 11.0)))] = 0.90

    cfg = ProdDecodeConfig(
        viterbi_off_to_on_penalty=999.0,
        viterbi_on_to_off_penalty=999.0,
        viterbi_cost_mult=1.0,
        viterbi_min_on_s=0.0,
        viterbi_min_off_s=0.0,
        viterbi_smooth_win_s=0.0,
        split_lo=0.0,  # disable valley splitting
        split_min_off_s=10.0,
        use_boundary_cues=True,
        boundary_mode="pair_end_start",
        boundary_start_thr=0.50,
        boundary_end_thr=0.50,
        boundary_smooth_win_s=0.0,
        boundary_nms_sep_s=0.0,
        boundary_pair_max_gap_s=5.0,
        boundary_split_margin_s=0.0,
        boundary_split_gap_s=0.20,
        min_duration_s=0.5,
        max_segment_s=10_000.0,
        mean_in_call_min=0.0,
        max_in_call_min=0.0,
        trim_s=0.0,
    )

    segs = decode_production_segments(times_s=times, in_call_p=in_call, start_p=start, end_p=end, cfg=cfg)
    assert len(segs) == 2
    assert segs[0]["was_split"] is True
    assert "boundary" in str(segs[0]["split_reason"])
    assert segs[0]["end_s"] <= segs[1]["start_s"]
