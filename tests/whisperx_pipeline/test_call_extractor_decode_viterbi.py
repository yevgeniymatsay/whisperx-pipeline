from __future__ import annotations

import numpy as np

from pipeline.call_extractor_wavlm.decode import DecodeConfig, probabilities_to_segments


def test_viterbi_produces_segment_from_in_call_region() -> None:
    times = np.arange(0.0, 10.0, 1.0, dtype=np.float32)
    in_call = np.full_like(times, 0.05, dtype=np.float32)
    in_call[2:6] = 0.95
    start = np.zeros_like(times, dtype=np.float32)
    end = np.zeros_like(times, dtype=np.float32)

    cfg = DecodeConfig(
        mode="viterbi",
        in_call_mean_min=0.2,
        min_duration_s=1.0,
        viterbi_off_to_on_penalty=2.0,
        viterbi_on_to_off_penalty=2.0,
        viterbi_min_on_s=1.0,
        viterbi_min_off_s=0.0,
        viterbi_smooth_win_s=0.0,
    )
    segs = probabilities_to_segments(times_s=times, in_call_p=in_call, start_p=start, end_p=end, cfg=cfg)
    assert len(segs) == 1
    assert segs[0]["start_s"] == 2.0
    assert segs[0]["end_s"] == 5.0


def test_viterbi_respects_min_on_duration() -> None:
    times = np.arange(0.0, 10.0, 1.0, dtype=np.float32)
    in_call = np.full_like(times, 0.05, dtype=np.float32)
    in_call[3] = 0.95  # single-frame blip
    start = np.zeros_like(times, dtype=np.float32)
    end = np.zeros_like(times, dtype=np.float32)

    cfg = DecodeConfig(
        mode="viterbi",
        in_call_mean_min=0.2,
        min_duration_s=1.0,
        viterbi_off_to_on_penalty=2.0,
        viterbi_on_to_off_penalty=2.0,
        viterbi_min_on_s=3.0,  # requires >=3s ON
        viterbi_min_off_s=0.0,
    )
    segs = probabilities_to_segments(times_s=times, in_call_p=in_call, start_p=start, end_p=end, cfg=cfg)
    assert segs == []


def test_in_call_threshold_decoder_closes_short_off_gaps() -> None:
    # ON region with a brief dip below threshold should remain one segment when min_off_s is enforced.
    times = np.arange(0.0, 10.0, 1.0, dtype=np.float32)
    in_call = np.full_like(times, 0.95, dtype=np.float32)
    in_call[4] = 0.05  # 1s OFF gap
    start = np.zeros_like(times, dtype=np.float32)
    end = np.zeros_like(times, dtype=np.float32)

    cfg = DecodeConfig(
        mode="in_call",
        in_call_threshold=0.5,
        in_call_mean_min=0.2,
        min_duration_s=1.0,
        in_call_min_on_s=1.0,
        in_call_min_off_s=2.0,  # require OFF >=2s to split; fill 1s gap
        in_call_smooth_win_s=0.0,
    )
    segs = probabilities_to_segments(times_s=times, in_call_p=in_call, start_p=start, end_p=end, cfg=cfg)
    assert len(segs) == 1
    assert segs[0]["start_s"] == 0.0
    assert segs[0]["end_s"] == 9.0
