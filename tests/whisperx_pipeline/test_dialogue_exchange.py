from __future__ import annotations

from pipeline.call_extractor_wavlm.dialogue_exchange import Turn, compute_t_bridge_s, dialogue_exchange_intervals


def test_compute_t_bridge_clips_to_min() -> None:
    turns = [
        Turn(start_s=0.0, end_s=5.0, speaker="S0"),
        Turn(start_s=5.5, end_s=10.0, speaker="S1"),
        Turn(start_s=10.2, end_s=15.0, speaker="S0"),
    ]
    # p90(gaps) is < 2s, so T_bridge should clip to min_s.
    tb = compute_t_bridge_s(turns, gap_pctl=90.0, min_s=2.0, max_s=15.0)
    assert tb == 2.0


def test_dialogue_exchange_intervals_detects_back_and_forth() -> None:
    turns = [
        Turn(start_s=0.0, end_s=5.0, speaker="S0"),
        Turn(start_s=5.5, end_s=10.0, speaker="S1"),
        Turn(start_s=10.2, end_s=15.0, speaker="S0"),
    ]
    tb, intervals = dialogue_exchange_intervals(turns, clip_duration_s=20.0, dt_s=0.1, t_bridge_min_s=2.0, t_bridge_max_s=15.0)
    assert tb == 2.0
    assert intervals

    # With T_bridge clipped to 2s, "exchange" only holds near alternations: it drops
    # during long single-speaker stretches and turns back on at the next alternation.
    assert len(intervals) >= 2
    s0, e0 = intervals[0]
    s1, e1 = intervals[1]
    assert abs(s0 - 5.5) <= 0.2
    assert abs(e0 - 7.0) <= 0.3
    assert abs(s1 - 10.2) <= 0.2
    assert abs(e1 - 12.0) <= 0.3


def test_dialogue_exchange_intervals_empty_for_single_speaker() -> None:
    turns = [
        Turn(start_s=0.0, end_s=8.0, speaker="S0"),
        Turn(start_s=9.0, end_s=12.0, speaker="S0"),
    ]
    tb, intervals = dialogue_exchange_intervals(turns, clip_duration_s=15.0, dt_s=0.1)
    assert tb >= 2.0
    assert intervals == []
