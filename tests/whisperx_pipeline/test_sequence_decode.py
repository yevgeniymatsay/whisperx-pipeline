import numpy as np
import pytest

from pipeline.call_segmenter.sequence_decode import ViterbiParams, viterbi_decode_call_mask


def test_viterbi_all_low_probs_yields_no_call():
    probs = np.full(200, 0.01, dtype=np.float64)
    mask, cost = viterbi_decode_call_mask(probs, ViterbiParams(enter_cost=1.0, exit_cost=1.0))
    assert mask.dtype == bool
    assert mask.shape == probs.shape
    assert mask.sum() == 0
    assert cost >= 0.0


def test_viterbi_all_high_probs_yields_call():
    probs = np.full(200, 0.99, dtype=np.float64)
    mask, _ = viterbi_decode_call_mask(probs, ViterbiParams(enter_cost=1.0, exit_cost=1.0))
    assert mask.all()


def test_viterbi_single_high_block_yields_single_segment():
    probs = np.full(120, 0.02, dtype=np.float64)
    probs[30:60] = 0.98
    mask, _ = viterbi_decode_call_mask(probs, ViterbiParams(enter_cost=0.5, exit_cost=0.5))
    # Should decode at least the obvious high block and avoid long call spillover.
    assert mask[:20].sum() == 0
    assert mask[40:50].all()
    assert mask[-20:].sum() == 0


def test_viterbi_jitter_is_smoothed_by_switch_cost():
    probs = np.array([0.6, 0.4] * 100, dtype=np.float64)
    mask, _ = viterbi_decode_call_mask(probs, ViterbiParams(enter_cost=0.3, exit_cost=0.3))
    # With symmetric jitter and positive enter_cost, NO_CALL everywhere should win.
    assert mask.sum() == 0


def test_viterbi_eps_clamps_extremes():
    probs = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    mask, _ = viterbi_decode_call_mask(probs, ViterbiParams(enter_cost=1.0, exit_cost=1.0, eps=1e-6))
    assert mask.shape == probs.shape


def test_viterbi_invalid_eps_raises():
    probs = np.array([0.5, 0.5], dtype=np.float64)
    with pytest.raises(ValueError):
        viterbi_decode_call_mask(probs, ViterbiParams(enter_cost=1.0, exit_cost=1.0, eps=0.0))
