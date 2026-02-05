"""Sequence decoding utilities for the call segmenter.

This module intentionally has *no* ML dependencies. It decodes a 2-state
sequence (NO_CALL vs CALL) from per-window call probabilities using a simple
Viterbi dynamic program.

We use the decoder as a principled replacement for heuristic post-processing
(threshold/hysteresis + gap merging) when segment split correctness matters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ViterbiParams:
    """Parameters for 2-state Viterbi decoding."""

    enter_cost: float  # NO_CALL -> CALL
    exit_cost: float  # CALL -> NO_CALL
    call_bias: float = 0.0  # per-step bias added to CALL emission cost (acts like a soft threshold)
    eps: float = 1e-6  # clamp probs to [eps, 1-eps] for log stability


def viterbi_decode_call_mask(
    probs: np.ndarray,
    params: ViterbiParams,
) -> Tuple[np.ndarray, float]:
    """Decode most-likely CALL/NO_CALL mask from per-window call probabilities.

    The objective minimized is:
      sum_t -log p(y_t | state_t)  +  sum_t transition_cost(state_{t-1} -> state_t)

    Where:
      emission_cost(CALL)    = -log(p_call)
      emission_cost(NO_CALL) = -log(1 - p_call)
      transition_cost(NO_CALL->CALL) = enter_cost
      transition_cost(CALL->NO_CALL) = exit_cost

    Args:
      probs: float array shape (T,) with per-window P(CALL).
      params: ViterbiParams

    Returns:
      (call_mask, best_cost)
        call_mask: bool array shape (T,), True for decoded CALL state.
        best_cost: float, total decoded cost (useful for debugging).
    """
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    t = int(p.shape[0])
    if t == 0:
        return np.zeros(0, dtype=bool), 0.0

    eps = float(params.eps)
    if eps <= 0.0 or eps >= 0.5:
        raise ValueError("eps must be in (0, 0.5)")

    p = np.clip(p, eps, 1.0 - eps)

    # Emission costs
    # call_bias acts like a prior/threshold: without transitions, CALL is preferred iff logit(p) >= call_bias.
    e_call = -np.log(p) + float(params.call_bias)
    e_nocall = -np.log(1.0 - p)

    enter_cost = float(params.enter_cost)
    exit_cost = float(params.exit_cost)

    # DP arrays: best cost up to t with state 0/1.
    dp0 = np.empty(t, dtype=np.float64)
    dp1 = np.empty(t, dtype=np.float64)
    # Backpointers: prev state for each state at each time (0 or 1).
    bp0 = np.empty(t, dtype=np.int8)
    bp1 = np.empty(t, dtype=np.int8)

    # Initialize: assume implicit start state is NO_CALL; starting in CALL pays enter_cost.
    dp0[0] = e_nocall[0]
    dp1[0] = e_call[0] + enter_cost
    bp0[0] = 0
    bp1[0] = 0

    for i in range(1, t):
        # End in NO_CALL
        stay0 = dp0[i - 1]
        switch10 = dp1[i - 1] + exit_cost
        if stay0 <= switch10:
            dp0[i] = e_nocall[i] + stay0
            bp0[i] = 0
        else:
            dp0[i] = e_nocall[i] + switch10
            bp0[i] = 1

        # End in CALL
        stay1 = dp1[i - 1]
        switch01 = dp0[i - 1] + enter_cost
        if stay1 <= switch01:
            dp1[i] = e_call[i] + stay1
            bp1[i] = 1
        else:
            dp1[i] = e_call[i] + switch01
            bp1[i] = 0

    # Backtrace
    if dp1[-1] <= dp0[-1]:
        state = 1
        best_cost = float(dp1[-1])
    else:
        state = 0
        best_cost = float(dp0[-1])

    states = np.empty(t, dtype=np.int8)
    for i in range(t - 1, -1, -1):
        states[i] = state
        if i == 0:
            break
        state = int(bp1[i]) if state == 1 else int(bp0[i])

    return states.astype(bool), best_cost
