"""Probability calibration helpers for the call segmenter.

Currently we support Platt scaling (a sigmoid applied to the model's logit).

The intent is to make per-window probabilities more comparable across videos so
sequence decoders (e.g., Viterbi/HMM) and thresholds are easier to tune.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class PlattCalibration:
    """Sigmoid calibration on the logit of the raw probability.

    p_cal = sigmoid(a * logit(p_raw) + b)
    """

    a: float
    b: float
    eps: float = 1e-6

    def to_dict(self) -> Dict[str, Any]:
        return {"method": "platt", "a": float(self.a), "b": float(self.b), "eps": float(self.eps)}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PlattCalibration":
        if d.get("method") not in (None, "platt"):
            raise ValueError(f"Unsupported calibration method: {d.get('method')}")
        return PlattCalibration(a=float(d["a"]), b=float(d["b"]), eps=float(d.get("eps", 1e-6)))


def load_calibration(path: Path) -> PlattCalibration:
    data = json.loads(path.read_text())
    return PlattCalibration.from_dict(data)


def apply_calibration(probs: np.ndarray, calib: Optional[PlattCalibration]) -> np.ndarray:
    """Apply calibration to raw probabilities.

    Args:
        probs: array-like of P(CALL) in [0,1]
        calib: calibration params; if None, returns probs as-is (float64)

    Returns:
        Calibrated probabilities as float64 ndarray, same shape as probs.
    """
    p = np.asarray(probs, dtype=np.float64)
    if calib is None:
        return p

    eps = float(calib.eps)
    if eps <= 0.0 or eps >= 0.5:
        raise ValueError("calibration eps must be in (0, 0.5)")

    p = np.clip(p, eps, 1.0 - eps)
    logit = np.log(p / (1.0 - p))
    z = float(calib.a) * logit + float(calib.b)
    # Numerically stable sigmoid for large magnitude inputs.
    out = 1.0 / (1.0 + np.exp(-z))
    return out

