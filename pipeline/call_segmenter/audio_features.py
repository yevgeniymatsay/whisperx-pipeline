"""Audio-derived features for the call segmenter.

Design goals:
- Cheap features computed from decoded mono audio (16kHz float32 PCM).
- Deterministic + consistent between dataset building and inference.
- Avoid large intermediate WAVs (caller provides decoded PCM).

Features (per window):
- rms_energy
- log_rms
- log_rms_z (z-score of log_rms within the video)
- zcr (zero-crossing rate)
- low_band_frac (<300Hz)
- phone_band_frac (300-3400Hz)
- hf_energy_frac (>4000Hz)
- mid_hf_band_frac (3400-7000Hz)
- ultra_hf_frac (>7000Hz)
- hi_ratio_3p5_7k = E(3500-7000) / max(E(0-3500), eps)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class AudioFeatureConfig:
    """Configuration for audio feature computation."""

    enabled: bool = True
    sr_hz: int = 16000
    filter_order: int = 4

    # Band definitions (Hz). These are intentionally simple and robust.
    low_band_high_hz: float = 300.0
    phone_low_hz: float = 300.0
    phone_high_hz: float = 3400.0
    hf_low_hz: float = 4000.0
    mid_hf_low_hz: float = 3400.0
    mid_hf_high_hz: float = 7000.0
    ultra_hf_low_hz: float = 7000.0
    hi_ratio_split_hz: float = 3500.0
    hi_ratio_high_hz: float = 7000.0

    # Numerical stability / normalization.
    eps: float = 1e-8
    rms_log_eps: float = 1e-8
    rms_z_eps: float = 1e-6

    def to_meta(self) -> Dict:
        """Return a JSON-serializable config dict suitable for meta.json."""
        return {
            "enabled": bool(self.enabled),
            "sr_hz": int(self.sr_hz),
            "filter_order": int(self.filter_order),
            "bands_hz": {
                "low_band": [0.0, float(self.low_band_high_hz)],
                "phone_band": [float(self.phone_low_hz), float(self.phone_high_hz)],
                "mid_hf_band": [float(self.mid_hf_low_hz), float(self.mid_hf_high_hz)],
                "hf": [float(self.hf_low_hz), None],
                "ultra_hf": [float(self.ultra_hf_low_hz), None],
                "hi_ratio_num": [float(self.hi_ratio_split_hz), float(self.hi_ratio_high_hz)],
                "hi_ratio_den": [0.0, float(self.hi_ratio_split_hz)],
            },
            "rms_normalization": {"mode": "zscore", "eps": float(self.rms_z_eps)},
            "eps": float(self.eps),
            "rms_log_eps": float(self.rms_log_eps),
        }

    @staticmethod
    def from_meta(meta: Optional[Dict]) -> "AudioFeatureConfig":
        """Create config from a meta dict.

        Supports legacy meta formats that only include phone_band_hz + hf_low_hz.
        """
        if not meta:
            return AudioFeatureConfig(enabled=False)

        enabled = bool(meta.get("enabled", True))
        sr_hz = int(meta.get("sr_hz", 16000))
        order = int(meta.get("filter_order", 4))

        # Defaults
        cfg = {
            "enabled": enabled,
            "sr_hz": sr_hz,
            "filter_order": order,
            "low_band_high_hz": 300.0,
            "phone_low_hz": 300.0,
            "phone_high_hz": 3400.0,
            "hf_low_hz": float(meta.get("hf_low_hz", 4000.0)),
            "mid_hf_low_hz": 3400.0,
            "mid_hf_high_hz": 7000.0,
            "ultra_hf_low_hz": 7000.0,
            "hi_ratio_split_hz": 3500.0,
            "hi_ratio_high_hz": 7000.0,
            "eps": float(meta.get("eps", 1e-8)),
            "rms_log_eps": float(meta.get("rms_log_eps", 1e-8)),
            "rms_z_eps": float(meta.get("rms_normalization", {}).get("eps", 1e-6))
            if isinstance(meta.get("rms_normalization"), dict)
            else float(meta.get("rms_z_eps", 1e-6)),
        }

        # Legacy: phone_band_hz: [low, high]
        phone_band = meta.get("phone_band_hz")
        if isinstance(phone_band, (list, tuple)) and len(phone_band) == 2:
            cfg["phone_low_hz"] = float(phone_band[0])
            cfg["phone_high_hz"] = float(phone_band[1])

        # New: bands_hz
        bands = meta.get("bands_hz")
        if isinstance(bands, dict):
            try:
                low_band = bands.get("low_band")
                if isinstance(low_band, (list, tuple)) and len(low_band) == 2:
                    cfg["low_band_high_hz"] = float(low_band[1])

                phone_band = bands.get("phone_band")
                if isinstance(phone_band, (list, tuple)) and len(phone_band) == 2:
                    cfg["phone_low_hz"] = float(phone_band[0])
                    cfg["phone_high_hz"] = float(phone_band[1])

                mid_hf = bands.get("mid_hf_band")
                if isinstance(mid_hf, (list, tuple)) and len(mid_hf) == 2:
                    cfg["mid_hf_low_hz"] = float(mid_hf[0])
                    cfg["mid_hf_high_hz"] = float(mid_hf[1])

                ultra = bands.get("ultra_hf")
                if isinstance(ultra, (list, tuple)) and len(ultra) == 2:
                    cfg["ultra_hf_low_hz"] = float(ultra[0])

                hf = bands.get("hf")
                if isinstance(hf, (list, tuple)) and len(hf) == 2:
                    cfg["hf_low_hz"] = float(hf[0])

                num = bands.get("hi_ratio_num")
                if isinstance(num, (list, tuple)) and len(num) == 2:
                    cfg["hi_ratio_split_hz"] = float(num[0])
                    cfg["hi_ratio_high_hz"] = float(num[1])
            except Exception:
                # Be robust: if bands parsing fails, keep defaults/legacy values.
                pass

        return AudioFeatureConfig(**cfg)


def _to_1d_float32(audio: np.ndarray) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    return x


def _sample_indices(
    t_starts: Sequence[float],
    t_ends: Sequence[float],
    sr_hz: int,
    n_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    t_starts_arr = np.asarray(t_starts, dtype=np.float64)
    t_ends_arr = np.asarray(t_ends, dtype=np.float64)
    i0 = np.rint(t_starts_arr * float(sr_hz)).astype(np.int64)
    i1 = np.rint(t_ends_arr * float(sr_hz)).astype(np.int64)
    i0 = np.clip(i0, 0, n_samples)
    i1 = np.clip(i1, 0, n_samples)
    return i0, i1


def _energy_per_window_from_signal(
    x: np.ndarray,
    i0: np.ndarray,
    i1: np.ndarray,
) -> np.ndarray:
    """Compute sum(x^2) for each window using a float64 prefix sum.

    Uses a single prefix buffer to avoid allocating a separate squared array.
    """
    n = int(x.shape[0])
    prefix = np.empty(n + 1, dtype=np.float64)
    prefix[0] = 0.0
    # Store squared values directly into the prefix buffer, then cumsum in place.
    np.multiply(x, x, out=prefix[1:])
    np.cumsum(prefix[1:], dtype=np.float64, out=prefix[1:])
    e = prefix[i1] - prefix[i0]
    return e


def _band_energy_per_window(
    x: np.ndarray,
    sr_hz: int,
    order: int,
    btype: str,
    cutoff: Sequence[float],
    i0: np.ndarray,
    i1: np.ndarray,
) -> np.ndarray:
    sos = signal.butter(order, cutoff, btype=btype, fs=sr_hz, output="sos")
    x_f = signal.sosfilt(sos, x).astype(np.float32, copy=False)
    e = _energy_per_window_from_signal(x_f, i0, i1)
    return e


def compute_audio_features_for_windows(
    audio: np.ndarray,
    t_starts: Sequence[float],
    t_ends: Sequence[float],
    config: AudioFeatureConfig,
    required_columns: Optional[Iterable[str]] = None,
) -> Dict[str, np.ndarray]:
    """Compute audio features for many windows in a video.

    Args:
        audio: 1D float32 PCM at config.sr_hz (mono).
        t_starts / t_ends: per-window bounds in seconds (same length).
        config: audio feature config
        required_columns: if provided, only compute those feature columns.

    Returns:
        Dict mapping feature name -> float32 array length n_windows.
    """
    required: Optional[set] = set(required_columns) if required_columns is not None else None

    if not config.enabled:
        # Return zeros for requested columns.
        n = len(t_starts)
        out: Dict[str, np.ndarray] = {}
        for k in (required or []):
            out[k] = np.zeros(n, dtype=np.float32)
        return out

    x = _to_1d_float32(audio)
    n_samples = int(x.shape[0])
    i0, i1 = _sample_indices(t_starts, t_ends, config.sr_hz, n_samples)
    win_n = np.maximum(i1 - i0, 0)
    valid = win_n > 0

    out: Dict[str, np.ndarray] = {}

    # Total energy + RMS are used by many features; compute once.
    total_e = _energy_per_window_from_signal(x, i0, i1)
    total_e = np.where(valid, total_e, 0.0)

    # RMS energy
    if required is None or "rms_energy" in required or "log_rms" in required or "log_rms_z" in required:
        rms = np.zeros_like(total_e, dtype=np.float64)
        rms[valid] = np.sqrt(np.maximum(total_e[valid] / np.maximum(win_n[valid], 1), 0.0))
        if required is None or "rms_energy" in required:
            out["rms_energy"] = rms.astype(np.float32)

        log_rms = None
        if required is None or "log_rms" in required or "log_rms_z" in required:
            log_rms = np.log(rms + float(config.rms_log_eps))
            if required is None or "log_rms" in required:
                out["log_rms"] = log_rms.astype(np.float32)

        if required is None or "log_rms_z" in required:
            if log_rms is None:
                log_rms = np.log(rms + float(config.rms_log_eps))
            mean = float(np.mean(log_rms)) if log_rms.size else 0.0
            std = float(np.std(log_rms)) if log_rms.size else 0.0
            denom = std + float(config.rms_z_eps)
            if denom <= 0.0:
                z = np.zeros_like(log_rms, dtype=np.float32)
            else:
                z = ((log_rms - mean) / denom).astype(np.float32)
            out["log_rms_z"] = z

    # ZCR
    if required is None or "zcr" in required:
        if n_samples <= 1:
            out["zcr"] = np.zeros(len(i0), dtype=np.float32)
        else:
            sign = x >= 0.0
            changes = sign[1:] != sign[:-1]  # len n_samples - 1
            prefix = np.empty(n_samples, dtype=np.uint32)
            prefix[0] = 0
            # prefix[k] = sum(changes[:k]) for k>=1
            np.cumsum(changes.astype(np.uint8), dtype=np.uint32, out=prefix[1:])
            # transitions within window: changes[i0 : i1-1]
            b = np.clip(i1 - 1, 0, n_samples - 1)
            a = np.clip(i0, 0, n_samples - 1)
            cnt = prefix[b] - prefix[a]
            cnt = np.where(win_n > 1, cnt.astype(np.float64), 0.0)
            zcr = np.zeros_like(total_e, dtype=np.float64)
            zcr[valid] = cnt[valid] / np.maximum(win_n[valid], 1)
            out["zcr"] = zcr.astype(np.float32)

    # Helper for fractions.
    denom_total = total_e + float(config.eps)

    def frac(name: str, e_band: np.ndarray) -> None:
        if required is None or name in required:
            f = np.zeros_like(e_band, dtype=np.float64)
            f[valid] = e_band[valid] / denom_total[valid]
            # Clamp: filtering can lead to tiny overshoots.
            f = np.clip(f, 0.0, 1.0)
            out[name] = f.astype(np.float32)

    # low_band_frac (<300Hz): lowpass
    if required is None or "low_band_frac" in required:
        e_low = _band_energy_per_window(
            x,
            sr_hz=config.sr_hz,
            order=config.filter_order,
            btype="lowpass",
            cutoff=[float(config.low_band_high_hz)],
            i0=i0,
            i1=i1,
        )
        e_low = np.where(valid, e_low, 0.0)
        frac("low_band_frac", e_low)

    # phone_band_frac (300-3400Hz): bandpass
    if required is None or "phone_band_frac" in required:
        e_phone = _band_energy_per_window(
            x,
            sr_hz=config.sr_hz,
            order=config.filter_order,
            btype="bandpass",
            cutoff=[float(config.phone_low_hz), float(config.phone_high_hz)],
            i0=i0,
            i1=i1,
        )
        e_phone = np.where(valid, e_phone, 0.0)
        frac("phone_band_frac", e_phone)

    # hf_energy_frac (>4000Hz): highpass
    if required is None or "hf_energy_frac" in required:
        e_hf = _band_energy_per_window(
            x,
            sr_hz=config.sr_hz,
            order=config.filter_order,
            btype="highpass",
            cutoff=[float(config.hf_low_hz)],
            i0=i0,
            i1=i1,
        )
        e_hf = np.where(valid, e_hf, 0.0)
        frac("hf_energy_frac", e_hf)

    # mid_hf_band_frac (3400-7000Hz): bandpass
    if required is None or "mid_hf_band_frac" in required:
        e_mid_hf = _band_energy_per_window(
            x,
            sr_hz=config.sr_hz,
            order=config.filter_order,
            btype="bandpass",
            cutoff=[float(config.mid_hf_low_hz), float(config.mid_hf_high_hz)],
            i0=i0,
            i1=i1,
        )
        e_mid_hf = np.where(valid, e_mid_hf, 0.0)
        frac("mid_hf_band_frac", e_mid_hf)

    # ultra_hf_frac (>7000Hz): highpass
    if required is None or "ultra_hf_frac" in required:
        e_ultra = _band_energy_per_window(
            x,
            sr_hz=config.sr_hz,
            order=config.filter_order,
            btype="highpass",
            cutoff=[float(config.ultra_hf_low_hz)],
            i0=i0,
            i1=i1,
        )
        e_ultra = np.where(valid, e_ultra, 0.0)
        frac("ultra_hf_frac", e_ultra)

    # hi_ratio_3p5_7k = E(3500-7000) / E(0-3500)
    if required is None or "hi_ratio_3p5_7k" in required:
        e_num = _band_energy_per_window(
            x,
            sr_hz=config.sr_hz,
            order=config.filter_order,
            btype="bandpass",
            cutoff=[float(config.hi_ratio_split_hz), float(config.hi_ratio_high_hz)],
            i0=i0,
            i1=i1,
        )
        e_den = _band_energy_per_window(
            x,
            sr_hz=config.sr_hz,
            order=config.filter_order,
            btype="lowpass",
            cutoff=[float(config.hi_ratio_split_hz)],
            i0=i0,
            i1=i1,
        )
        e_num = np.where(valid, e_num, 0.0)
        e_den = np.where(valid, e_den, 0.0)

        ratio = np.zeros_like(e_num, dtype=np.float64)
        ratio[valid] = e_num[valid] / np.maximum(e_den[valid], float(config.eps))
        out["hi_ratio_3p5_7k"] = ratio.astype(np.float32)

    return out

