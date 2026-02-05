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
- spec_centroid_hz, spec_centroid_z
- spec_rolloff_hz, spec_rolloff_z
- mfcc_00..mfcc_12, mfcc_z_00..mfcc_z_12
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import fft, signal


@dataclass(frozen=True)
class AudioFeatureConfig:
    """Configuration for audio feature computation."""

    enabled: bool = True
    sr_hz: int = 16000
    filter_order: int = 4

    # Spectral features (centroid + rolloff) computed from an rFFT of the window.
    spectral_enabled: bool = True
    spectral_n_fft: int = 16384
    spectral_rolloff_pct: float = 0.85
    spectral_fmin_hz: float = 0.0
    spectral_fmax_hz: Optional[float] = None  # default: sr/2

    # MFCC features (computed from log-mel energies + DCT).
    mfcc_enabled: bool = False
    mfcc_n_fft: int = 16384
    mfcc_n_mels: int = 40
    mfcc_n_mfcc: int = 13
    mfcc_fmin_hz: float = 0.0
    mfcc_fmax_hz: Optional[float] = None  # default: sr/2
    mfcc_mel_eps: float = 1e-10

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
            "spectral": {
                "enabled": bool(self.spectral_enabled),
                "n_fft": int(self.spectral_n_fft),
                "rolloff_pct": float(self.spectral_rolloff_pct),
                "fmin_hz": float(self.spectral_fmin_hz),
                "fmax_hz": float(self.spectral_fmax_hz) if self.spectral_fmax_hz is not None else None,
            },
            "mfcc": {
                "enabled": bool(self.mfcc_enabled),
                "n_fft": int(self.mfcc_n_fft),
                "n_mels": int(self.mfcc_n_mels),
                "n_mfcc": int(self.mfcc_n_mfcc),
                "fmin_hz": float(self.mfcc_fmin_hz),
                "fmax_hz": float(self.mfcc_fmax_hz) if self.mfcc_fmax_hz is not None else None,
                "mel_eps": float(self.mfcc_mel_eps),
            },
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

        spectral = meta.get("spectral") if isinstance(meta.get("spectral"), dict) else {}
        mfcc = meta.get("mfcc") if isinstance(meta.get("mfcc"), dict) else {}

        # Defaults
        cfg = {
            "enabled": enabled,
            "sr_hz": sr_hz,
            "filter_order": order,
            "spectral_enabled": bool(spectral.get("enabled", True)),
            "spectral_n_fft": int(spectral.get("n_fft", 16384)),
            "spectral_rolloff_pct": float(spectral.get("rolloff_pct", 0.85)),
            "spectral_fmin_hz": float(spectral.get("fmin_hz", 0.0)),
            "spectral_fmax_hz": float(spectral.get("fmax_hz")) if spectral.get("fmax_hz") is not None else None,
            "mfcc_enabled": bool(mfcc.get("enabled", False)),
            "mfcc_n_fft": int(mfcc.get("n_fft", 16384)),
            "mfcc_n_mels": int(mfcc.get("n_mels", 40)),
            "mfcc_n_mfcc": int(mfcc.get("n_mfcc", 13)),
            "mfcc_fmin_hz": float(mfcc.get("fmin_hz", 0.0)),
            "mfcc_fmax_hz": float(mfcc.get("fmax_hz")) if mfcc.get("fmax_hz") is not None else None,
            "mfcc_mel_eps": float(mfcc.get("mel_eps", 1e-10)),
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


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + (hz / 700.0))


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (np.power(10.0, mel / 2595.0) - 1.0)


def _mel_filterbank(
    *,
    sr_hz: int,
    n_fft: int,
    n_mels: int,
    fmin_hz: float,
    fmax_hz: float,
) -> np.ndarray:
    """Create a simple triangular mel filterbank.

    Returns:
        float32 array shape (n_mels, n_freq_bins) suitable for multiplying
        power spectra from rfft(n_fft).
    """
    if n_mels <= 0:
        raise ValueError("n_mels must be > 0")
    if n_fft <= 0:
        raise ValueError("n_fft must be > 0")

    n_freq = n_fft // 2 + 1
    fmin_hz = float(max(0.0, fmin_hz))
    fmax_hz = float(min(fmax_hz, sr_hz / 2.0))
    if fmax_hz <= fmin_hz:
        raise ValueError("fmax_hz must be > fmin_hz")

    # Mel-spaced points (include endpoints).
    mmin = _hz_to_mel(np.array([fmin_hz], dtype=np.float64))[0]
    mmax = _hz_to_mel(np.array([fmax_hz], dtype=np.float64))[0]
    m_pts = np.linspace(mmin, mmax, n_mels + 2, dtype=np.float64)
    f_pts = _mel_to_hz(m_pts)

    # FFT bin indices for each mel point.
    bins = np.floor((n_fft + 1) * f_pts / float(sr_hz)).astype(np.int64)
    bins = np.clip(bins, 0, n_freq - 1)

    fb = np.zeros((n_mels, n_freq), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left = int(bins[m - 1])
        center = int(bins[m])
        right = int(bins[m + 1])
        if center <= left or right <= center:
            continue
        up = np.arange(left, center, dtype=np.float32)
        down = np.arange(center, right, dtype=np.float32)
        fb[m - 1, left:center] = (up - float(left)) / float(center - left)
        fb[m - 1, center:right] = (float(right) - down) / float(right - center)

    return fb


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
            vals = log_rms[valid]
            mean = float(np.mean(vals)) if vals.size else 0.0
            std = float(np.std(vals)) if vals.size else 0.0
            denom = std + float(config.rms_z_eps)
            if denom <= 0.0:
                z = np.zeros_like(log_rms, dtype=np.float32)
            else:
                z = ((log_rms - mean) / denom).astype(np.float32)
                z[~valid] = 0.0
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

    # Spectral features / MFCCs. These are more expensive than band-energy features.
    spec_cols = {"spec_centroid_hz", "spec_centroid_z", "spec_rolloff_hz", "spec_rolloff_z"}
    if required is not None and (not config.spectral_enabled) and any(c in required for c in spec_cols):
        raise ValueError("Spectral features requested but spectral_enabled is False in config/meta")
    need_spec = bool(config.spectral_enabled) and (required is None or any(c in required for c in spec_cols))

    mfcc_requested = False
    if required is not None:
        mfcc_requested = any((c.startswith("mfcc_") or c.startswith("mfcc_z_")) for c in required)
    if mfcc_requested and (not config.mfcc_enabled):
        raise ValueError("MFCC features requested but mfcc_enabled is False in config/meta")
    need_mfcc = bool(config.mfcc_enabled) and (
        required is None or any((c.startswith("mfcc_") or c.startswith("mfcc_z_")) for c in (required or []))
    )

    if need_spec or need_mfcc:
        idxs = np.nonzero(valid)[0]
        if idxs.size:
            frame_len = int(np.rint(float(np.median(win_n[valid]))))
        else:
            frame_len = 0
        frame_len = max(frame_len, 1)

        # FFT settings. We expect the spectral + MFCC FFT sizes to match for efficiency.
        n_fft_spec = int(config.spectral_n_fft)
        n_fft_mfcc = int(config.mfcc_n_fft)
        if need_spec and need_mfcc and n_fft_spec != n_fft_mfcc:
            raise ValueError("spectral_n_fft and mfcc_n_fft must match when both spectral and MFCC features are used")
        n_fft = n_fft_spec if need_spec else n_fft_mfcc

        if n_fft < frame_len:
            # Be safe: zero-padding doesn't help if the FFT is smaller than the window.
            n_fft = 1 << int(math.ceil(math.log2(frame_len)))

        # Windowing to reduce spectral leakage.
        hann = signal.windows.hann(frame_len, sym=False).astype(np.float32, copy=False)

        # Precompute frequency bins.
        freqs_full = np.fft.rfftfreq(n_fft, d=1.0 / float(config.sr_hz)).astype(np.float32, copy=False)

        # Spectral mask (defaults to full band).
        spec_fmin = float(config.spectral_fmin_hz)
        spec_fmax = float(config.spectral_fmax_hz) if config.spectral_fmax_hz is not None else (float(config.sr_hz) / 2.0)
        spec_fmin = max(0.0, spec_fmin)
        spec_fmax = min(spec_fmax, float(config.sr_hz) / 2.0)
        if spec_fmax <= spec_fmin:
            spec_fmin, spec_fmax = 0.0, float(config.sr_hz) / 2.0

        spec_mask = None
        if spec_fmin > 0.0 or spec_fmax < float(config.sr_hz) / 2.0 - 1e-3:
            spec_mask = (freqs_full >= spec_fmin) & (freqs_full <= spec_fmax)

        spec_centroid = np.zeros(len(i0), dtype=np.float32) if need_spec else None
        spec_rolloff = np.zeros(len(i0), dtype=np.float32) if need_spec else None
        mfcc_mat = np.zeros((len(i0), int(config.mfcc_n_mfcc)), dtype=np.float32) if need_mfcc else None

        # MFCC filterbank.
        fb_T = None
        if need_mfcc:
            mfcc_fmin = float(config.mfcc_fmin_hz)
            mfcc_fmax = float(config.mfcc_fmax_hz) if config.mfcc_fmax_hz is not None else (float(config.sr_hz) / 2.0)
            mfcc_fmin = max(0.0, mfcc_fmin)
            mfcc_fmax = min(mfcc_fmax, float(config.sr_hz) / 2.0)
            fb = _mel_filterbank(
                sr_hz=int(config.sr_hz),
                n_fft=int(n_fft),
                n_mels=int(config.mfcc_n_mels),
                fmin_hz=mfcc_fmin,
                fmax_hz=mfcc_fmax,
            )
            fb_T = fb.T  # (n_freq, n_mels)

        # Process in chunks to bound memory.
        chunk_size = 128
        for start in range(0, int(idxs.size), chunk_size):
            chunk_idxs = idxs[start : start + chunk_size]
            bsz = int(chunk_idxs.size)
            frames = np.zeros((bsz, frame_len), dtype=np.float32)
            for j, wi in enumerate(chunk_idxs):
                s = int(i0[wi])
                e = int(i1[wi])
                if e <= s:
                    continue
                seg = x[s:e]
                n = min(int(seg.size), frame_len)
                if n > 0:
                    frames[j, :n] = seg[:n]

            frames *= hann[None, :]
            spec = fft.rfft(frames, n=n_fft, axis=1)
            power = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float32, copy=False)

            total_power = power.sum(axis=1).astype(np.float32, copy=False)
            has_power = total_power > float(config.eps)

            if need_spec and spec_centroid is not None and spec_rolloff is not None:
                if spec_mask is None:
                    p_use = power
                    f_use = freqs_full
                    tot_use = total_power
                else:
                    p_use = power[:, spec_mask]
                    f_use = freqs_full[spec_mask]
                    tot_use = p_use.sum(axis=1).astype(np.float32, copy=False)

                denom = tot_use + float(config.eps)
                centroid = np.zeros(bsz, dtype=np.float32)
                centroid[has_power] = (p_use[has_power] * f_use[None, :]).sum(axis=1) / denom[has_power]

                rolloff = np.zeros(bsz, dtype=np.float32)
                roll_pct = float(config.spectral_rolloff_pct)
                if roll_pct <= 0.0:
                    roll_pct = 0.85
                roll_pct = min(max(roll_pct, 0.5), 0.99)

                if np.any(has_power):
                    csum = np.cumsum(p_use[has_power], axis=1)
                    thresh = roll_pct * csum[:, -1]
                    # First bin where cumulative energy crosses threshold.
                    idx = (csum >= thresh[:, None]).argmax(axis=1)
                    rolloff_vals = f_use[idx]
                    rolloff[has_power] = rolloff_vals.astype(np.float32, copy=False)

                spec_centroid[chunk_idxs] = centroid
                spec_rolloff[chunk_idxs] = rolloff

            if need_mfcc and mfcc_mat is not None and fb_T is not None:
                mel = power @ fb_T  # (bsz, n_mels)
                log_mel = np.log(mel + float(config.mfcc_mel_eps)).astype(np.float32, copy=False)
                mfcc = fft.dct(log_mel, type=2, axis=1, norm="ortho")[:, : int(config.mfcc_n_mfcc)]
                mfcc = mfcc.astype(np.float32, copy=False)
                # Silence: avoid a constant log(eps) offset dominating; emit zeros.
                mfcc[~has_power, :] = 0.0
                mfcc_mat[chunk_idxs, :] = mfcc

        # Write spectral outputs (+ per-video z-score).
        if need_spec and spec_centroid is not None and spec_rolloff is not None:
            if required is None or "spec_centroid_hz" in required:
                out["spec_centroid_hz"] = spec_centroid
            if required is None or "spec_rolloff_hz" in required:
                out["spec_rolloff_hz"] = spec_rolloff

            if required is None or "spec_centroid_z" in required:
                std = float(np.std(spec_centroid[valid])) if np.any(valid) else 0.0
                denom = std + float(config.rms_z_eps)
                if denom <= 0.0:
                    out["spec_centroid_z"] = np.zeros_like(spec_centroid, dtype=np.float32)
                else:
                    mean = float(np.mean(spec_centroid[valid])) if np.any(valid) else 0.0
                    z = ((spec_centroid - mean) / denom).astype(np.float32, copy=False)
                    z[~valid] = 0.0
                    out["spec_centroid_z"] = z

            if required is None or "spec_rolloff_z" in required:
                std = float(np.std(spec_rolloff[valid])) if np.any(valid) else 0.0
                denom = std + float(config.rms_z_eps)
                if denom <= 0.0:
                    out["spec_rolloff_z"] = np.zeros_like(spec_rolloff, dtype=np.float32)
                else:
                    mean = float(np.mean(spec_rolloff[valid])) if np.any(valid) else 0.0
                    z = ((spec_rolloff - mean) / denom).astype(np.float32, copy=False)
                    z[~valid] = 0.0
                    out["spec_rolloff_z"] = z

        # Write MFCC outputs (+ per-video z-score).
        if need_mfcc and mfcc_mat is not None:
            n_mfcc = int(mfcc_mat.shape[1])

            # Per-video z-score per coefficient.
            mfcc_z = None
            if required is None or any(c.startswith("mfcc_z_") for c in (required or [])):
                means = np.mean(mfcc_mat[valid], axis=0) if np.any(valid) else np.zeros(n_mfcc, dtype=np.float32)
                stds = np.std(mfcc_mat[valid], axis=0) if np.any(valid) else np.zeros(n_mfcc, dtype=np.float32)
                denom = stds + float(config.rms_z_eps)
                mfcc_z = (mfcc_mat - means[None, :]) / denom[None, :]
                # If std is ~0 (e.g., single window), force z to 0.
                mfcc_z[:, stds <= 0.0] = 0.0
                mfcc_z[~valid, :] = 0.0
                mfcc_z = mfcc_z.astype(np.float32, copy=False)

            for k in range(n_mfcc):
                name = f"mfcc_{k:02d}"
                if required is None or name in required:
                    out[name] = mfcc_mat[:, k].astype(np.float32, copy=False)
                name_z = f"mfcc_z_{k:02d}"
                if mfcc_z is not None and (required is None or name_z in required):
                    out[name_z] = mfcc_z[:, k]

    return out
