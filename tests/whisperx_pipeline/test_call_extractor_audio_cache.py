from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline.call_extractor_wavlm.audio_cache import read_flac_segment_float32


def test_read_flac_segment_matches_sample_slice(tmp_path: Path) -> None:
    try:
        import soundfile as sf
    except Exception as e:  # pragma: no cover
        raise RuntimeError("soundfile is required for this test") from e

    sr = 16000
    t = np.arange(sr * 2, dtype=np.float32) / float(sr)
    audio = (0.1 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)

    flac_path = tmp_path / "test.flac"
    sf.write(str(flac_path), audio, sr, format="FLAC", subtype="PCM_16")

    start_s = 0.5
    dur_s = 0.25
    seg = read_flac_segment_float32(flac_path, start_s=start_s, duration_s=dur_s, sr_hz=sr)

    i0 = int(round(start_s * sr))
    n = int(round(dur_s * sr))
    expected = audio[i0 : i0 + n]

    assert seg.shape == expected.shape
    assert np.max(np.abs(seg - expected)) < 5e-3

