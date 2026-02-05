import numpy as np

from pipeline.call_segmenter.audio_features import AudioFeatureConfig, compute_audio_features_for_windows


def test_audio_features_silence_has_zero_spectral_and_mfcc():
    sr = 16000
    audio = np.zeros(sr, dtype=np.float32)
    cfg = AudioFeatureConfig(
        enabled=True,
        sr_hz=sr,
        spectral_enabled=True,
        mfcc_enabled=True,
    )

    cols = [
        "spec_centroid_hz",
        "spec_rolloff_hz",
        "spec_centroid_z",
        "spec_rolloff_z",
        "mfcc_00",
        "mfcc_z_00",
    ]
    feats = compute_audio_features_for_windows(audio, [0.0], [1.0], cfg, required_columns=cols)

    assert float(feats["spec_centroid_hz"][0]) == 0.0
    assert float(feats["spec_rolloff_hz"][0]) == 0.0
    assert float(feats["spec_centroid_z"][0]) == 0.0
    assert float(feats["spec_rolloff_z"][0]) == 0.0
    assert float(feats["mfcc_00"][0]) == 0.0
    assert float(feats["mfcc_z_00"][0]) == 0.0


def test_audio_features_high_freq_has_higher_centroid_and_ultra_frac():
    sr = 16000
    t = np.arange(sr, dtype=np.float32) / float(sr)

    # Two windows in one "video": 1kHz then ~7.5kHz.
    x_lo = 0.5 * np.sin(2.0 * np.pi * 1000.0 * t)
    x_hi = 0.5 * np.sin(2.0 * np.pi * 7500.0 * t)
    audio = np.concatenate([x_lo, x_hi]).astype(np.float32, copy=False)

    cfg = AudioFeatureConfig(
        enabled=True,
        sr_hz=sr,
        spectral_enabled=True,
        mfcc_enabled=False,
    )

    cols = [
        "spec_centroid_hz",
        "spec_rolloff_hz",
        "phone_band_frac",
        "ultra_hf_frac",
    ]
    feats = compute_audio_features_for_windows(audio, [0.0, 1.0], [1.0, 2.0], cfg, required_columns=cols)

    c0 = float(feats["spec_centroid_hz"][0])
    c1 = float(feats["spec_centroid_hz"][1])
    r0 = float(feats["spec_rolloff_hz"][0])
    r1 = float(feats["spec_rolloff_hz"][1])

    assert c1 > c0
    assert r1 > r0

    phone0 = float(feats["phone_band_frac"][0])
    phone1 = float(feats["phone_band_frac"][1])
    ultra0 = float(feats["ultra_hf_frac"][0])
    ultra1 = float(feats["ultra_hf_frac"][1])

    assert phone0 > phone1
    assert ultra1 > ultra0

