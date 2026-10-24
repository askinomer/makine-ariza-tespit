"""Özellik çıkarma ve pencereleme."""
import numpy as np
import pytest

from src.features import (AudioTooShortError, BandStandardizer, baseline_vector, logmel_from_waveform,
                          make_windows, window_starts)
from src.utils import load_config


def test_window_starts_10s():
    # 10 s @16 kHz, hop 512, center=True → 313 çerçeve
    s = window_starts(313, 64, 32)
    assert s[:3] == [0, 32, 64]
    assert s[-1] == 313 - 64           # son pencere sona yaslanır, doldurma yok
    assert len(s) == len(set(s)) == 9


def test_window_starts_exact_and_short():
    assert window_starts(64, 64, 32) == [0]
    assert window_starts(96, 64, 32) == [0, 32]   # tam oturuyor, tekrar yok
    with pytest.raises(AudioTooShortError):
        window_starts(63, 64, 32)


def test_logmel_shape_and_range():
    f = load_config()["features"]
    y = np.random.default_rng(0).standard_normal(160000).astype(np.float32) * 0.1
    m = logmel_from_waveform(y, f)
    assert m.shape == (64, 313) and m.dtype == np.float32
    assert m.max() - m.min() <= f["top_db"] + 1e-3
    w, starts = make_windows(m, 64, 32)
    assert w.shape == (9, 64, 64)
    np.testing.assert_array_equal(w[-1], m[:, -64:])


def test_no_amplitude_normalization():
    """Ses seviyesi farkı korunmalı: 2× genlik ≈ +6 dB."""
    f = load_config()["features"]
    y = np.random.default_rng(1).standard_normal(32000).astype(np.float32) * 0.05
    d = logmel_from_waveform(2 * y, f).mean() - logmel_from_waveform(y, f).mean()
    assert 5.5 < d < 6.5


def test_band_standardizer_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    train = [rng.normal(5, 2, (64, 100)).astype(np.float32) for _ in range(5)]
    sc = BandStandardizer().fit(train)
    z = sc.transform(np.concatenate(train, axis=1))
    assert abs(z.mean()) < 1e-4 and abs(z.std() - 1) < 1e-3
    sc.save(tmp_path / "s.npz")
    sc2 = BandStandardizer.load(tmp_path / "s.npz")
    np.testing.assert_allclose(sc2.transform(train[0]), sc.transform(train[0]))
    # batch boyutlu girdi
    assert sc.transform(np.stack(train)[:, :, :64]).shape == (5, 64, 64)


def test_baseline_vector():
    m = np.random.default_rng(0).normal(size=(64, 313)).astype(np.float32)
    v = baseline_vector(m)
    assert v.shape == (128,)
    np.testing.assert_allclose(v[:64], m.mean(axis=1), rtol=1e-5)
