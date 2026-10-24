"""Gürültü dayanıklılığı deneyi için sentetik beyaz gürültü ekleme."""
from __future__ import annotations

import zlib

import numpy as np


def add_white_noise(y: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Kayıt RMS'ine göre ölçeklenmiş beyaz gürültü ekler.

    Gürültü RMS = sinyal RMS × 10^(−SNR/20)  →  20 dB: 0,1×, 10 dB: ≈0,316×
    """
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    noise = rng.standard_normal(len(y))
    noise *= rms * 10 ** (-snr_db / 20) / (np.sqrt(np.mean(noise ** 2)) + 1e-12)
    return (y + noise).astype(np.float32)


def file_rng(base_seed: int, snr_db: float, key: str) -> np.random.Generator:
    """Her (dosya, SNR) için sabit gürültü: aynı kayıt iki yöntemde ve her çalıştırmada aynı gürültüyü alır."""
    return np.random.default_rng([base_seed, int(snr_db * 10), zlib.crc32(key.encode())])
