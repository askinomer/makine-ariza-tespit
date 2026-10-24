"""Ses → log mel spektrogram → 64×64 pencereler.

Eğitim, kalibrasyon, test, gürültü deneyi ve demo AYNI fonksiyonları çağırır.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .data import resolve_path


class AudioTooShortError(ValueError):
    pass


def load_audio(path_or_file, sr: int) -> np.ndarray:
    """Mono float32 ses okur; gerekirse hedef örnekleme hızına yeniden örnekler.

    Genlik normalizasyonu YAPILMAZ: ses seviyesi farkı arızayla ilişkili olabilir.
    """
    y, file_sr = sf.read(path_or_file, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y.astype(np.float32)


def logmel_from_waveform(y: np.ndarray, fcfg: dict) -> np.ndarray:
    """(n_mels, T) float32 log mel spektrogram."""
    S = librosa.feature.melspectrogram(
        y=y, sr=fcfg["sr"], n_fft=fcfg["n_fft"], win_length=fcfg["win_length"],
        hop_length=fcfg["hop_length"], window=fcfg["window"], n_mels=fcfg["n_mels"],
        power=fcfg["power"], fmin=fcfg["fmin"], fmax=fcfg["fmax"],
    )
    return librosa.power_to_db(S, ref=fcfg["ref"], top_db=fcfg["top_db"]).astype(np.float32)


def window_starts(n_frames: int, frames: int, hop: int) -> list[int]:
    """Pencere başlangıçları. Son pencere sessizlikle doldurulmaz; son `frames` adım alınır
    ve aynı başlangıç iki kez sayılmaz."""
    if n_frames < frames:
        raise AudioTooShortError(
            f"Kayıt {n_frames} zaman adımı; en az {frames} adım (~{frames * 0.032:.1f} s) gerekli."
        )
    starts = list(range(0, n_frames - frames + 1, hop))
    last = n_frames - frames
    if starts[-1] != last:
        starts.append(last)
    return starts


def make_windows(logmel: np.ndarray, frames: int, hop: int) -> tuple[np.ndarray, list[int]]:
    """(n_mels, T) → (W, n_mels, frames) pencereler ve başlangıç indeksleri."""
    starts = window_starts(logmel.shape[1], frames, hop)
    return np.stack([logmel[:, s:s + frames] for s in starts]), starts


def feature_fingerprint(fcfg: dict) -> str:
    keys = ["sr", "n_fft", "win_length", "hop_length", "window", "n_mels", "power", "fmin", "fmax",
            "top_db", "ref"]
    blob = json.dumps({k: fcfg[k] for k in keys}, sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()[:8]


def cache_path(cfg: dict, machine_type: str, source_recording_id: str) -> Path:
    fp = feature_fingerprint(cfg["features"])
    return Path(cfg["paths"]["processed_dir"]) / f"logmel_{fp}" / machine_type / f"{source_recording_id}.npy"


def compute_and_cache(cfg: dict, row) -> Path:
    out = cache_path(cfg, row["machine_type"], row["source_recording_id"])
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        y = load_audio(resolve_path(row["path"]), cfg["features"]["sr"])
        np.save(out, logmel_from_waveform(y, cfg["features"]))
    return out


def load_logmels(cfg: dict, df) -> list[np.ndarray]:
    """Önbellekteki log mel matrislerini okur (yoksa hesaplar)."""
    return [np.load(compute_and_cache(cfg, row)) for _, row in df.iterrows()]


class BandStandardizer:
    """Mel bandı başına ortalama/standart sapma. YALNIZCA eğitim kayıtlarından öğrenilir."""

    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.mean = mean
        self.std = std

    def fit(self, logmels: list[np.ndarray], eps: float = 1e-6) -> "BandStandardizer":
        allf = np.concatenate(logmels, axis=1)  # (n_mels, toplam T)
        self.mean = allf.mean(axis=1).astype(np.float32)
        self.std = (allf.std(axis=1) + eps).astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        # x: (..., n_mels, T) — mel ekseni sondan ikinci
        return ((x - self.mean[:, None]) / self.std[:, None]).astype(np.float32)

    def save(self, path) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path) -> "BandStandardizer":
        d = np.load(path)
        return cls(d["mean"], d["std"])


def baseline_vector(logmel: np.ndarray) -> np.ndarray:
    """Referans yöntem için 2×n_mels boyutlu vektör: zaman boyunca ortalama ve std."""
    return np.concatenate([logmel.mean(axis=1), logmel.std(axis=1)]).astype(np.float32)
