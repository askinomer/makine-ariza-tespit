"""Eğitilmiş modelleri yükleyip skor üreten ortak arayüz.

Kalibrasyon, değerlendirme, gürültü deneyi ve Streamlit demosu bu sınıfları kullanır;
böylece ön işleme ve skor kuralı her yerde aynıdır. Her model klasörü kendi
ön işleme ayarlarını (meta.json) ve eşiğini (threshold.json) taşır.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .features import BandStandardizer, baseline_vector, load_audio, logmel_from_waveform, make_windows
from .utils import load_json

METHODS = ("iforest", "cnn_ae")
METHOD_LABELS = {"iforest": "Isolation Forest (referans)", "cnn_ae": "CNN autoencoder"}


def detector_dir(cfg: dict, method: str, machine_type: str, machine_id: str, seed: int | None = None) -> Path:
    root = Path(cfg["paths"]["checkpoints"])
    if method == "iforest":
        return root / "iforest" / machine_type / machine_id
    if method == "cnn_ae":
        seed = cfg["seed"] if seed is None else seed
        return root / "cnn_ae" / f"seed{seed}" / machine_type / machine_id
    raise ValueError(f"Bilinmeyen yöntem: {method}")


def aggregate(window_scores: np.ndarray, how: str = "mean") -> float:
    if how == "mean":
        return float(np.mean(window_scores))
    if how == "max":
        return float(np.max(window_scores))
    if how.startswith("top"):  # örn. top3: en yüksek 3 pencerenin ortalaması
        k = int(how[3:])
        return float(np.mean(np.sort(window_scores)[-k:]))
    raise ValueError(f"Bilinmeyen toplama kuralı: {how}")


class BaseDetector:
    method: str = ""

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.meta = load_json(self.dir / "meta.json")
        self.fcfg = self.meta["features"]
        thr_path = self.dir / "threshold.json"
        self.threshold = load_json(thr_path)["threshold"] if thr_path.exists() else None

    # --- ses → log mel (demo ve gürültü deneyi için) ---
    def logmel_from_file(self, path_or_file) -> np.ndarray:
        return logmel_from_waveform(load_audio(path_or_file, self.fcfg["sr"]), self.fcfg)

    def logmel_from_waveform(self, y: np.ndarray) -> np.ndarray:
        return logmel_from_waveform(y, self.fcfg)

    def score_logmels(self, logmels: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def decide(self, score: float) -> bool:
        if self.threshold is None:
            raise RuntimeError("Eşik yok; önce scripts/calibrate.py çalıştırın.")
        return bool(score > self.threshold)


class IForestDetector(BaseDetector):
    method = "iforest"

    def __init__(self, directory: Path):
        super().__init__(directory)
        self.pipeline = joblib.load(self.dir / "model.joblib")  # StandardScaler + IsolationForest

    def score_logmels(self, logmels: list[np.ndarray]) -> np.ndarray:
        X = np.stack([baseline_vector(m) for m in logmels])
        return -self.pipeline.score_samples(X)  # yüksek = daha anormal


class AEDetector(BaseDetector):
    method = "cnn_ae"

    def __init__(self, directory: Path, device: str = "cpu"):
        super().__init__(directory)
        import torch

        from .models import ConvAutoencoder

        self.torch = torch
        self.device = torch.device(device)
        self.scaler = BandStandardizer.load(self.dir / "scaler.npz")
        self.model = ConvAutoencoder(self.fcfg["n_mels"], self.fcfg["frames"], self.meta["bottleneck"])
        state = torch.load(self.dir / "best.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.aggregation = self.meta.get("aggregation", "mean")

    def window_errors(self, logmel: np.ndarray) -> tuple[np.ndarray, list[int]]:
        """Kaydın her penceresi için yeniden oluşturma MSE'si ve pencere başlangıçları."""
        win, starts = make_windows(self.scaler.transform(logmel), self.fcfg["frames"], self.fcfg["window_hop"])
        return self._errors(win), starts

    def _errors(self, windows: np.ndarray, batch: int = 256) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(windows), batch):
                x = self.torch.from_numpy(windows[i:i + batch][:, None]).to(self.device)
                err = ((self.model(x) - x) ** 2).mean(dim=(1, 2, 3))
                out.append(err.cpu().numpy())
        return np.concatenate(out)

    def score_logmels(self, logmels: list[np.ndarray], how: str | None = None) -> np.ndarray:
        how = how or self.aggregation
        all_w, counts = [], []
        for m in logmels:
            w, _ = make_windows(self.scaler.transform(m), self.fcfg["frames"], self.fcfg["window_hop"])
            all_w.append(w)
            counts.append(len(w))
        errs = self._errors(np.concatenate(all_w))
        scores, k = [], 0
        for c in counts:
            scores.append(aggregate(errs[k:k + c], how))
            k += c
        return np.asarray(scores)


def load_detector(method: str, directory: Path, device: str = "cpu") -> BaseDetector:
    if method == "iforest":
        return IForestDetector(directory)
    if method == "cnn_ae":
        return AEDetector(directory, device=device)
    raise ValueError(method)


def available_detectors(cfg: dict) -> list[dict]:
    """Eşiği kalibre edilmiş tüm modeller (demo seçim listesi için)."""
    root = Path(cfg["paths"]["checkpoints"])
    found = []
    for thr in sorted(root.rglob("threshold.json")):
        d = thr.parent
        if not (d / "meta.json").exists():
            continue
        meta = load_json(d / "meta.json")
        found.append({"method": meta["method"], "machine_type": meta["machine_type"],
                      "machine_id": meta["machine_id"], "seed": meta.get("seed"), "dir": d})
    return found
