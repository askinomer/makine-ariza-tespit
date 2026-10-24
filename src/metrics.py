"""Değerlendirme ölçütleri: ROC AUC, pAUC, precision/recall/F1, yanlış alarm oranı, bootstrap GA."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def threshold_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(scores) > threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (tp and precision + recall) else 0.0
    far = fp / (fp + tn) if fp + tn else float("nan")
    return {"precision": precision, "recall": recall, "f1": f1, "far": far,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def ranking_metrics(y: np.ndarray, scores: np.ndarray, max_fpr: float = 0.1) -> dict:
    """AUC sürekli skorlardan hesaplanır. Tek sınıf varsa NaN döner."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return {"auc": float("nan"), "pauc": float("nan")}
    return {"auc": float(roc_auc_score(y, scores)),
            "pauc": float(roc_auc_score(y, scores, max_fpr=max_fpr))}  # sklearn standartlaştırılmış pAUC


def all_metrics(y, scores, threshold, max_fpr=0.1) -> dict:
    out = ranking_metrics(y, scores, max_fpr)
    out.update(threshold_metrics(y, scores, threshold))
    out["n_normal"] = int((np.asarray(y) == 0).sum())
    out["n_anomaly"] = int((np.asarray(y) == 1).sum())
    return out


def bootstrap_ci(y, scores, threshold, n_boot: int = 1000, max_fpr: float = 0.1,
                 seed: int = 0, alpha: float = 0.05) -> dict:
    """Sınıf içi (tabakalı) bootstrap ile %95 güven aralıkları.

    Kimlik başına yalnızca ~100 normal test kaydı olduğu için yanlış alarm oranı
    gibi oranlar tek sayı olarak fazla kesin görünür; aralık bunu gösterir.
    """
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores)
    idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0 or n_boot <= 0:
        return {}
    rng = np.random.default_rng(seed)
    keys = ("auc", "pauc", "recall", "far", "f1")
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        b = np.concatenate([rng.choice(idx0, len(idx0)), rng.choice(idx1, len(idx1))])
        m = all_metrics(y[b], scores[b], threshold, max_fpr)
        for k in keys:
            samples[k].append(m[k])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {f"{k}_ci": (float(np.nanpercentile(v, lo)), float(np.nanpercentile(v, hi)))
            for k, v in samples.items()}
