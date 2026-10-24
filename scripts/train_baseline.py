"""Adım 7: Isolation Forest referans yöntemi (makine kimliği başına ayrı model).

Özellik: her kaydın log mel matrisinde zaman boyunca ortalama + standart sapma (2×64 = 128 boyut).
Ölçekleyici ve model YALNIZCA normal eğitim kayıtlarıyla öğrenilir.
"""
import _bootstrap  # noqa: F401

import argparse
import time

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_split, machine_keys, subset
from src.detectors import detector_dir
from src.features import baseline_vector, load_logmels
from src.utils import (append_run_log, ensure_dir, feature_settings_str, load_config, make_run_id,
                       save_json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--machine-ids", nargs="*", default=None, help="örn. id_00 (boş = hepsi)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    bcfg = cfg["baseline"]
    split_df = load_split(cfg)

    for mt, mid in machine_keys(split_df):
        if args.machine_ids and mid not in args.machine_ids:
            continue
        t0 = time.time()
        train = subset(split_df, mt, mid, "train")
        X = np.stack([baseline_vector(m) for m in load_logmels(cfg, train)])
        pipe = make_pipeline(
            StandardScaler(),
            IsolationForest(n_estimators=bcfg["n_estimators"], random_state=bcfg["random_state"], n_jobs=-1),
        )
        pipe.fit(X)
        out = ensure_dir(detector_dir(cfg, "iforest", mt, mid))
        joblib.dump(pipe, out / "model.joblib")
        val = subset(split_df, mt, mid, "val")
        val_scores = -pipe.score_samples(np.stack([baseline_vector(m) for m in load_logmels(cfg, val)]))
        meta = {"method": "iforest", "machine_type": mt, "machine_id": mid, "seed": bcfg["random_state"],
                "features": cfg["features"], "n_train": len(train), "feature_dim": int(X.shape[1]),
                "val_score_mean": float(val_scores.mean())}
        save_json(meta, out / "meta.json")
        dt = time.time() - t0
        append_run_log(cfg, {
            "run_id": make_run_id("iforest", mt, mid, bcfg["random_state"]), "method": "iforest",
            "machine_type": mt, "machine_id": mid, "seed": bcfg["random_state"],
            "feature_settings": feature_settings_str(cfg),
            "model": f"IsolationForest(n_estimators={bcfg['n_estimators']}) on mean+std log mel",
            "train_time_s": f"{dt:.1f}", "val_loss": f"{val_scores.mean():.4f}",
            "notes": "val_loss = normal doğrulama skor ortalaması",
        })
        print(f"[iforest] {mt} {mid}: {len(train)} eğitim kaydı, {X.shape[1]} boyut, "
              f"doğrulama skor ort. {val_scores.mean():.4f} ({dt:.1f} s) → {out}")


if __name__ == "__main__":
    main()
