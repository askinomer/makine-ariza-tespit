"""Adım 10: karar eşiğini YALNIZCA normal kalibrasyon kayıtlarından belirler.

Eşik = kalibrasyon skorlarının 95. persentili (config: scoring.threshold_percentile).
Bu, kalibrasyon verisinde ~%5 yanlış alarm hedefler; testte aynı oran garanti değildir.
Hiçbir arızalı kayıt ve hiçbir test kaydı kullanılmaz.
"""
import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_split, machine_keys, subset
from src.detectors import METHODS, detector_dir, load_detector
from src.features import load_logmels
from src.utils import append_run_log, ensure_dir, feature_settings_str, load_config, make_run_id, save_json


def calibrate_one(cfg, split_df, method, mt, mid, seed=None):
    d = detector_dir(cfg, method, mt, mid, seed)
    if not (d / "meta.json").exists():
        print(f"  atlandı (model yok): {d}")
        return None
    det = load_detector(method, d)
    cal = subset(split_df, mt, mid, "cal")
    assert (cal["label"] == "normal").all(), "Kalibrasyonda anomali olmamalı"
    scores = det.score_logmels(load_logmels(cfg, cal))
    p = cfg["scoring"]["threshold_percentile"]
    thr = float(np.percentile(scores, p))
    info = {"threshold": thr, "percentile": p, "n_cal": len(cal),
            "cal_far": float((scores > thr).mean()),
            "cal_score_mean": float(scores.mean()), "cal_score_std": float(scores.std()),
            "source": "normal kalibrasyon kayıtları (split=cal)"}
    save_json(info, d / "threshold.json")
    out = ensure_dir(Path(cfg["paths"]["results"]) / "scores")
    tag = method if method == "iforest" else f"{method}_seed{det.meta['seed']}"
    pd.DataFrame({"path": cal["path"], "score": scores}).to_csv(out / f"cal_{tag}_{mt}_{mid}.csv", index=False)
    append_run_log(cfg, {
        "run_id": make_run_id(f"calib_{method}", mt, mid, det.meta.get("seed", "")), "method": method,
        "machine_type": mt, "machine_id": mid, "seed": det.meta.get("seed", ""),
        "feature_settings": feature_settings_str(cfg), "threshold": f"{thr:.6f}",
        "notes": f"p{p} of {len(cal)} normal cal records",
    })
    print(f"  [{tag}] {mt} {mid}: eşik={thr:.5f} (n_cal={len(cal)}, kalibrasyonda yanlış alarm {info['cal_far']:.1%})")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="cnn_ae tohumları (varsayılan: config seed)")
    ap.add_argument("--machine-ids", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seeds = args.seeds or [cfg["seed"]]
    split_df = load_split(cfg)
    print("Eşikler normal kalibrasyon kayıtlarından hesaplanıyor...")
    for mt, mid in machine_keys(split_df):
        if args.machine_ids and mid not in args.machine_ids:
            continue
        for method in args.methods:
            for s in (seeds if method == "cnn_ae" else [None]):
                calibrate_one(cfg, split_df, method, mt, mid, s)


if __name__ == "__main__":
    main()
