"""Adım 12: gürültü dayanıklılığı. Modeller ve eşikler DEĞİŞTİRİLMEDEN üç koşul karşılaştırılır:
özgün test, +20 dB SNR beyaz gürültü, +10 dB SNR beyaz gürültü.

Gürültü sabit tohumla ve dosya başına deterministik üretilir; iki yöntem aynı gürültülü kaydı görür.
Not: özgün kayıtlar zaten fabrika ortam gürültüsü içerir; bu deney EK sentetik gürültüye
duyarlılığı ölçer, gerçek fabrika gürültüsüne genelleme kanıtı değildir.

--recalibrate ile ek bir koşul raporlanır: eşik, aynı SNR'de gürültü eklenmiş NORMAL kalibrasyon
kayıtlarından yeniden hesaplanır (model aynı). Bu, düşüşün ne kadarının sıralama gücünden, ne kadarının
eşik kaymasından geldiğini ayırmaya yarar.
"""
import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import load_split, machine_keys, resolve_path, subset
from src.detectors import METHOD_LABELS, METHODS, detector_dir, load_detector
from src.features import feature_fingerprint, load_audio, load_logmels
from src.metrics import all_metrics
from src.noise import add_white_noise, file_rng
from src.utils import ensure_dir, load_config

_CACHE: dict = {}


def noisy_logmels(det, df, snr, base_seed):
    fp = feature_fingerprint(det.fcfg)
    out = []
    for _, r in df.iterrows():
        key = (fp, snr, r["path"])
        if key not in _CACHE:
            y = load_audio(resolve_path(r["path"]), det.fcfg["sr"])
            yn = add_white_noise(y, snr, file_rng(base_seed, snr, r["source_recording_id"]))
            _CACHE[key] = det.logmel_from_waveform(yn)
        out.append(_CACHE[key])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--machine-ids", nargs="*", default=None)
    ap.add_argument("--recalibrate", action="store_true", help="Ek koşul: gürültülü normal kalibrasyonla yeni eşik")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seeds = args.seeds or [cfg["seed"]]
    ncfg, sc = cfg["noise"], cfg["scoring"]
    res = ensure_dir(cfg["paths"]["results"])
    split_df = load_split(cfg)

    rows = []
    for mt, mid in machine_keys(split_df):
        if args.machine_ids and mid not in args.machine_ids:
            continue
        test = subset(split_df, mt, mid, "test")
        cal = subset(split_df, mt, mid, "cal")
        y = test["label_int"].to_numpy()
        for method in args.methods:
            for s in (seeds if method == "cnn_ae" else [None]):
                d = detector_dir(cfg, method, mt, mid, s)
                if not (d / "threshold.json").exists():
                    print(f"  atlandı: {d}")
                    continue
                det = load_detector(method, d)
                conds = [("özgün", None)] + [(f"+{snr} dB", snr) for snr in ncfg["snr_db"]]
                for name, snr in conds:
                    lms = load_logmels(cfg, test) if snr is None else noisy_logmels(det, test, snr, ncfg["seed"])
                    scores = det.score_logmels(lms)
                    m = all_metrics(y, scores, det.threshold, sc["pauc_max_fpr"])
                    rows.append({"method": method, "seed": det.meta["seed"], "machine_type": mt, "machine_id": mid,
                                 "condition": name, "snr_db": snr if snr is not None else np.inf,
                                 "threshold_source": "sabit (özgün kalibrasyon)", "threshold": det.threshold,
                                 "mean_normal_score": float(scores[y == 0].mean()), **m})
                    if args.recalibrate and snr is not None:
                        cal_scores = det.score_logmels(noisy_logmels(det, cal, snr, ncfg["seed"]))
                        thr2 = float(np.percentile(cal_scores, sc["threshold_percentile"]))
                        m2 = all_metrics(y, scores, thr2, sc["pauc_max_fpr"])
                        rows.append({"method": method, "seed": det.meta["seed"], "machine_type": mt,
                                     "machine_id": mid, "condition": name, "snr_db": snr,
                                     "threshold_source": "yeniden kalibre (gürültülü normal cal)",
                                     "threshold": thr2, "mean_normal_score": float(scores[y == 0].mean()), **m2})
                    print(f"  [{method} s{det.meta['seed']}] {mt} {mid} {name:>7}: AUC={m['auc']:.3f} recall={m['recall']:.3f} "
                          f"FAR={m['far']:.3f}")
    if not rows:
        print("Değerlendirilecek model yok.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(res / "noise_results.csv", index=False)

    keys = ["method", "threshold_source", "condition", "snr_db"]
    macro = (df.groupby(keys + ["seed"])[["auc", "pauc", "recall", "f1", "far"]].mean()
             .groupby(keys).mean().reset_index().sort_values(["method", "threshold_source", "snr_db"],
                                                             ascending=[True, True, False]))
    clean = macro[macro["condition"] == "özgün"].set_index("method")
    for k in ("auc", "recall", "far"):
        macro[f"Δ{k}"] = macro.apply(lambda r: r[k] - clean.loc[r["method"], k], axis=1)
    macro.to_csv(res / "noise_summary.csv", index=False)

    lines = ["# Gürültü dayanıklılığı (kimliklerin eşit ağırlıklı ortalaması)\n",
             "Δ = özgün teste göre fark. Sabit eşik satırlarında model ve eşik değiştirilmedi.\n",
             "| Yöntem | Eşik | Koşul | AUC | ΔAUC | Recall | ΔRecall | Yanlış alarm | ΔYanlış alarm |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in macro.iterrows():
        lines.append(f"| {METHOD_LABELS[r['method']]} | {r['threshold_source']} | {r['condition']} | "
                     f"{r['auc']:.3f} | {r['Δauc']:+.3f} | {r['recall']:.3f} | {r['Δrecall']:+.3f} | "
                     f"{r['far']:.3f} | {r['Δfar']:+.3f} |")
    (res / "noise_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fixed = macro[macro["threshold_source"].str.startswith("sabit")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, k, lab in zip(axes, ["auc", "recall", "far"], ["ROC AUC", "Recall (sabit eşik)", "Yanlış alarm oranı"]):
        for method, g in fixed.groupby("method"):
            ax.plot(g["condition"], g[k], marker="o", label=METHOD_LABELS[method])
        ax.set_title(lab, fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Ek beyaz gürültü altında performans (model ve eşik sabit)", fontsize=10)
    fig.tight_layout()
    fig.savefig(ensure_dir(res / "figures") / "noise_robustness.png", dpi=110)
    plt.close(fig)
    print(f"\nÖzet: {res / 'noise_summary.md'}")
    print(macro.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
