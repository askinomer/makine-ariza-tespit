"""Guilin: yeni çalışma koşulunda kısa normal kayıtla eşik ayarı (hedef koşul kalibrasyonu).

    python tools/guilin_target_calibration.py --wav /content/guilin_wav --results /content/guilin_results

Koşul-dışı bırakma deneyinin kaydedilmiş skorlarını kullanır (yeni model eğitilmez). Kural, sonuçlara bakmadan sabitlendi:
  eşik  = dışarıda bırakılan koşulun 1. normal dosyasının (ilk 10 dk, 60 klip) skorlarının 95. persentili
  test  = o koşulun 2-5. normal dosyaları (240 klip) ve tüm arıza klipleri
Gürültülü koşullarda eşik aynı gürültü düzeyindeki 1. dosyadan alınır (sahada kalibrasyon kaydı da aynı ortamda yapılır).
Çıktı: target_calibration.csv (kat düzeyi) ve target_calibration.md (özet).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from guilin_experiment import FAULT_TR, FAULTS, METHOD_TR, NOISES, PERCENTILE, auc, fmt, load_table  # noqa: E402
from src.metrics import threshold_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    res = Path(a.results)
    tab = load_table(Path(a.wav))
    S = pd.concat([pd.read_parquet(p) for p in sorted((res / "scores").glob("*.parquet"))], ignore_index=True)
    S = S[S.role == "test"].merge(tab[["label", "part"]], left_on="idx", right_index=True)
    rows = []
    for (m, seed, fold, nz), t in S.groupby(["method", "seed", "fold", "noise"]):
        thr = np.percentile(t[(t.label == "normal") & (t.part == 1)].score, PERCENTILE)
        nrm = t[(t.label == "normal") & (t.part >= 2)].score.values
        for fl in FAULTS + ["tümü"]:
            fs = t[t.label != "normal"].score.values if fl == "tümü" else t[t.label == fl].score.values
            y = np.r_[np.zeros(len(nrm)), np.ones(len(fs))]
            sc = np.r_[nrm, fs]
            tm = threshold_metrics(y, sc, thr)
            rows.append(dict(method=m, seed=seed, fold=fold, noise=nz, fault=fl, auc=auc(y, sc), recall=tm["recall"],
                             far=tm["far"], precision=tm["precision"], f1=tm["f1"], n_normal=len(nrm), n_fault=len(fs)))
    R = pd.DataFrame(rows)
    R.to_csv(res / "target_calibration.csv", index=False)
    per_seed = R.groupby(["method", "seed", "noise", "fault"])[["auc", "recall", "far", "f1"]].mean()
    agg = per_seed.groupby(["method", "noise", "fault"]).agg(["mean", "std"])
    worst = R.groupby(["method", "seed", "noise", "fault"]).far.max().groupby(["method", "noise", "fault"]).mean()

    L = ["# Hedef koşul kalibrasyonu", "",
         "Eşik, dışarıda bırakılan (yeni) koşulun ilk 10 dakikalık normal kaydından (60 klip, 95. persentil) alındı; "
         "test: aynı koşulun kalan 40 dakikalık normal kaydı (240 klip) ve tüm arıza klipleri. Yeni model eğitilmedi.", "",
         "## Tüm arızalar", "",
         "| Yöntem | Koşul | Yakalama | Yanlış alarm | En kötü kat yanlış alarm | F1 |", "|---|---|---|---|---|---|"]
    for m in ("iforest", "cnn_ae"):
        for nz in NOISES:
            r = agg.loc[(m, nz, "tümü")]
            pm = lambda k: fmt(r[(k, "mean")]) + ("" if m == "iforest" else f" ± {fmt(r[(k, 'std')])}")
            L.append(f"| {METHOD_TR[m]} | {'temiz' if nz == 'temiz' else '+' + nz + ' dB'} | {pm('recall')} | "
                     f"{pm('far')} | {fmt(worst.loc[(m, nz, 'tümü')])} | {pm('f1')} |")
    L += ["", "## Temiz — arıza türü başına yakalama", "", "| Yöntem | " + " | ".join(FAULT_TR[f] for f in FAULTS) + " |",
          "|---|" + "---|" * len(FAULTS)]
    for m in ("iforest", "cnn_ae"):
        L.append(f"| {METHOD_TR[m]} | " + " | ".join(fmt(agg.loc[(m, 'temiz', f)][('recall', 'mean')]) for f in FAULTS) + " |")
    L.append("")
    (res / "target_calibration.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
