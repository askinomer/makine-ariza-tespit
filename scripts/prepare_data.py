"""Adım 3–5: veri envanteri, kopya kontrolü ve sızıntısız kayıt düzeyinde bölünme.

Çıktılar:
    data/manifest.csv          her dosya için yol, kimlik, etiket, süre, RMS, içerik özeti, bölüm
    data/split.csv             sabit bölünme (proje sonuna kadar değiştirmeyin)
    results/data_counts.csv    kimlik × bölüm × etiket sayıları
    results/figures/samples/   5 normal eğitim örneğinin dalga biçimi ve log mel görseli

split.csv zaten varsa üzerine yazmaz (test listesi sabit kalsın diye); --force ile zorlanır.
"""
import _bootstrap  # noqa: F401

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import assign_splits, build_manifest, check_no_leakage, resolve_path
from src.features import load_audio, logmel_from_waveform
from src.utils import ensure_dir, load_config


def sample_figures(cfg, split_df, n=5):
    out = ensure_dir(Path(cfg["paths"]["results"]) / "figures" / "samples")
    fcfg = cfg["features"]
    for (mt, mid), sub in split_df[(split_df["split"] == "train")].groupby(["machine_type", "machine_id"]):
        rows = sub.sample(min(n, len(sub)), random_state=cfg["seed"])
        fig, axes = plt.subplots(len(rows), 2, figsize=(12, 2.4 * len(rows)), squeeze=False)
        for ax_row, (_, r) in zip(axes, rows.iterrows()):
            y = load_audio(resolve_path(r["path"]), fcfg["sr"])
            t = np.arange(len(y)) / fcfg["sr"]
            ax_row[0].plot(t, y, lw=0.4)
            ax_row[0].set_title(Path(r["path"]).name, fontsize=9)
            ax_row[0].set_xlabel("s")
            m = logmel_from_waveform(y, fcfg)
            im = ax_row[1].imshow(m, origin="lower", aspect="auto",
                                  extent=[0, m.shape[1] * fcfg["hop_length"] / fcfg["sr"], 0, fcfg["n_mels"]])
            ax_row[1].set_ylabel("mel bandı")
            fig.colorbar(im, ax=ax_row[1], label="dB")
        fig.suptitle(f"{mt} {mid} — normal eğitim örnekleri")
        fig.tight_layout()
        fig.savefig(out / f"{mt}_{mid}_normal_samples.png", dpi=110)
        plt.close(fig)
    print(f"Örnek görseller: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true", help="Var olan split.csv dosyasının üzerine yaz")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)

    split_path = Path(cfg["paths"]["split"])
    if split_path.exists() and not args.force:
        print(f"{split_path} zaten var; test listesi sabit kalsın diye yeniden oluşturulmadı. "
              "Bilerek yenilemek için --force kullanın.")
        split_df = pd.read_csv(split_path)
    else:
        print("Veri envanteri çıkarılıyor...")
        manifest = build_manifest(cfg)
        split_df = assign_splits(manifest, cfg)
        ensure_dir(split_path.parent)
        split_df.to_csv(cfg["paths"]["manifest"], index=False)
        split_df[["path", "machine_type", "machine_id", "label", "label_int", "official_split",
                  "source_recording_id", "group_id", "split"]].to_csv(split_path, index=False)
        print(f"Yazıldı: {cfg['paths']['manifest']} ve {split_path}")

    # --- Kontroller ---
    problems = check_no_leakage(split_df)
    if problems:
        print("SIZINTI KONTROLÜ BAŞARISIZ:\n  - " + "\n  - ".join(problems))
        sys.exit(1)
    print("Sızıntı kontrolü: geçti (kayıtlar ve içerik kopyaları bölümler arasında kesişmiyor).")

    if "issue" in split_df.columns:
        issues = split_df[split_df["issue"].fillna("") != ""]
        if len(issues):
            print(f"Not: {len(issues)} dosyada uyarı var (ilk 10):")
            print(issues[["path", "issue"]].head(10).to_string(index=False))
    if "content_md5" in split_df.columns:
        dups = split_df[split_df.duplicated("content_md5", keep=False) & (split_df["content_md5"] != "")]
        print(f"Aynı içerikli kopya dosya sayısı: {len(dups)}")
    excl = split_df[split_df["split"].str.startswith("excluded")]
    if len(excl):
        print(f"Dışlanan dosya: {len(excl)} ({excl['split'].value_counts().to_dict()})")

    # --- Sayı tablosu ---
    counts = (split_df[~split_df["split"].str.startswith("excluded")]
              .pivot_table(index=["machine_type", "machine_id"], columns=["split", "label"],
                           values="path", aggfunc="count", fill_value=0))
    counts.columns = [f"{s}_{l}" for s, l in counts.columns]
    order = [c for c in ["train_normal", "val_normal", "cal_normal", "test_normal", "test_anomaly"]
             if c in counts.columns]
    counts = counts[order]
    out = ensure_dir(cfg["paths"]["results"]) / "data_counts.csv"
    counts.to_csv(out)
    print("\nKimlik başına kayıt sayıları:")
    print(counts.to_string())
    for (mt, mid), r in counts.iterrows():
        if r.get("test_normal", 0) == 0 or r.get("test_anomaly", 0) == 0:
            print(f"UYARI: {mt} {mid} testinde tek sınıf var; AUC hesaplanmayacak.")

    if not args.no_figures:
        sample_figures(cfg, split_df)


if __name__ == "__main__":
    main()
