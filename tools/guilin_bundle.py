"""Colab'daki Guilin sonuçlarını repo düzeninde tek zip'e toplar (repo köküne açılır).

    python tools/guilin_bundle.py --results /content/guilin_results --ckpt /content/guilin_ckpt \
        --wav /content/guilin_wav --out /content/guilin_sonuclar.zip

İçerik: results_guilin/ (özet tablolar, figure_data.json, grafikler, günlük), checkpoints_guilin/ (demo modelleri),
app/examples/guilin/ (her durumdan bir örnek klip, 1000 rpm %15 yük, dosya 5) ve data/guilin_examples.csv.
Büyük skor/önbellek dosyaları dahil edilmez.
"""
import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

RESULT_FILES = ["summary.md", "metrics_summary.csv", "metrics_fold.csv", "bootstrap.json", "same_day.csv",
                "train_log.csv", "run_config.json", "log.txt", "figure_data.json",
                "target_calibration.md", "target_calibration.csv"]
EXAMPLE_COND = (1000, 15, 5, 10)  # rpm, yük, dosya (parça), klip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res, ckpt, wav = Path(a.results), Path(a.ckpt), Path(a.wav)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        rd = root / "results_guilin"
        rd.mkdir()
        for f in RESULT_FILES:
            if (res / f).exists():
                shutil.copy(res / f, rd / f)
        if (res / "figures").exists():
            shutil.copytree(res / "figures", rd / "figures")
        shutil.copytree(ckpt, root / "checkpoints_guilin")
        ex = root / "app" / "examples" / "guilin"
        ex.mkdir(parents=True)
        rpm, load, part, clip = EXAMPLE_COND
        rows = []
        for lab in ("normal", "outer", "inner", "cage", "compound"):
            name = f"{lab}_{rpm}r_{load}_f{part}_c{clip:03d}.wav"
            shutil.copy(wav / lab / name, ex / name)
            rows.append(dict(path=f"app/examples/guilin/{name}", split="test", machine_type="rulman",
                             machine_id="tum_kosullar", label="normal" if lab == "normal" else "anomaly", durum=lab))
        (root / "data").mkdir()
        pd.DataFrame(rows).to_csv(root / "data" / "guilin_examples.csv", index=False)
        with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(root))
    print(f"{a.out}: {Path(a.out).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
