"""Guilin: kayıt günleri ve frekans bandı başına LOCO testi (kayıt zinciri/oturum artefaktı şüphesi için).

    python tools/guilin_bands.py --wav /content/guilin_wav --features /content/guilin_check/features.parquet
"""
import argparse
import glob
import subprocess
import sys
from pathlib import Path

import pandas as pd

BANDS = [(0, 1000), (1000, 2000), (2000, 5000), (5000, 16385)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--features", required=True)
    a = ap.parse_args()
    m = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{a.wav}/manifest_*.csv"))])
    m["gun"] = m.start_utc.str[:10]
    print("Kayıt günü × etiket (kaynak dosya sayısı):")
    print(m.groupby(["gun", "label"]).source_file.nunique().unstack(1).fillna(0).astype(int))
    loco = Path(__file__).with_name("guilin_loco.py")
    for lo, hi in BANDS:
        r = subprocess.run([sys.executable, str(loco), "--features", a.features, "--fmin", str(lo), "--fmax", str(hi)],
                           capture_output=True, text=True)
        out = [l for l in r.stdout.splitlines() if "Warning" not in l]
        print(f"\n=== BANT {lo}-{hi} Hz ===")
        print("\n".join(out[:12]) if r.returncode == 0 else r.stderr[-800:])


if __name__ == "__main__":
    main()
