"""Adım 6: tüm kayıtlar için log mel spektrogramı hesaplayıp data/processed altında önbelleğe alır.

Önbellek klasör adı özellik ayarlarının özetini içerir (logmel_<hash>); ayar değişirse
eski önbellek yanlışlıkla kullanılmaz.
"""
import _bootstrap  # noqa: F401

import argparse
import time

import numpy as np
from joblib import Parallel, delayed

from src.data import SPLITS, load_split
from src.features import cache_path, compute_and_cache
from src.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    cfg = load_config(args.config)

    df = load_split(cfg)
    df = df[df["split"].isin(SPLITS)].reset_index(drop=True)
    todo = [r for _, r in df.iterrows() if not cache_path(cfg, r["machine_type"], r["source_recording_id"]).exists()]
    print(f"{len(df)} kayıt; {len(todo)} tanesi hesaplanacak.")
    t0 = time.time()
    if todo:
        Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(compute_and_cache)(cfg, r) for r in todo)
    shapes = {np.load(cache_path(cfg, r["machine_type"], r["source_recording_id"]), mmap_mode="r").shape
              for _, r in df.head(50).iterrows()}
    print(f"Bitti ({time.time() - t0:.1f} s). Örnek matris boyutları (mel × zaman): {sorted(shapes)}")
    print(f"Önbellek: {cache_path(cfg, 'x', 'y').parent.parent}")


if __name__ == "__main__":
    main()
