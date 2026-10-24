"""Tüm hattı sırayla çalıştırır.

    python run_pipeline.py                                # gerçek veri (configs/config.yaml)
    python run_pipeline.py --config configs/synthetic.yaml --synthetic   # sentetik hat testi
    python run_pipeline.py --machine-ids id_00            # tek kimlik prototipi
    python run_pipeline.py --seeds 42 43 44               # 3 tohum tekrarı

Sıra: prepare_data → extract_features → train_baseline → train_autoencoder (overfit testi + eğitim)
      → calibrate → evaluate → noise_experiment
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(script, *args):
    cmd = [sys.executable, str(ROOT / script), *map(str, args)]
    print(f"\n=== {' '.join(cmd[1:])}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode:
        sys.exit(f"Durdu: {script} hata verdi (kod {r.returncode}).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--synthetic", action="store_true", help="Önce sentetik veri seti üret")
    ap.add_argument("--machine-ids", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--skip-noise", action="store_true")
    args = ap.parse_args()

    c = ["--config", args.config]
    ids = ["--machine-ids", *args.machine_ids] if args.machine_ids else []
    seeds = args.seeds or []
    if args.synthetic:
        run("tools/make_synthetic_dataset.py", "--out", "data/synthetic")
        run("scripts/prepare_data.py", *c, "--force")
    else:
        run("scripts/prepare_data.py", *c)
    run("scripts/extract_features.py", *c)
    run("scripts/train_baseline.py", *c, *ids)
    run("scripts/train_autoencoder.py", *c, *ids, "--overfit-test")
    for s in (seeds or [None]):
        run("scripts/train_autoencoder.py", *c, *ids, *(["--seed", s] if s is not None else []))
    sd = ["--seeds", *seeds] if seeds else []
    run("scripts/calibrate.py", *c, *ids, *sd)
    run("scripts/evaluate.py", *c, *ids, *sd)
    if not args.skip_noise:
        run("scripts/noise_experiment.py", *c, *ids, *sd, "--recalibrate")
    print("\nHat tamamlandı. Demo: streamlit run app/app.py")


if __name__ == "__main__":
    main()
