"""Guilin deney hattı: küçük sentetik veriyle uçtan uca duman testi + bölünme kuralları."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tools"))

SR = 32768
CONDS = [(800, 0), (1000, 15), (1200, 30)]
LABELS = ["normal", "outer", "inner", "cage", "compound"]


def make_dataset(root: Path, clip_s=2.5, per_part=2):
    rng = np.random.default_rng(0)
    t = np.arange(int(clip_s * SR)) / SR
    for lab in LABELS:
        rows = []
        (root / lab).mkdir(parents=True, exist_ok=True)
        for k, (rpm, load) in enumerate(CONDS):
            for part in range(1, 6):
                for c in range(per_part):
                    y = 0.01 * rng.standard_normal(len(t)) + 0.005 * np.sin(2 * np.pi * (rpm / 60) * 10 * t)
                    if lab != "normal":  # arıza: darbe dizisi
                        y += 0.05 * (np.sin(2 * np.pi * (40 + 10 * LABELS.index(lab)) * t) > 0.995)
                    name = f"{lab}_{rpm}r_{load}_f{part}_c{c:03d}.wav"
                    sf.write(root / lab / name, y.astype(np.float32), SR, subtype="PCM_16")
                    rows.append(dict(path=f"{lab}/{name}", label=lab, rpm=rpm, load=load, part=part, clip=c,
                                     channel=1, source_file=f"x{rpm}.{part}.bkc",
                                     start_utc="2025-11-02T00:00:00" if lab in ("normal", "outer") else "2025-11-03T00:00:00",
                                     clip_start_s=c * clip_s, rms_pa=0.01, peak_pa=0.05, n_clipped=0))
        pd.DataFrame(rows).to_csv(root / f"manifest_{lab}.csv", index=False)


def test_fold_split_has_no_leakage(tmp_path):
    import guilin_experiment as ge
    make_dataset(tmp_path)
    tab = ge.load_table(tmp_path)
    for h in tab.cond.unique():
        tr, va, ca, te = ge.split_fold(tab, h)
        fit = np.r_[tr, va, ca]
        assert (tab.label.iloc[fit] == "normal").all()          # arıza eğitim/doğrulama/kalibrasyonda yok
        assert (tab.cond.iloc[fit] != h).all()                  # test koşulu hiç görülmez
        assert (tab.cond.iloc[te] == h).all()
        assert set(tab.part.iloc[tr]) == {1, 2, 3} and set(tab.part.iloc[va]) == {4} and set(tab.part.iloc[ca]) == {5}
        assert not set(fit) & set(te)


def test_guilin_experiment_end_to_end(tmp_path):
    wav, out, ckpt = tmp_path / "wav", tmp_path / "out", tmp_path / "ckpt"
    make_dataset(wav)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "guilin_experiment.py"), "--wav", str(wav),
                        "--out", str(out), "--ckpt", str(ckpt), "--seeds", "42", "--max-epochs", "2", "--n-boot", "5"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    m = pd.read_csv(out / "metrics_summary.csv")
    assert set(m.method) == {"iforest", "cnn_ae"}
    assert {"temiz", "20", "10"} <= set(m.noise.astype(str))
    assert (out / "summary.md").exists() and json.loads((out / "bootstrap.json").read_text())["n_boot"] == 5
    # demo modeli ortak arayüzle yüklenip aynı ön işlemeyle skorlanabilmeli
    sys.path.insert(0, str(ROOT))
    from src.detectors import available_detectors, load_detector
    dets = available_detectors({"paths": {"checkpoints": str(ckpt)}})
    assert {d["method"] for d in dets} == {"iforest", "cnn_ae"}
    y, _ = sf.read(wav / "normal" / "normal_800r_0_f1_c000.wav", dtype="float32")
    for d in dets:
        det = load_detector(d["method"], d["dir"])
        s = det.score_logmels([det.logmel_from_waveform(y)])
        assert np.isfinite(s).all() and det.threshold is not None
