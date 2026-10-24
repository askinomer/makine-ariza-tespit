"""DCASE 2020 Task 2 düzeninde KÜÇÜK SENTETİK fan veri seti üretir — yalnızca hattı test etmek için.

Gerçek veri indirilmeden önce tüm betiklerin uçtan uca çalıştığını doğrulamaya yarar.
Bu veriyle elde edilen sayılar HİÇBİR ŞEY KANITLAMAZ; raporda kullanmayın.

    python tools/make_synthetic_dataset.py --out data/synthetic
    python scripts/prepare_data.py --config configs/synthetic.yaml   (vb.)

Hattın kenar durumlarını da sınar: bir kopya kayıt, bir test kopyası, bir kısa ve bir sessiz dosya.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

SR = 16000
DUR = 10.0


def colored_noise(n, rng, slope=0.6):
    w = rng.standard_normal(n)
    spec = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1 / SR)
    spec /= np.maximum(f, 20) ** (slope / 2)
    return np.fft.irfft(spec, n)


def bandpass(x, lo, hi):
    b, a = butter(4, [lo / (SR / 2), hi / (SR / 2)], btype="band")
    return lfilter(b, a, x)


def fan_sound(rng, f0, anomaly=None, severity=1.0):
    n = int(SR * DUR)
    t = np.arange(n) / SR
    f = f0 * (1 + rng.normal(0, 0.01))
    if anomaly == "clog":
        f *= 1 - 0.10 * severity
    hum = np.zeros(n)
    for k in range(1, 16):
        amp = 1.0 / k ** 1.1
        hum += amp * np.sin(2 * np.pi * k * f * t + rng.uniform(0, 2 * np.pi))
    am_depth = 0.05 + (0.6 * severity if anomaly == "imbalance" else 0)
    hum *= 1 + am_depth * np.sin(2 * np.pi * (f / 8) * t)
    bg = colored_noise(n, rng)
    x = 0.25 * hum / np.std(hum) + 0.6 * bg / np.std(bg)
    if anomaly == "rattle":
        imp = np.zeros(n)
        rate = rng.uniform(3, 8)
        for s in rng.uniform(0, DUR, int(rate * DUR * severity)):
            imp[int(s * SR)] = rng.uniform(0.5, 1.0)
        x += 2.5 * severity * bandpass(imp, 2000, 4500) * 8
    if anomaly == "whine":
        fw = rng.uniform(3000, 5000)
        x += 0.35 * severity * np.sin(2 * np.pi * fw * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.3 * t))
    x *= 10 ** (rng.normal(0, 1) / 20) * 0.05
    return x.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--ids", nargs="*", default=["id_00", "id_02"])
    ap.add_argument("--n-train", type=int, default=120)
    ap.add_argument("--n-test-normal", type=int, default=40)
    ap.add_argument("--n-test-anomaly", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    root = Path(args.out) / "fan"
    if root.exists():
        shutil.rmtree(root)
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir(parents=True)
    f0s = {mid: 40 + 9 * i for i, mid in enumerate(args.ids)}
    for mid in args.ids:
        for i in range(args.n_train):
            sf.write(root / "train" / f"normal_{mid}_{i:08d}.wav", fan_sound(rng, f0s[mid]), SR)
        for i in range(args.n_test_normal):
            sf.write(root / "test" / f"normal_{mid}_{i:08d}.wav", fan_sound(rng, f0s[mid]), SR)
        for i in range(args.n_test_anomaly):
            kind = rng.choice(["imbalance", "rattle", "whine", "clog"])
            sev = rng.uniform(0.3, 1.0)
            sf.write(root / "test" / f"anomaly_{mid}_{i:08d}.wav", fan_sound(rng, f0s[mid], kind, sev), SR)
    # Kenar durumları (yalnızca ilk kimlikte)
    mid = args.ids[0]
    tr = root / "train"
    shutil.copy(tr / f"normal_{mid}_00000000.wav", tr / f"normal_{mid}_{args.n_train:08d}.wav")          # kopya
    shutil.copy(root / "test" / f"normal_{mid}_00000001.wav", tr / f"normal_{mid}_{args.n_train + 1:08d}.wav")  # test kopyası
    sf.write(tr / f"normal_{mid}_{args.n_train + 2:08d}.wav", fan_sound(rng, f0s[mid])[: SR], SR)           # 1 s, kısa
    sf.write(tr / f"normal_{mid}_{args.n_train + 3:08d}.wav", np.zeros(int(SR * DUR), np.float32), SR)     # sessiz
    n = len(list(root.rglob("*.wav")))
    print(f"{n} sentetik WAV yazıldı → {root}")


if __name__ == "__main__":
    main()
