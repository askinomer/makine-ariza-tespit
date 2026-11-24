"""Aynı koşuldaki normal ve arızalı .bkc kayıtlarını karşılaştırır (veri setini indirmeden önce hızlı kontrol).

    python tools/guilin_compare.py --normal nm1000r-0-32Khz.1.bkc --fault outer=Ods1000r-0-32Khz.1.bkc \
        --fault inner=Ids1000r-0-32Khz.1.bkc --out results/guilin_compare.png

Üretir: (1) güç spektral yoğunluğu, (2) zaman ortalamalı log mel farkı (arıza − normal), (3) zarf spektrumu
(0–400 Hz; mil frekansı ve harmonikleri işaretli), (4) 10 sn'lik klip skorları.

Mini anomali testi (etiket kullanılmaz): normal dosyanın İLK %67 klibinden log mel ortalama+std vektörünün
ortalaması ve kovaryansı (küçültmeli) öğrenilir; kalan normal klipler ve arızalı klipler Mahalanobis uzaklığıyla
skorlanır → AUC. İyimserdir: tutulan normal klipler aynı kaydın komşu dakikalarıdır ve normal/arızalı kayıtlar
farklı oturumlardadır (oturum farkı da ayırt edici olabilir). Yalnızca "sinyal var mı" sorusunu yanıtlar.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert, welch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bkc import read_info, read_samples  # noqa: E402

CH = 1
CLIP_S = 10
MEL = dict(n_fft=2048, hop_length=1024, n_mels=64, fmin=0, fmax=16384)


def logmel(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr, power=2.0, **MEL)
    return librosa.power_to_db(S, ref=1.0, top_db=80.0)


def clip_features(x, sr):
    n = CLIP_S * sr
    feats, mels = [], []
    for k in range(len(x) // n):
        m = logmel(x[k * n:(k + 1) * n], sr)
        mels.append(m.mean(axis=1))
        feats.append(np.concatenate([m.mean(axis=1), m.std(axis=1)]))
    return np.array(feats), np.array(mels)


def envelope_spectrum(x, sr, seconds=60, band=(2000, 10000)):
    from scipy.signal import butter, sosfiltfilt
    y = x[:seconds * sr].astype(np.float64)
    y = sosfiltfilt(butter(4, band, btype="band", fs=sr, output="sos"), y)
    env = np.abs(hilbert(y))
    env -= env.mean()
    f, p = welch(env, fs=sr, nperseg=sr * 4)  # 0,25 Hz çözünürlük
    return f, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", required=True)
    ap.add_argument("--fault", action="append", required=True, help="etiket=dosya.bkc (birden fazla verilebilir)")
    ap.add_argument("--rpm", type=float, default=1000)
    ap.add_argument("--out", default="guilin_compare.png")
    a = ap.parse_args()

    recs = {"normal": a.normal, **dict(f.split("=", 1) for f in a.fault)}
    data = {}
    for lab, p in recs.items():
        info = read_info(p)
        sr = int(info.sample_rate)
        x = read_samples(p, channels=[CH])[0].astype(np.float32)
        feats, mels = clip_features(x, sr)
        data[lab] = dict(x=x, sr=sr, feats=feats, mels=mels, start=info.start_utc)
        print(f"{lab:9s} {Path(p).name:28s} kayıt başlangıcı {info.start_utc}  rms {np.sqrt(np.mean(x.astype(np.float64)**2)):.4f} Pa"
              f"  ({20*np.log10(np.sqrt(np.mean(x.astype(np.float64)**2))/2e-5):.1f} dB SPL)")

    # --- mini anomali testi ---
    nf = data["normal"]["feats"]
    n_fit = int(len(nf) * 2 / 3)
    lw = LedoitWolf().fit(nf[:n_fit])
    score = lambda F: lw.mahalanobis(F)
    s_norm = score(nf[n_fit:])
    print(f"\nMini anomali testi: {n_fit} normal klip ile öğrenildi, {len(nf) - n_fit} normal klip tutuldu.")
    scores = {"normal (tutulan)": s_norm}
    for lab in recs:
        if lab == "normal":
            continue
        s = score(data[lab]["feats"])
        scores[lab] = s
        y = np.r_[np.zeros(len(s_norm)), np.ones(len(s))]
        print(f"  {lab:9s} AUC {roc_auc_score(y, np.r_[s_norm, s]):.3f}   medyan skor {np.median(s):.1f} "
              f"(normal tutulan medyan {np.median(s_norm):.1f})")

    # --- grafikler ---
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    shaft = a.rpm / 60
    for lab, d in data.items():
        f, p = welch(d["x"][:60 * d["sr"]], fs=d["sr"], nperseg=8192)
        ax[0, 0].semilogx(f[1:], 10 * np.log10(p[1:] / (2e-5) ** 2), lw=1, label=lab)
        fe, pe = envelope_spectrum(d["x"], d["sr"])
        m = fe <= 400
        ax[1, 0].plot(fe[m], 10 * np.log10(pe[m] + 1e-30), lw=0.9, label=lab)
    ax[0, 0].set(title=f"Güç spektral yoğunluğu (kanal {CH}, ilk 60 sn)", xlabel="Hz", ylabel="dB re 20 µPa²/Hz",
                 xlim=(20, 16384))
    for h in range(1, 13):
        ax[1, 0].axvline(h * shaft, color="0.75", lw=0.6, zorder=0)
    ax[1, 0].set(title=f"Zarf spektrumu (2–10 kHz bandı; gri = mil {shaft:.2f} Hz harmonikleri)", xlabel="Hz",
                 ylabel="dB")
    mel_f = librosa.mel_frequencies(n_mels=MEL["n_mels"], fmin=MEL["fmin"], fmax=MEL["fmax"])
    base = data["normal"]["mels"].mean(axis=0)
    for lab, d in data.items():
        if lab != "normal":
            ax[0, 1].plot(mel_f, d["mels"].mean(axis=0) - base, label=lab)
    ax[0, 1].axhline(0, color="k", lw=0.8)
    ax[0, 1].set(title="Ortalama log mel farkı (arıza − normal)", xlabel="mel bandı merkez frekansı (Hz)", ylabel="dB")
    ax[1, 1].boxplot(list(scores.values()), tick_labels=list(scores.keys()))
    ax[1, 1].set_yscale("log")
    ax[1, 1].set(title="10 sn'lik klip anomali skoru (Mahalanobis, normalden öğrenildi)", ylabel="skor")
    for a_ in (ax[0, 0], ax[1, 0], ax[0, 1]):
        a_.legend(fontsize=8)
        a_.grid(alpha=0.3)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=110)
    print(f"\nGrafik: {a.out}")


if __name__ == "__main__":
    main()
