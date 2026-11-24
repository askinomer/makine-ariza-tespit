"""Guilin veri seti ön kontrolü: AUC'nin "arıza"dan mı yoksa "oturum/ses seviyesi"nden mi geldiğini ölçer.

    python tools/guilin_session_check.py --wav /content/guilin_wav --out /content/guilin_check

Protokol (koşul = hız × yük, 9 koşul):
  normal parça 1-3 → model öğrenir (180 klip) · parça 4 → doğrulama · parça 5 → normal TEST (60 klip)
  arıza: ilgili koşulun 5 parçasının tamamı (300 klip) → anomali TEST
Skor: Ledoit-Wolf kovaryanslı Mahalanobis uzaklığı, klip başına 128 boyutlu log mel (ortalama+std) vektörü.
Varyantlar:
  seviye_var   : ham log mel (ses seviyesi dahil)
  seviye_yok   : her kliple log mel ortalaması (genel seviye) çıkarılır → yalnızca spektral ŞEKİL
  tek_model    : 9 koşulun normal parça 1-3'ü birleştirilir, tek model (koşul bilgisi yok)
Ek: normal parçaların model skoru ortalaması (parça 1-3 → 4 → 5) = aynı kayıt içi kayma.
Sonuç AUC'leri optimistiktir: normal ve arıza farklı oturumlardan; yalnızca "ayrım şekil/seviye mi" sorusuna bakar.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

SR = 32768
MEL = dict(n_fft=2048, hop_length=1024, n_mels=64, fmin=0, fmax=16384)


def clip_vec(path: str) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32")
    assert sr == SR, sr
    m = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, power=2.0, **MEL), ref=1.0, top_db=80.0)
    return np.concatenate([m.mean(axis=1), m.std(axis=1)]).astype(np.float32)


def features(wav: Path, out: Path) -> pd.DataFrame:
    cache = out / "features.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    m = pd.concat([pd.read_csv(f) for f in sorted(wav.glob("manifest_*.csv"))], ignore_index=True)
    X = Parallel(n_jobs=-1, batch_size=32)(delayed(clip_vec)(str(wav / p)) for p in m["path"])
    df = pd.concat([m[["path", "label", "rpm", "load", "part", "clip"]],
                    pd.DataFrame(np.vstack(X), columns=[f"f{i}" for i in range(128)])], axis=1)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def level_free(X: np.ndarray) -> np.ndarray:
    X = X.copy()
    X[:, :64] -= X[:, :64].mean(axis=1, keepdims=True)  # ortalama log mel: genel seviye çıkar
    return X


def auc(s_norm, s_fault):
    y = np.r_[np.zeros(len(s_norm)), np.ones(len(s_fault))]
    return roc_auc_score(y, np.r_[s_norm, s_fault])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    df = features(Path(a.wav), out)
    cols = [f"f{i}" for i in range(128)]
    faults = [l for l in df.label.unique() if l != "normal"]
    rows, drift = [], []
    conds = sorted(df[["rpm", "load"]].drop_duplicates().itertuples(index=False))

    for variant in ("seviye_var", "seviye_yok", "tek_model"):
        f = (lambda X: level_free(X)) if variant == "seviye_yok" else (lambda X: X)
        pooled = None
        if variant == "tek_model":
            tr = df[(df.label == "normal") & (df.part <= 3)]
            pooled = LedoitWolf().fit(f(tr[cols].values))
        for rpm, load in conds:
            c = df[(df.rpm == rpm) & (df.load == load)]
            n = c[c.label == "normal"]
            lw = pooled or LedoitWolf().fit(f(n[n.part <= 3][cols].values))
            s_test = lw.mahalanobis(f(n[n.part == 5][cols].values))
            if variant != "tek_model":
                for p in (3, 4, 5):
                    drift.append(dict(rpm=rpm, load=load, part=p, skor=np.median(lw.mahalanobis(f(n[n.part == p][cols].values)))))
            for fl in faults:
                s_f = lw.mahalanobis(f(c[c.label == fl][cols].values))
                rows.append(dict(variant=variant, rpm=rpm, load=load, fault=fl, auc=auc(s_test, s_f)))

    r = pd.DataFrame(rows)
    r.to_csv(out / "session_check_auc.csv", index=False)
    print("Ortalama AUC (9 koşul), varyant × arıza türü:")
    print(r.pivot_table(index="variant", columns="fault", values="auc", aggfunc="mean").round(3))
    print("\nEn düşük koşul AUC'si:")
    print(r.pivot_table(index="variant", columns="fault", values="auc", aggfunc="min").round(3))
    d = pd.DataFrame(drift)
    d["skor_göreli"] = d.groupby(["rpm", "load"]).skor.transform(lambda s: s / s.iloc[0])
    print("\nNormal parçalarda model skoru (parça 3 = 1.0): kayıt içi kayma")
    print(d.pivot_table(index=["rpm", "load"], columns="part", values="skor_göreli").round(2))
    lvl = df.groupby("label").apply(lambda g: g[cols[:64]].mean(axis=1).mean()).round(2)
    print("\nKlip başına ortalama log mel (dB, genel seviye göstergesi):")
    print(lvl.to_string())


if __name__ == "__main__":
    main()
