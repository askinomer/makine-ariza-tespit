"""Guilin: koşul-dışı bırakma (leave-one-condition-out) testi.

    python tools/guilin_loco.py --features /content/guilin_check/features.parquet

Her koşul h için: kalan 8 koşulun TÜM normal kayıtlarıyla model öğrenilir (h hiç görülmez).
Skorlanan: h'nin normal kayıtları (görülmemiş koşul/oturum) ve h'nin arıza kayıtları.
AUC(h normal vs h arıza) 0,5'in belirgin üstündeyse arıza, koşul/oturum kaymasından fazla ses bırakıyor demektir.
Üç skor: mahalanobis (Ledoit-Wolf), iforest, kayıp_pca (PCA yeniden oluşturma hatası). Her biri ham ve seviyesiz özellikle.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

COLS = [f"f{i}" for i in range(128)]


def level_free(X):
    X = X.copy()
    X[:, :64] -= X[:, :64].mean(axis=1, keepdims=True)
    return X


def scorers(Xtr):
    sc = StandardScaler().fit(Xtr)
    Z = sc.transform(Xtr)
    lw = LedoitWolf().fit(Xtr)
    iso = IsolationForest(n_estimators=200, random_state=0).fit(Z)
    pca = PCA(n_components=20, random_state=0).fit(Z)
    return {
        "mahalanobis": lambda X: lw.mahalanobis(X),
        "iforest": lambda X: -iso.score_samples(sc.transform(X)),
        "pca_hata": lambda X: ((sc.transform(X) - pca.inverse_transform(pca.transform(sc.transform(X)))) ** 2).mean(axis=1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = pd.read_parquet(a.features)
    faults = sorted(l for l in df.label.unique() if l != "normal")
    conds = sorted(df[["rpm", "load"]].drop_duplicates().itertuples(index=False))
    rows = []
    for variant, f in (("seviye_var", lambda X: X), ("seviye_yok", level_free)):
        for rpm, load in conds:
            held = (df.rpm == rpm) & (df.load == load)
            tr = df[(df.label == "normal") & ~held]
            n_h = df[(df.label == "normal") & held]
            S = scorers(f(tr[COLS].values))
            for name, fn in S.items():
                s_n = fn(f(n_h[COLS].values))
                for fl in faults:
                    s_f = fn(f(df[held & (df.label == fl)][COLS].values))
                    y = np.r_[np.zeros(len(s_n)), np.ones(len(s_f))]
                    rows.append(dict(variant=variant, model=name, rpm=rpm, load=load, fault=fl,
                                     auc=roc_auc_score(y, np.r_[s_n, s_f])))
    r = pd.DataFrame(rows)
    if a.out:
        r.to_csv(a.out, index=False)
    print("LOCO — ortalama AUC (9 koşul), model × arıza türü:")
    print(r.pivot_table(index=["variant", "model"], columns="fault", values="auc", aggfunc="mean").round(3))
    print("\nLOCO — en düşük koşul AUC'si:")
    print(r.pivot_table(index=["variant", "model"], columns="fault", values="auc", aggfunc="min").round(3))
    print("\nMahalanobis seviye_yok — koşul × arıza AUC:")
    print(r[(r.variant == "seviye_yok") & (r.model == "mahalanobis")].pivot_table(
        index=["rpm", "load"], columns="fault", values="auc").round(2))


if __name__ == "__main__":
    main()
