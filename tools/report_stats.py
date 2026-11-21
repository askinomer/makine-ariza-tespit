"""Rapor için ek istatistikler: tohum ortalamalı AUC/pAUC ve eşleştirilmiş bootstrap farkı.

    python tools/report_stats.py                 # results/scores/test_*.csv dosyalarından
    python tools/report_stats.py --n-boot 2000

Her bootstrap tekrarında test kayıtları sınıf içinde yeniden örneklenir ve AYNI örnek iki yönteme
uygulanır (eşleştirilmiş). CNN AE için her tekrarda tohumların AUC'si hesaplanıp ortalanır; böylece
aralık test örnekleminin belirsizliğini, nokta değer üç tohumun ortalamasını yansıtır.
Çıktı: results/report_stats.json (+ ekrana özet).
"""
import argparse
import glob
import json
import os
import re

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_scores(res_dir):
    """{(method, id): {"y": array, "S": array (n_seeds × n_files), "paths": list}}"""
    data = {}
    for p in sorted(glob.glob(os.path.join(res_dir, "scores", "test_*.csv"))):
        name = os.path.basename(p)[5:-4]  # iforest_fan_id_00 | cnn_ae_seed42_fan_id_00
        m = re.match(r"(iforest|cnn_ae)(?:_seed(\d+))?_(\w+?)_(id_\d+)$", name)
        if not m:
            continue
        method, _, _mt, mid = m.groups()
        import csv
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        paths = [r["path"] for r in rows]
        y = np.array([1 if r["label"] == "anomaly" else 0 for r in rows])
        s = np.array([float(r["score"]) for r in rows])
        d = data.setdefault((method, mid), {"y": y, "paths": paths, "S": []})
        assert d["paths"] == paths, f"{p}: dosya sırası farklı"
        d["S"].append(s)
    for d in data.values():
        d["S"] = np.vstack(d["S"])
    return data


def auc_pair(y, S, max_fpr):
    a = np.mean([roc_auc_score(y, s) for s in S])
    p = np.mean([roc_auc_score(y, s, max_fpr=max_fpr) for s in S])
    return a, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--max-fpr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    data = load_scores(args.results)
    ids = sorted({mid for (_, mid) in data})
    methods = [m for m in ("iforest", "cnn_ae") if any((m, i) in data for i in ids)]
    rng = np.random.default_rng(args.seed)

    point = {(m, i): auc_pair(data[(m, i)]["y"], data[(m, i)]["S"], args.max_fpr) for m in methods for i in ids}
    boots = {(m, i): [] for m in methods for i in ids}
    for _ in range(args.n_boot):
        for i in ids:
            y = data[(methods[0], i)]["y"]
            i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
            b = np.concatenate([rng.choice(i0, len(i0)), rng.choice(i1, len(i1))])
            for m in methods:
                d = data[(m, i)]
                boots[(m, i)].append(auc_pair(d["y"][b], d["S"][:, b], args.max_fpr))
    B = {k: np.array(v) for k, v in boots.items()}  # n_boot × 2 (auc, pauc)

    def ci(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    out = {"n_boot": args.n_boot, "max_fpr": args.max_fpr, "ids": ids, "methods": methods,
           "n_seeds": {m: int(data[(m, ids[0])]["S"].shape[0]) for m in methods}, "per_id": [], "macro": {}, "diff": {}}
    for i in ids:
        y = data[(methods[0], i)]["y"]
        row = {"machine_id": i, "n_normal": int((y == 0).sum()), "n_anomaly": int((y == 1).sum())}
        for m in methods:
            row[f"{m}_auc"], row[f"{m}_pauc"] = map(float, point[(m, i)])
            row[f"{m}_auc_ci"], row[f"{m}_pauc_ci"] = ci(B[(m, i)][:, 0]), ci(B[(m, i)][:, 1])
        if len(methods) == 2:
            d = B[("cnn_ae", i)] - B[("iforest", i)]
            row["diff_auc"] = row["cnn_ae_auc"] - row["iforest_auc"]
            row["diff_auc_ci"] = ci(d[:, 0])
            row["diff_pauc"] = row["cnn_ae_pauc"] - row["iforest_pauc"]
            row["diff_pauc_ci"] = ci(d[:, 1])
        out["per_id"].append(row)
    for m in methods:
        mb = np.mean([B[(m, i)] for i in ids], axis=0)
        out["macro"][m] = {"auc": float(np.mean([point[(m, i)][0] for i in ids])),
                           "pauc": float(np.mean([point[(m, i)][1] for i in ids])),
                           "auc_ci": ci(mb[:, 0]), "pauc_ci": ci(mb[:, 1])}
    if len(methods) == 2:
        d = np.mean([B[("cnn_ae", i)] - B[("iforest", i)] for i in ids], axis=0)
        out["diff"] = {"auc": out["macro"]["cnn_ae"]["auc"] - out["macro"]["iforest"]["auc"], "auc_ci": ci(d[:, 0]),
                       "pauc": out["macro"]["cnn_ae"]["pauc"] - out["macro"]["iforest"]["pauc"], "pauc_ci": ci(d[:, 1])}
    with open(os.path.join(args.results, "report_stats.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    for r in out["per_id"]:
        print(r["machine_id"], {k: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v] if isinstance(v, list) else v)
                                for k, v in r.items() if k != "machine_id"})
    print("macro", json.dumps(out["macro"], indent=0)[:400])
    print("diff", out["diff"])


if __name__ == "__main__":
    main()
