"""Rapor ve sunum için tüm sonuçları tek bir JSON dosyasında toplar (yalnızca standart kütüphane).

    python tools/collect_results.py > results/report_numbers.json

Okunanlar: results/*.csv, checkpoints/**/meta.json ve threshold.json, logs/run_real.log,
data/manifest.csv, results/test_access_log.csv, logs/installed_versions.txt.
"""
import csv
import glob
import json
import os
import re
import statistics as st
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def path(*p):
    return os.path.join(ROOT, *p)


def read_csv(p):
    if not os.path.exists(p):
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        x = float(v)
        return None if x != x else x  # NaN -> None
    except (TypeError, ValueError):
        return v


def clean(rows):
    return [{k: num(v) for k, v in r.items()} for r in rows]


out = {}

# Veri sayıları ve kalite
out["data_counts"] = clean(read_csv(path("results", "data_counts.csv")))
man = read_csv(path("data", "manifest.csv"))
if man:
    issues = {}
    for r in man:
        for i in filter(None, (r.get("issue") or "").split(";")):
            issues[i] = issues.get(i, 0) + 1
    md5 = {}
    for r in man:
        if r.get("content_md5"):
            md5[r["content_md5"]] = md5.get(r["content_md5"], 0) + 1
    out["manifest"] = {
        "n_files": len(man),
        "issues": issues,
        "duplicate_groups": sum(1 for v in md5.values() if v > 1),
        "duplicate_files": sum(v for v in md5.values() if v > 1),
        "splits": {s: sum(1 for r in man if r["split"] == s) for s in sorted({r["split"] for r in man})},
        "sample_rates": sorted({r["sample_rate"] for r in man}),
        "durations": [min(float(r["duration"]) for r in man), max(float(r["duration"]) for r in man)],
    }

# Ana sonuçlar
per_id = clean(read_csv(path("results", "metrics_per_id.csv")))
out["per_id"] = per_id
keys = ["auc", "pauc", "precision", "recall", "f1", "far"]
macro = {}
for r in per_id:
    macro.setdefault((r["method"], r["seed"]), []).append(r)
by_method = {}
for (m, s), rows in macro.items():
    by_method.setdefault(m, []).append({k: st.mean(x[k] for x in rows if x[k] is not None) for k in keys})
out["macro"] = {
    m: {k: {"mean": st.mean(v[k] for v in vals), "std": (st.stdev(v[k] for v in vals) if len(vals) > 1 else None),
            "per_seed": [v[k] for v in vals]} for k in keys} | {"n_seeds": len(vals)}
    for m, vals in by_method.items()
}
# Kimlik başına tohum ortalaması (AE)
seedavg = {}
for r in per_id:
    seedavg.setdefault((r["method"], r["machine_id"]), []).append(r)
out["per_id_seed_mean"] = [
    {"method": m, "machine_id": i, "n": len(rows),
     **{k: st.mean(x[k] for x in rows if x[k] is not None) for k in keys},
     **{f"{k}_std": (st.stdev(x[k] for x in rows) if len(rows) > 1 else None) for k in keys}}
    for (m, i), rows in sorted(seedavg.items())
]
out["summary_md"] = open(path("results", "summary.md"), encoding="utf-8").read() if os.path.exists(path("results", "summary.md")) else None
out["noise_summary"] = clean(read_csv(path("results", "noise_summary.csv")))
out["noise_per_id"] = clean(read_csv(path("results", "noise_results.csv")))
out["exploratory"] = clean(read_csv(path("results", "exploratory_aggregations.csv")))
out["test_access_log"] = read_csv(path("results", "test_access_log.csv"))

# Eğitim ve eşik bilgileri
models = []
for meta_p in sorted(glob.glob(path("checkpoints", "**", "meta.json"), recursive=True)):
    meta = json.load(open(meta_p, encoding="utf-8"))
    meta.pop("features", None)
    thr_p = os.path.join(os.path.dirname(meta_p), "threshold.json")
    if os.path.exists(thr_p):
        meta["threshold_info"] = json.load(open(thr_p, encoding="utf-8"))
    hist_p = os.path.join(os.path.dirname(meta_p), "history.json")
    if os.path.exists(hist_p):
        h = json.load(open(hist_p, encoding="utf-8"))
        meta["history"] = {"train_loss": h["train_loss"], "val_loss": h["val_loss"]}
    meta["dir"] = os.path.relpath(os.path.dirname(meta_p), ROOT)
    models.append(meta)
out["models"] = models

# Adım süreleri
log_p = path("logs", "run_real.log")
if os.path.exists(log_p):
    txt = open(log_p, encoding="utf-8", errors="replace").read()
    out["step_times_s"] = [{"step": a, "s": int(b)} for a, b in re.findall(r"^--- (\S+) bitti \((\d+) s\)", txt, re.M)]
    out["overfit"] = re.findall(r"Aşırı öğrenme testi: .*", txt)
    out["run_header"] = re.findall(r"^##### .*", txt, re.M)
# Hata analizi
err = {}
for p in sorted(glob.glob(path("results", "error_analysis", "*.csv"))):
    err[os.path.basename(p)] = read_csv(p)[:20]
out["error_analysis"] = err
vers = path("logs", "installed_versions.txt")
if os.path.exists(vers):
    out["versions"] = [l.strip() for l in open(vers) if re.match(r"(torch|librosa|numpy|scikit-learn|pandas|scipy|streamlit)==", l)]

json.dump(out, sys.stdout, ensure_ascii=False, indent=1, default=str)
