"""Adım 11: son test. Ayarlar ve eşikler DONDURULDUKTAN sonra çalıştırın.

Ölçütler: ROC AUC, pAUC (max_fpr=0.1, sklearn standartlaştırılmış), precision, recall, F1,
yanlış alarm oranı (FP/(FP+TN)); kimlik başına ve kimliklerin eşit ağırlıklı ortalaması.
Bootstrap ile %95 güven aralıkları eklenir.

Her çalıştırma results/test_access_log.csv dosyasına yazılır. Testten sonra model veya ayar
değiştirirseniz bunu raporda belirtin (önceki test sonucu model seçiminde kullanılmış olur).

Çıktılar:
    results/metrics_per_id.csv, results/metrics_summary.csv, results/summary.md
    results/scores/test_<yöntem>_<tür>_<id>.csv     dosya başına skor ve karar
    results/error_analysis/<yöntem>_<tür>_<id>.csv  en kötü yanlış alarm ve kaçırılan arızalar
    results/figures/                                  skor dağılımları, ROC eğrileri, confusion matrix
"""
import _bootstrap  # noqa: F401

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src.data import load_split, machine_keys, subset
from src.detectors import METHOD_LABELS, METHODS, detector_dir, load_detector
from src.features import load_logmels
from src.metrics import all_metrics, bootstrap_ci
from src.plotstyle import CLASS_COLORS, GRID, INK2, METHOD_COLORS, MUTED, apply as apply_style, seq_cmap, text_on
from src.utils import ensure_dir, git_commit, load_config

apply_style()

METRIC_COLS = ["auc", "pauc", "precision", "recall", "f1", "far"]


def tag_of(method, seed):
    return method if method == "iforest" else f"{method}_seed{seed}"


def plot_distribution(scores, y, thr, title, path):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bins = np.histogram_bin_edges(scores, bins=40)
    for cls, lab, name in ((0, "normal", "normal"), (1, "anomaly", "anomali")):
        c = CLASS_COLORS[lab]
        ax.hist(scores[y == cls], bins=bins, histtype="stepfilled", color=c, alpha=0.22, lw=0)
        ax.hist(scores[y == cls], bins=bins, histtype="step", color=c, lw=1.6,
                label=f"{name} (n={int((y == cls).sum())})")
    ax.axvline(thr, color=INK2, lw=1.2)
    ax.annotate("eşik (kalibrasyon p95)", xy=(thr, 1), xycoords=("data", "axes fraction"),
                xytext=(4, -4), textcoords="offset points", ha="left", va="top", fontsize=8, color=INK2)
    ax.set_xlabel("anomali skoru")
    ax.set_ylabel("kayıt sayısı")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion(m, title, path):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(3.8, 3.3))
    # Satır bazında oran rengi: sınıf büyüklükleri çok farklı olduğu için ham sayı yanıltır
    rates = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(rates, cmap=seq_cmap(), vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({rates[i, j]:.0%})", ha="center", va="center", fontsize=10.5,
                    color=text_on(rates[i, j], 1.0))
    ax.set_xticks([0, 1], ["normal", "anomali şüphesi"])
    ax.set_yticks([0, 1], ["normal", "anomali"])
    ax.set_xlabel("tahmin")
    ax.set_ylabel("gerçek")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def evaluate_one(cfg, split_df, method, mt, mid, seed, fig_dir, exploratory=False):
    d = detector_dir(cfg, method, mt, mid, seed)
    if not (d / "threshold.json").exists():
        print(f"  atlandı (model veya eşik yok): {d}")
        return None, None
    det = load_detector(method, d)
    test = subset(split_df, mt, mid, "test")
    y = test["label_int"].to_numpy()
    lms = load_logmels(cfg, test)
    scores = det.score_logmels(lms)
    thr = det.threshold
    sc = cfg["scoring"]
    m = all_metrics(y, scores, thr, sc["pauc_max_fpr"])
    m.update(bootstrap_ci(y, scores, thr, sc["bootstrap"], sc["pauc_max_fpr"], seed=cfg["seed"]))
    tag = tag_of(method, det.meta["seed"])
    row = {"method": method, "seed": det.meta["seed"], "machine_type": mt, "machine_id": mid,
           "threshold": thr, "aggregation": det.meta.get("aggregation", "-"), **m}

    pred = (scores > thr).astype(int)
    scores_dir = ensure_dir(Path(cfg["paths"]["results"]) / "scores")
    pd.DataFrame({"path": test["path"], "label": test["label"], "score": scores, "pred": pred}).to_csv(
        scores_dir / f"test_{tag}_{mt}_{mid}.csv", index=False)

    err_dir = ensure_dir(Path(cfg["paths"]["results"]) / "error_analysis")
    df = pd.DataFrame({"path": test["path"], "label": test["label"], "score": scores, "threshold": thr})
    fps = df[(df.label == "normal") & (df.score > thr)].sort_values("score", ascending=False).head(10)
    fns = df[(df.label == "anomaly") & (df.score <= thr)].sort_values("score").head(10)
    pd.concat([fps.assign(kind="yanlış alarm (FP)"), fns.assign(kind="kaçırılan anomali (FN)")]).to_csv(
        err_dir / f"{tag}_{mt}_{mid}.csv", index=False)

    title = f"{METHOD_LABELS[method]} — {mt} {mid}"
    plot_distribution(scores, y, thr, title, fig_dir / f"dist_{tag}_{mt}_{mid}.png")
    plot_confusion(m, title, fig_dir / f"cm_{tag}_{mt}_{mid}.png")

    explore = []
    if exploratory and method == "cnn_ae":
        for how in ("max", "top3"):
            s2 = det.score_logmels(lms, how=how)
            r = ranking_metrics_only(y, s2, sc["pauc_max_fpr"])
            explore.append({"method": method, "seed": det.meta["seed"], "machine_type": mt,
                            "machine_id": mid, "aggregation": how, **r})
    return row, {"y": y, "scores": scores, "explore": explore}


def ranking_metrics_only(y, s, max_fpr):
    from src.metrics import ranking_metrics
    return ranking_metrics(y, s, max_fpr)


def fmt_ci(row, k):
    v = row[k]
    ci = row.get(f"{k}_ci")
    if isinstance(ci, (tuple, list)) and len(ci) == 2 and not np.isnan(v):
        return f"{v:.3f} [{ci[0]:.3f}–{ci[1]:.3f}]"
    return "—" if pd.isna(v) else f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--machine-ids", nargs="*", default=None)
    ap.add_argument("--exploratory-aggregations", action="store_true",
                    help="AE için max/top3 toplama kurallarını AYRICA raporla (keşif amaçlı, model seçimi için değil)")
    ap.add_argument("--note", default="", help="Test erişim kaydına not")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seeds = args.seeds or [cfg["seed"]]
    res = ensure_dir(cfg["paths"]["results"])
    fig_dir = ensure_dir(res / "figures")
    split_df = load_split(cfg)

    log_path = res / "test_access_log.csv"
    prev = len(pd.read_csv(log_path)) if log_path.exists() else 0
    if prev:
        print(f"UYARI: son test bu projede daha önce {prev} kez çalıştırıldı. Bu çalıştırmadan önce "
              "model/ayar değiştirdiyseniz raporda belirtin.")

    rows, curves, explore = [], {}, []
    for mt, mid in machine_keys(split_df):
        if args.machine_ids and mid not in args.machine_ids:
            continue
        for method in args.methods:
            for s in (seeds if method == "cnn_ae" else [None]):
                row, extra = evaluate_one(cfg, split_df, method, mt, mid, s, fig_dir, args.exploratory_aggregations)
                if row is None:
                    continue
                rows.append(row)
                curves[(method, row["seed"], mt, mid)] = extra
                explore += extra["explore"]
                print(f"  [{tag_of(method, row['seed'])}] {mt} {mid}: AUC={row['auc']:.3f} pAUC={row['pauc']:.3f} "
                      f"recall={row['recall']:.3f} F1={row['f1']:.3f} FAR={row['far']:.3f}")
    if not rows:
        print("Değerlendirilecek model yok. Önce eğitim ve kalibrasyon betiklerini çalıştırın.")
        return

    per_id = pd.DataFrame(rows)
    csv_df = per_id.copy()
    for col in [c for c in csv_df.columns if c.endswith("_ci")]:
        csv_df[f"{col}_lo"] = csv_df[col].map(lambda t: t[0] if isinstance(t, (tuple, list)) else np.nan)
        csv_df[f"{col}_hi"] = csv_df[col].map(lambda t: t[1] if isinstance(t, (tuple, list)) else np.nan)
        csv_df = csv_df.drop(columns=col)
    csv_df.to_csv(res / "metrics_per_id.csv", index=False)

    # Kimliklerin eşit ağırlıklı ortalaması (yöntem × tohum)
    macro = per_id.groupby(["method", "seed"])[METRIC_COLS].mean().reset_index()
    macro["n_ids"] = per_id.groupby(["method", "seed"]).size().values
    # Tohumlar arası ortalama ± std (cnn_ae için birden çok tohum varsa)
    summary = macro.groupby("method")[METRIC_COLS].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary["n_seeds"] = macro.groupby("method").size()
    summary.reset_index().to_csv(res / "metrics_summary.csv", index=False)

    # ROC eğrileri: kimlik başına yöntemler üst üste
    for mt, mid in sorted({(k[2], k[3]) for k in curves}):
        fig, ax = plt.subplots(figsize=(4.5, 4.2))
        for (method, seed, m2, i2), ex in curves.items():
            if (m2, i2) != (mt, mid) or len(np.unique(ex["y"])) < 2:
                continue
            fpr, tpr, _ = roc_curve(ex["y"], ex["scores"])
            seeds_here = sorted(s for (m3, s, a3, b3) in curves if m3 == method and (a3, b3) == (mt, mid))
            first = seed == seeds_here[0]
            alpha = 1.0 if (method != "cnn_ae" or first) else 0.45
            lab = METHOD_LABELS[method] + (f" (tohum {seed})" if method == "cnn_ae" and len(seeds_here) > 1 else "")
            ax.plot(fpr, tpr, color=METHOD_COLORS[method], alpha=alpha, lw=2 if alpha == 1 else 1.2, label=lab)
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8)
        ax.axvspan(0, cfg["scoring"]["pauc_max_fpr"], color=GRID, alpha=0.6, lw=0, label="pAUC bölgesi")
        ax.set_xlabel("yanlış alarm oranı (FPR)")
        ax.set_ylabel("yakalama oranı (TPR)")
        ax.set_title(f"ROC — {mt} {mid}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.legend(fontsize=7.5, loc="lower right")
        fig.tight_layout()
        fig.savefig(fig_dir / f"roc_{mt}_{mid}.png")
        plt.close(fig)

    # Markdown özeti
    lines = [f"# Son test sonuçları\n", f"Tarih: {datetime.now():%Y-%m-%d %H:%M} · git: {git_commit()}  ",
             "Eşikler normal kalibrasyon kayıtlarının 95. persentilinden; köşeli parantez = %95 bootstrap GA.\n",
             "## Kimlik başına\n",
             "| Yöntem | Tohum | Kimlik | n (normal/anomali) | AUC | pAUC | Recall | F1 | Yanlış alarm |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in per_id.iterrows():
        lines.append(f"| {METHOD_LABELS[r['method']]} | {r['seed']} | {r['machine_type']} {r['machine_id']} | "
                     f"{r['n_normal']}/{r['n_anomaly']} | {fmt_ci(r, 'auc')} | {fmt_ci(r, 'pauc')} | "
                     f"{fmt_ci(r, 'recall')} | {r['f1']:.3f} | {fmt_ci(r, 'far')} |")
    lines += ["\n## Kimliklerin eşit ağırlıklı ortalaması\n",
              "| Yöntem | Tohum sayısı | AUC | pAUC | Precision | Recall | F1 | Yanlış alarm |",
              "|---|---|---|---|---|---|---|---|"]
    for method, r in summary.iterrows():
        def ms(k):
            s = r[f"{k}_std"]
            return f"{r[f'{k}_mean']:.3f}" + (f" ± {s:.3f}" if r["n_seeds"] > 1 and not pd.isna(s) else "")
        lines.append(f"| {METHOD_LABELS[method]} | {int(r['n_seeds'])} | " + " | ".join(ms(k) for k in METRIC_COLS) + " |")
    if explore:
        ex = pd.DataFrame(explore)
        ex.to_csv(res / "exploratory_aggregations.csv", index=False)
        lines += ["\n## Keşif amaçlı: alternatif toplama kuralları (model seçiminde KULLANILMADI)\n",
                  "| Tohum | Kimlik | Kural | AUC | pAUC |", "|---|---|---|---|---|"]
        for _, r in ex.iterrows():
            lines.append(f"| {r['seed']} | {r['machine_id']} | {r['aggregation']} | {r['auc']:.3f} | {r['pauc']:.3f} |")
    (res / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    entry = pd.DataFrame([{"timestamp": datetime.now().isoformat(timespec="seconds"), "git_commit": git_commit(),
                           "methods": " ".join(args.methods), "seeds": " ".join(map(str, seeds)),
                           "n_evaluations": len(per_id), "note": args.note}])
    entry.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    print(f"\nÖzet: {res / 'summary.md'}")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
