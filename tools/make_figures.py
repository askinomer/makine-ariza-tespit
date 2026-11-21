"""Sunum ve rapor için özet grafikler (results/report_stats.json ve results/noise_summary.csv'den).

    python tools/report_stats.py      # önce
    python tools/make_figures.py      # results/figures/sunum_*.png

Grafikler model veya test sonucu üretmez; yalnızca kaydedilmiş sonuçları çizer.
"""
import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INK, PAPER, ACCENT, GREY, SOFT, LINE = "#16202C", "#F6F5F1", "#2F6FD6", "#8A94A3", "#4A5568", "#C6CDD6"
# DCASE 2020 Görev 2 resmî baseline, fan, geliştirme testi (README, y-kawagu/dcase2020_task2_baseline)
DCASE_BASELINE_AUC = {"id_00": 0.5396, "id_02": 0.7219, "id_04": 0.6221, "id_06": 0.7228}
LABEL = {"iforest": "Isolation Forest", "cnn_ae": "CNN autoencoder"}


def tr(v, nd=3):
    """Türkçe ondalık: 0.718 -> 0,718"""
    return f"{v:.{nd}f}".replace(".", ",")


def tr_axis(ax, axis="x", nd=1):
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: tr(v, nd))
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(f)


def style(ax):
    ax.set_facecolor(PAPER)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(colors=SOFT, labelsize=20, length=0)
    ax.grid(axis="x", color=LINE, lw=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def auc_dumbbell(stats, out):
    ids = stats["ids"]
    rows = [r for r in stats["per_id"]]
    labels = ids + ["Ortalama"]
    fig, ax = plt.subplots(figsize=(12, 6.4), dpi=200)
    fig.patch.set_facecolor(PAPER)
    style(ax)
    ys = list(range(len(labels)))[::-1]
    for y, lab in zip(ys, labels):
        if lab == "Ortalama":
            m = stats["macro"]
            vals = {k: (m[k]["auc"], m[k]["auc_ci"]) for k in ("iforest", "cnn_ae")}
            base = sum(DCASE_BASELINE_AUC.values()) / len(DCASE_BASELINE_AUC)
        else:
            r = next(r for r in rows if r["machine_id"] == lab)
            vals = {k: (r[f"{k}_auc"], r[f"{k}_auc_ci"]) for k in ("iforest", "cnn_ae")}
            base = DCASE_BASELINE_AUC.get(lab)
        for k, col, off in (("iforest", GREY, -0.17), ("cnn_ae", ACCENT, 0.17)):
            v, (lo, hi) = vals[k]
            ax.plot([lo, hi], [y + off, y + off], color=col, lw=2.2, alpha=0.55, zorder=2, solid_capstyle="butt")
            ax.scatter([v], [y + off], s=170, color=col, zorder=3, edgecolor=PAPER, linewidth=1.5)
            ax.text(hi + 0.008, y + off, tr(v), va="center", ha="left", fontsize=17,
                    color=INK if k == "cnn_ae" else SOFT, fontweight="bold" if k == "cnn_ae" else "normal")
        if base is not None:
            ax.plot([base, base], [y - 0.36, y + 0.36], color=INK, lw=2.5, zorder=4)
    ax.axvline(0.5, color=SOFT, lw=1.2, ls=(0, (4, 4)), zorder=0)
    tr_axis(ax, "x", 1)
    ax.set_yticks(ys, labels, fontsize=22, color=INK)
    ax.axhline(0.5, color=LINE, lw=1)
    lo = min(0.45, min(min(r["iforest_auc_ci"][0], r["cnn_ae_auc_ci"][0]) for r in rows) - 0.02)
    ax.set_xlim(lo, 1.03)
    ax.set_xlabel("ROC AUC (test, %95 bootstrap güven aralığı)", fontsize=19, color=SOFT, labelpad=12)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=ACCENT, ms=13, label="CNN autoencoder (3 tohum ort.)"),
         Line2D([], [], marker="o", ls="", color=GREY, ms=13, label="Isolation Forest"),
         Line2D([], [], marker="|", ls="", color=INK, ms=20, mew=2.5, label="DCASE resmî baseline"),
         Line2D([], [], color=SOFT, lw=1.2, ls=(0, (4, 4)), label="şans (0,5)")]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=16,
              labelcolor=INK, handletextpad=0.4, columnspacing=1.6)
    fig.tight_layout()
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)


def noise_panels(noise_rows, out):
    from matplotlib.lines import Line2D
    conds = ["özgün", "+20 dB", "+10 dB"]
    fixed = [r for r in noise_rows if r["threshold_source"].startswith("sabit")]
    recal = [r for r in noise_rows if r["threshold_source"].startswith("yeniden")]
    get = lambda rows, m, c, k: next((float(r[k]) for r in rows if r["method"] == m and r["condition"] == c), None)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6), dpi=200)
    fig.patch.set_facecolor(PAPER)
    x = list(range(len(conds)))
    for ax, key, title in ((axes[0], "auc", "ROC AUC (sıralama gücü)"), (axes[1], "far", "Yanlış alarm oranı (test)")):
        style(ax)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=LINE, lw=0.8, alpha=0.7)
        ax.tick_params(axis="x", pad=12)
        ends = []  # (değer, metin, renk) — FAR panelinde yalnızca uç değerler etiketlenir
        for m, col, dy in (("iforest", GREY, -1), ("cnn_ae", ACCENT, 1)):
            ys = [get(fixed, m, c, key) for c in conds]
            ax.plot(x, ys, color=col, lw=3, marker="o", ms=10, zorder=3)
            if key == "auc":
                for xi, v in zip(x, ys):
                    ax.annotate(tr(v, 2), (xi, v), xytext=(0, 14 * dy), textcoords="offset points", ha="center",
                                va="bottom" if dy > 0 else "top", fontsize=16, color=INK if m == "cnn_ae" else SOFT)
            else:
                yr = [get(fixed, m, "özgün", key)] + [get(recal, m, c, key) for c in conds[1:]]
                ax.plot(x, yr, color=col, lw=2, ls=(0, (5, 4)), marker="o", ms=7, mfc=PAPER, zorder=2)
                ends += [(ys[-1], tr(ys[-1], 2), col), (yr[-1], tr(yr[-1], 2) + " (yeniden)", col)]
        if ends:
            lo_, hi_ = 0, max(0.3, max(e[0] for e in ends) + 0.12)
            gap, placed = (hi_ - lo_) * 0.075, []
            for v, t, col in sorted(ends):
                yv = max(v, placed[-1] + gap) if placed else v
                placed.append(yv)
                ax.annotate(t, (2, v), xytext=(2.1, yv), textcoords="data", va="center", ha="left", fontsize=15,
                            color=col if col == ACCENT else SOFT,
                            arrowprops=dict(arrowstyle="-", color=LINE, lw=1) if abs(yv - v) > 1e-9 else None)
        ax.set_xticks(x, conds, fontsize=20, color=INK)
        ax.set_xlim(-0.3, 2.3 if key == "auc" else 3.05)
        ax.set_title(title, fontsize=22, color=INK, loc="left", pad=14)
        if key == "auc":
            ax.set_ylim(0.4, 1.0)
            ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            ax.axhline(0.5, color=SOFT, lw=1, ls=(0, (4, 4)))
            tr_axis(ax, "y", 1)
        else:
            top = max(0.3, max(float(r["far"]) for r in noise_rows) + 0.12)
            ax.set_ylim(0, min(1.0, top))
            ax.axhline(0.05, color=SOFT, lw=1, ls=(0, (4, 4)))
            ax.text(-0.28, 0.047, "%5 hedef", fontsize=15, color=SOFT, ha="left", va="top")
            tr_axis(ax, "y", 1)
    h = [Line2D([], [], color=ACCENT, lw=3, marker="o", ms=9, label="CNN autoencoder"),
         Line2D([], [], color=GREY, lw=3, marker="o", ms=9, label="Isolation Forest"),
         Line2D([], [], color=SOFT, lw=2, label="sabit eşik (özgün kalibrasyon)"),
         Line2D([], [], color=SOFT, lw=2, ls=(0, (5, 4)), marker="o", ms=7, mfc=PAPER, label="yeniden kalibre eşik")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=16, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.8)
    fig.tight_layout(w_pad=3, rect=(0, 0.08, 1, 1))
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()
    figdir = os.path.join(args.results, "figures")
    os.makedirs(figdir, exist_ok=True)
    stats = json.load(open(os.path.join(args.results, "report_stats.json"), encoding="utf-8"))
    auc_dumbbell(stats, os.path.join(figdir, "sunum_auc.png"))
    noise_p = os.path.join(args.results, "noise_summary.csv")
    if os.path.exists(noise_p):
        rows = list(csv.DictReader(open(noise_p, encoding="utf-8")))
        noise_panels(rows, os.path.join(figdir, "sunum_gurultu.png"))
    print("Yazıldı:", figdir)


if __name__ == "__main__":
    main()
