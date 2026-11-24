"""Guilin sonuç grafikleri (scripts/guilin_experiment.py çıktısından).

    python tools/make_guilin_figures.py --results results_guilin

metrics_fold.csv varsa önce küçük bir figure_data.json üretir; yoksa doğrudan figure_data.json'dan çizer
(böylece grafikler yalnızca özet veriyle de yeniden üretilebilir). Çıktı: <results>/figures/guilin_*.png
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import ACCENT, GREY, INK, LINE, PAPER, SOFT, pct, style, tr, tr_axis  # noqa: E402

FAULTS = ["outer", "inner", "cage", "compound", "tümü"]
FAULT_TR = {"outer": "Dış bilezik", "inner": "İç bilezik", "cage": "Kafes kırığı", "compound": "Bileşik iç-dış",
            "tümü": "Tüm arızalar"}
COL = {"iforest": GREY, "cnn_ae": ACCENT}
LAB = {"iforest": "Isolation Forest", "cnn_ae": "CNN autoencoder (3 tohum ort.)"}


def build_figure_data(res: Path) -> dict:
    import pandas as pd
    M = pd.read_csv(res / "metrics_fold.csv", dtype={"noise": str})
    clean = M[M.noise == "temiz"]
    # kat başına değer: AE için tohum ortalaması
    per_fold = clean.groupby(["method", "fault", "fold"])[["auc", "far", "recall"]].mean().reset_index()
    d = {"faults": {}, "folds": {}, "noise": {}}
    for (m, fl), g in per_fold.groupby(["method", "fault"]):
        d["faults"].setdefault(m, {})[fl] = dict(mean=g.auc.mean(), min=g.auc.min(), max=g.auc.max())
    for m, g in per_fold[per_fold.fault == "tümü"].groupby("method"):
        d["folds"][m] = {r.fold: dict(auc=r.auc, far=r.far, recall=r.recall) for r in g.itertuples()}
    tum = M[M.fault == "tümü"].groupby(["method", "noise", "threshold"])[["auc", "far", "recall"]].mean().reset_index()
    for r in tum.itertuples():
        d["noise"].setdefault(r.method, {})[f"{r.noise}|{r.threshold}"] = dict(auc=r.auc, far=r.far, recall=r.recall)
    (res / "figure_data.json").write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    return d


def fig_auc(d, out):
    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=200)
    fig.patch.set_facecolor(PAPER)
    style(ax)
    ys = list(range(len(FAULTS)))[::-1]
    for y, fl in zip(ys, FAULTS):
        for m, off in (("iforest", -0.17), ("cnn_ae", 0.17)):
            v = d["faults"][m][fl]
            ax.plot([v["min"], v["max"]], [y + off] * 2, color=COL[m], lw=2.2, alpha=0.55, solid_capstyle="butt")
            ax.scatter([v["mean"]], [y + off], s=170, color=COL[m], zorder=3, edgecolor=PAPER, linewidth=1.5)
            ax.text(min(v["max"], 1.0) + 0.008, y + off, tr(v["mean"]), va="center", fontsize=16,
                    color=INK if m == "cnn_ae" else SOFT, fontweight="bold" if m == "cnn_ae" else "normal")
    ax.axvline(0.5, color=SOFT, lw=1.2, ls=(0, (4, 4)), zorder=0)
    ax.axhline(0.5, color=LINE, lw=1)
    ax.set_yticks(ys, [FAULT_TR[f] for f in FAULTS], fontsize=20, color=INK)
    lo = min(v["min"] for m in d["faults"] for v in d["faults"][m].values())
    ax.set_xlim(min(0.45, lo - 0.03), 1.07)
    tr_axis(ax, "x", 1)
    ax.set_xlabel("ROC AUC — nokta: 9 kat ortalaması, çizgi: en düşük–en yüksek kat", fontsize=17, color=SOFT, labelpad=12)
    h = [Line2D([], [], marker="o", ls="", color=COL[m], ms=13, label=LAB[m]) for m in ("cnn_ae", "iforest")]
    h.append(Line2D([], [], color=SOFT, lw=1.2, ls=(0, (4, 4)), label="şans (0,5)"))
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=15, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)


def cond_key(c):
    r, l = c.split("r_")
    return int(r), int(l)


def fig_folds(d, out):
    conds = sorted(d["folds"]["cnn_ae"], key=cond_key)
    labels = [f"{c.split('r_')[0]}\n%{c.split('r_')[1]}" for c in conds]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), dpi=200)
    fig.patch.set_facecolor(PAPER)
    x = list(range(len(conds)))
    for ax, key, title in ((axes[0], "auc", "ROC AUC (tüm arızalar)"), (axes[1], "far", "Yanlış alarm (görülmemiş koşulun normali)")):
        style(ax)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=LINE, lw=0.8, alpha=0.7)
        for m in ("iforest", "cnn_ae"):
            ys = [d["folds"][m][c][key] for c in conds]
            ax.plot(x, ys, color=COL[m], lw=2.5, marker="o", ms=9, zorder=3, label=LAB[m])
        if key == "far":
            ax.axhline(0.05, color=INK, lw=1.5, ls=(0, (4, 4)))
            ax.text(len(conds) - 1, 0.05, "hedef %5 ", va="bottom", ha="right", fontsize=13, color=INK)
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: pct(v, 0)))
        else:
            tr_axis(ax, "y", 2)
        ax.set_xticks(x, labels, fontsize=12, color=INK)
        ax.tick_params(axis="y", labelsize=13)
        ax.set_xlabel("dışarıda bırakılan koşul (rpm, yük)", fontsize=13, color=SOFT)
        ax.set_title(title, fontsize=16, color=INK, loc="left")
    axes[0].legend(loc="lower left", fontsize=12, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)


def fig_noise(d, out):
    conds = [("temiz", "temiz"), ("20", "+20 dB"), ("10", "+10 dB")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), dpi=200)
    fig.patch.set_facecolor(PAPER)
    x = list(range(len(conds)))
    for ax, key, title in ((axes[0], "auc", "ROC AUC"), (axes[1], "recall", "Yakalama (recall)"), (axes[2], "far", "Yanlış alarm")):
        style(ax)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=LINE, lw=0.8, alpha=0.7)
        for m in ("iforest", "cnn_ae"):
            n = d["noise"][m]
            ax.plot(x, [n[f"{c}|sabit"][key] for c, _ in conds], color=COL[m], lw=2.5, marker="o", ms=9)
            if key != "auc":
                ax.plot(x, [n["temiz|sabit" if c == "temiz" else f"{c}|yeniden"][key] for c, _ in conds],
                        color=COL[m], lw=2, ls=(0, (3, 2)), marker="s", ms=7, alpha=0.8)
        if key == "auc":
            tr_axis(ax, "y", 2)
        else:
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: pct(v, 0)))
        if key == "far":
            ax.axhline(0.05, color=INK, lw=1.2, ls=(0, (4, 4)))
        ax.set_xticks(x, [c[1] for c in conds], fontsize=14, color=INK)
        ax.tick_params(axis="y", labelsize=13)
        ax.set_title(title, fontsize=16, color=INK, loc="left")
    h = [Line2D([], [], color=COL[m], lw=2.5, marker="o", label=LAB[m]) for m in ("cnn_ae", "iforest")]
    h += [Line2D([], [], color=SOFT, lw=2.5, label="özgün eşik"),
          Line2D([], [], color=SOFT, lw=2, ls=(0, (3, 2)), marker="s", label="gürültülü normalle yeniden kalibre")]
    fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, fontsize=13, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_guilin")
    a = ap.parse_args()
    res = Path(a.results)
    d = build_figure_data(res) if (res / "metrics_fold.csv").exists() else json.loads((res / "figure_data.json").read_text())
    fd = res / "figures"
    fd.mkdir(parents=True, exist_ok=True)
    fig_auc(d, fd / "guilin_auc.png")
    fig_folds(d, fd / "guilin_kosul.png")
    fig_noise(d, fd / "guilin_gurultu.png")
    print("Grafikler:", ", ".join(str(p) for p in sorted(fd.glob("guilin_*.png"))))


if __name__ == "__main__":
    main()
