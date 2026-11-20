"""Ortak grafik stili: projedeki bütün şekiller aynı renkleri ve çizgi kurallarını kullanır.

Renk kuralı: renk varlığı izler. Bir yöntemin rengi her şekilde aynıdır
(CNN autoencoder mavi, Isolation Forest turuncu); normal/anomali sınıfları ayrı iki renk taşır.
Yazılar veri rengine boyanmaz; kimliği yanındaki işaret ve gösterge (legend) taşır.
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
METHOD_COLORS = {"cnn_ae": "#2a78d6", "iforest": "#eb6834"}
CLASS_COLORS = {"normal": "#1baf7a", "anomaly": "#6250d6"}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def apply() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 150,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10.5,
        "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlepad": 8,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.markersize": 7, "lines.markeredgewidth": 1.5, "lines.markeredgecolor": SURFACE,
        "legend.frameon": False, "legend.labelcolor": INK2, "legend.fontsize": 8.5,
        "font.size": 9.5, "text.color": INK,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
        "image.cmap": "magma",
    })


def seq_cmap() -> LinearSegmentedColormap:
    """Tek tonlu (mavi) sıralı renk ölçeği: açık = az, koyu = çok."""
    return LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)


def text_on(value: float, vmax: float) -> str:
    """Sıralı renkli bir hücrenin içindeki yazı rengi: koyu hücrede beyaz, açıkta mürekkep."""
    return "#ffffff" if vmax and value / vmax > 0.55 else INK
