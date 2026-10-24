"""Adım 13: Streamlit demosu.

    streamlit run app/app.py
    APP_CONFIG=configs/synthetic.yaml streamlit run app/app.py   # sentetik hat testi modelleriyle

Demo yeniden eğitim veya yeniden eşik hesabı YAPMAZ; eğitimde kaydedilen ön işleme ayarlarını,
ölçekleyiciyi, model ağırlıklarını ve kalibrasyon eşiğini yükler.
"""
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

from src.data import resolve_path
from src.detectors import METHOD_LABELS, available_detectors, load_detector
from src.features import AudioTooShortError, load_audio, window_starts
from src.utils import load_config

st.set_page_config(page_title="Sesten arıza tespiti", page_icon="🔊", layout="wide")

CONFIG_PATH = os.environ.get("APP_CONFIG", str(ROOT / "configs" / "config.yaml"))


@st.cache_resource
def get_cfg(path):
    return load_config(path)


@st.cache_resource
def get_detector(method, directory):
    return load_detector(method, Path(directory))


@st.cache_data
def test_examples(split_path):
    p = Path(split_path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    return df[df["split"] == "test"]


def frames_to_seconds(frames, fcfg):
    return frames * fcfg["hop_length"] / fcfg["sr"]


def plot_spectrogram(logmel, fcfg, highlight=None):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    dur = frames_to_seconds(logmel.shape[1], fcfg)
    im = ax.imshow(logmel, origin="lower", aspect="auto", extent=[0, dur, 0, fcfg["n_mels"]], cmap="magma")
    for i, (s, e) in enumerate(highlight or []):
        ax.axvspan(s, e, color="cyan", alpha=0.18)
        ax.axvline(s, color="cyan", lw=0.8)
        ax.axvline(e, color="cyan", lw=0.8)
        ax.text((s + e) / 2, fcfg["n_mels"] * 0.92, f"#{i + 1}", color="cyan", ha="center", fontsize=9,
                fontweight="bold")
    ax.set_xlabel("zaman (s)")
    ax.set_ylabel("mel bandı")
    fig.colorbar(im, ax=ax, label="dB", pad=0.01)
    fig.tight_layout()
    return fig


def plot_window_scores(centers, errs, threshold, top_idx):
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.plot(centers, errs, marker="o", ms=4, lw=1.2, label="pencere yeniden oluşturma hatası")
    ax.scatter(centers[top_idx], errs[top_idx], s=90, facecolors="none", edgecolors="crimson", lw=1.6,
               label="en yüksek 3 pencere", zorder=3)
    ax.axhline(threshold, ls="--", c="k", lw=1, label="kayıt eşiği (referans çizgi)")
    ax.set_xlabel("pencere merkezi (s)")
    ax.set_ylabel("MSE")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def main():
    cfg = get_cfg(CONFIG_PATH)
    st.title("🔊 Sesten makine arıza tespiti")
    st.caption("DCASE 2020 Task 2 · fan · yalnızca normal seslerle eğitilmiş modeller")

    dets = available_detectors(cfg)
    if not dets:
        st.error(f"Kalibre edilmiş model bulunamadı ({cfg['paths']['checkpoints']}). Önce eğitim ve "
                 "`scripts/calibrate.py` adımlarını çalıştırın.")
        st.stop()

    with st.sidebar:
        st.header("Model")
        methods = sorted({d["method"] for d in dets}, key=lambda m: m != "cnn_ae")
        method = st.selectbox("Yöntem", methods, format_func=lambda m: METHOD_LABELS[m])
        cands = [d for d in dets if d["method"] == method]
        machines = sorted({(d["machine_type"], d["machine_id"]) for d in cands})
        mt, mid = st.selectbox("Makine kimliği (eğitilmiş olanlar)", machines, format_func=lambda k: f"{k[0]} · {k[1]}")
        seeds = sorted({d["seed"] for d in cands if (d["machine_type"], d["machine_id"]) == (mt, mid)})
        seed = st.selectbox("Eğitim tohumu", seeds) if method == "cnn_ae" and len(seeds) > 1 else seeds[0]
        chosen = next(d for d in cands if (d["machine_type"], d["machine_id"], d["seed"]) == (mt, mid, seed))
        st.caption(f"Desteklenen: {', '.join(sorted({f'{a} {b}' for a, b in machines}))}. "
                   "Başka bir makinenin kaydına sonuç güvenilir değildir.")
        st.divider()
        st.header("Ses kaynağı")
        source = st.radio("Kaynak", ["WAV yükle", "Test setinden örnek"], label_visibility="collapsed")

    det = get_detector(method, str(chosen["dir"]))

    audio_bytes, name, true_label = None, None, None
    if source == "WAV yükle":
        up = st.file_uploader("WAV kaydı (yaklaşık 10 s, tek kanal, 16 kHz önerilir)", type=["wav"])
        if up is not None:
            audio_bytes, name = up.getvalue(), up.name
    else:
        ex = test_examples(cfg["paths"]["split"])
        ex = ex[(ex["machine_type"] == mt) & (ex["machine_id"] == mid)] if len(ex) else ex
        if not len(ex):
            st.info("Bu kimlik için test listesi bulunamadı.")
        else:
            c1, c2 = st.columns([1, 3])
            lab = c1.radio("Etiket", ["normal", "anomaly"], format_func=lambda x: "normal" if x == "normal" else "anomali")
            sub = ex[ex["label"] == lab]
            pick = c2.selectbox("Dosya", sub["path"].tolist(), format_func=lambda p: Path(p).name)
            if pick:
                p = resolve_path(pick)
                if p.exists():
                    audio_bytes, name, true_label = p.read_bytes(), p.name, lab
                else:
                    st.warning(f"Dosya bulunamadı: {p}")

    if audio_bytes is None:
        st.info("Bir kayıt seçin ve **Analiz et** düğmesine basın.")
        with st.expander("Bu sistem neyi yapar, neyi yapmaz?"):
            st.markdown(
                "- Makinenin **normal** sesini öğrenir; alışılmadık sesleri işaretler.\n"
                "- Skor, kalibrasyon kayıtlarından belirlenen eşikle karşılaştırılır (normal kayıtların "
                "yaklaşık %5'i eşiği aşacak şekilde).\n"
                "- Arızanın **nedenini** (ör. rulman) veya kalan ömrü tahmin etmez.\n"
                "- Anomali skoru kalibre edilmiş bir **arıza olasılığı değildir**.")
        return

    st.audio(audio_bytes, format="audio/wav")
    if not st.button("Analiz et", type="primary"):
        return

    try:
        info = sf.info(io.BytesIO(audio_bytes))
        y = load_audio(io.BytesIO(audio_bytes), det.fcfg["sr"])
    except Exception as e:
        st.error(f"Dosya okunamadı: geçerli bir WAV değil veya bozuk ({type(e).__name__}).")
        return
    if not np.isfinite(y).all() or np.sqrt(np.mean(y ** 2)) < cfg["data"]["silence_rms"]:
        st.error("Kayıt sessiz veya geçersiz değerler içeriyor; analiz edilemez.")
        return
    notes = []
    if info.samplerate != det.fcfg["sr"]:
        notes.append(f"örnekleme hızı {info.samplerate} Hz → {det.fcfg['sr']} Hz'e yeniden örneklendi")
    if info.channels > 1:
        notes.append(f"{info.channels} kanal → mono'ya indirildi")
    if not 8 <= info.duration <= 12:
        notes.append(f"süre {info.duration:.1f} s; modeller ~10 s kayıtlarla eğitildi")
    if notes:
        st.warning("Not: " + "; ".join(notes) + ".")

    logmel = det.logmel_from_waveform(y)
    try:
        window_starts(logmel.shape[1], det.fcfg["frames"], det.fcfg["window_hop"])
    except AudioTooShortError as e:
        st.error(f"Kayıt çok kısa: {e}")
        return

    score = float(det.score_logmels([logmel])[0])
    thr = det.threshold
    anomalous = score > thr

    c1, c2, c3 = st.columns(3)
    c1.metric("Anomali skoru", f"{score:.4f}")
    c2.metric("Eşik (kalibrasyon p95)", f"{thr:.4f}")
    c3.metric("Skor / eşik", f"{score / thr:.2f}×" if thr > 0 else "—")
    if anomalous:
        st.error("⚠️ **Anomali şüphesi** — skor eşiği aşıyor. Kaydı bir uzmanın dinlemesi önerilir.")
    else:
        st.success("✅ **Normal** — skor eşiğin altında.")
    if true_label:
        st.caption(f"Test setindeki gerçek etiket: **{'anomali' if true_label == 'anomaly' else 'normal'}**")

    highlight, top_idx = [], None
    if method == "cnn_ae":
        errs, starts = det.window_errors(logmel)
        starts = np.array(starts)
        top_idx = np.argsort(errs)[-3:][::-1]
        highlight = [(frames_to_seconds(starts[i], det.fcfg),
                      frames_to_seconds(starts[i] + det.fcfg["frames"], det.fcfg)) for i in top_idx]

    st.subheader("Log mel spektrogram")
    st.pyplot(plot_spectrogram(logmel, det.fcfg, highlight), clear_figure=True)
    if method == "cnn_ae":
        st.subheader("Zaman içinde pencere skorları")
        centers = frames_to_seconds(starts + det.fcfg["frames"] / 2, det.fcfg)
        st.pyplot(plot_window_scores(centers, errs, thr, top_idx), clear_figure=True)
        st.caption("İşaretli bölgeler incelemek için önerilir; arızanın kesin yerini göstermez. "
                   "Pencereler ~2 s uzunluğunda ve %50 örtüşmelidir. Kayıt skoru pencere hatalarının ortalamasıdır.")
        st.dataframe(pd.DataFrame({
            "sıra": [1, 2, 3],
            "başlangıç (s)": [round(h[0], 2) for h in highlight],
            "bitiş (s)": [round(h[1], 2) for h in highlight],
            "pencere MSE": [round(float(errs[i]), 4) for i in top_idx]}), hide_index=True)
    else:
        st.caption("Isolation Forest kayıt düzeyinde tek skor üretir; zaman içi pencere skoru yalnızca autoencoder için gösterilir.")

    with st.expander("Model ayrıntıları"):
        st.json({"yöntem": METHOD_LABELS[method], "makine": f"{mt} {mid}", "tohum": seed,
                 "model klasörü": str(chosen["dir"]), "dosya": name,
                 "ön işleme": {k: det.fcfg[k] for k in ("sr", "n_fft", "hop_length", "n_mels", "frames", "window_hop")}})


main()
