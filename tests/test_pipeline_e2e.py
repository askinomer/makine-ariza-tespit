"""Uçtan uca: küçük veri setinde tüm betikler + Streamlit demosu (WAV yükleme ve hata durumları)."""
import io
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import yaml

from tests.conftest import ROOT, _tone


def _write_cfg(cfg, tmp_path):
    c = {k: v for k, v in cfg.items() if not k.startswith("_")}
    c["autoencoder"] = {**c["autoencoder"], "max_epochs": 2, "batch_size": 16}
    c["scoring"] = {**c["scoring"], "bootstrap": 20}
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(c, allow_unicode=True), encoding="utf-8")
    return p


def _run(script, cfg_path, *extra):
    r = subprocess.run([sys.executable, str(ROOT / script), "--config", str(cfg_path), *extra],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"{script} başarısız:\n{r.stdout}\n{r.stderr}"
    return r.stdout


@pytest.fixture()
def trained(mini_dataset, tmp_path):
    p = _write_cfg(mini_dataset, tmp_path)
    _run("scripts/prepare_data.py", p, "--no-figures")
    _run("scripts/extract_features.py", p, "--n-jobs", "1")
    _run("scripts/train_baseline.py", p)
    _run("scripts/train_autoencoder.py", p)
    _run("scripts/calibrate.py", p)
    return mini_dataset, p


def test_full_pipeline(trained, tmp_path):
    cfg, p = trained
    out = _run("scripts/prepare_data.py", p, "--no-figures")
    assert "zaten var" in out                       # split.csv korunur
    _run("scripts/evaluate.py", p)
    _run("scripts/noise_experiment.py", p, "--recalibrate")
    res = tmp_path / "results"
    per_id = pd.read_csv(res / "metrics_per_id.csv")
    assert set(per_id["method"]) == {"iforest", "cnn_ae"}
    assert set(per_id["machine_id"]) == {"id_00", "id_04"}
    assert per_id["auc"].between(0, 1).all() and "auc_ci_lo" in per_id
    assert (res / "summary.md").exists() and (res / "noise_summary.md").exists()
    assert len(pd.read_csv(res / "test_access_log.csv")) == 1
    noise = pd.read_csv(res / "noise_results.csv")
    assert set(noise["condition"]) == {"özgün", "+20 dB", "+10 dB"}
    # sabit eşik satırlarında eşik gerçekten değişmemiş
    fixed = noise[noise["threshold_source"].str.startswith("sabit")]
    assert fixed.groupby(["method", "machine_id"])["threshold"].nunique().eq(1).all()
    runs = pd.read_csv(res / "runs.csv")
    assert {"run_id", "git_commit", "seed", "threshold"} <= set(runs.columns)


def test_scores_consistent_between_cache_and_waveform(trained):
    """Önbellekten skor ile demodaki gibi WAV'dan skor aynı olmalı (eğitim/demo tutarlılığı)."""
    cfg, _ = trained
    from src.data import load_split, resolve_path, subset
    from src.detectors import detector_dir, load_detector
    from src.features import load_logmels

    df = subset(load_split(cfg), "fan", "id_00", "test").head(3)
    for method in ("iforest", "cnn_ae"):
        det = load_detector(method, detector_dir(cfg, method, "fan", "id_00"))
        a = det.score_logmels(load_logmels(cfg, df))
        b = det.score_logmels([det.logmel_from_file(resolve_path(r)) for r in df["path"]])
        np.testing.assert_allclose(a, b, rtol=1e-5)


def test_streamlit_app(trained):
    _, p = trained
    from streamlit.testing.v1 import AppTest

    os.environ["APP_CONFIG"] = str(p)
    try:
        def upload(name, data):
            at = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=120).run()
            assert not at.exception
            at.file_uploader[0].set_value((name, data, "audio/wav")).run()
            at.button[0].click().run()
            assert not at.exception, at.exception
            return at

        buf = io.BytesIO()
        sf.write(buf, _tone(5), 16000, format="WAV")
        at = upload("ok.wav", buf.getvalue())
        labels = [m.label for m in at.metric]
        assert "Anomali skoru" in labels
        assert len(at.success) + len(at.error) == 1

        buf = io.BytesIO()
        sf.write(buf, _tone(5, dur=1.0), 16000, format="WAV")
        at = upload("short.wav", buf.getvalue())
        assert any("kısa" in e.value for e in at.error)

        at = upload("bad.wav", b"RIFF....not a wav")
        assert any("okunamadı" in e.value for e in at.error)

        buf = io.BytesIO()
        sf.write(buf, np.stack([_tone(3, sr=44100)] * 2, axis=1), 44100, format="WAV")
        at = upload("stereo44k.wav", buf.getvalue())
        assert any("yeniden örneklendi" in w.value for w in at.warning)
    finally:
        os.environ.pop("APP_CONFIG", None)
