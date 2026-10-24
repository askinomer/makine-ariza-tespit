import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config  # noqa: E402


def _tone(seed, dur=10.0, sr=16000, f0=50.0, extra=None):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    y = 0.3 * np.sin(2 * np.pi * f0 * t) + 0.1 * rng.standard_normal(len(t))
    if extra:
        y += 0.3 * np.sin(2 * np.pi * extra * t)
    return (0.1 * y).astype(np.float32)


@pytest.fixture()
def mini_dataset(tmp_path):
    """DCASE düzeninde çok küçük bir veri seti ve ona işaret eden config."""
    raw = tmp_path / "raw" / "fan"
    (raw / "train").mkdir(parents=True)
    (raw / "test").mkdir(parents=True)
    sr = 16000
    for mid in ("id_00", "id_04"):
        for i in range(20):
            sf.write(raw / "train" / f"normal_{mid}_{i:08d}.wav", _tone(i + 100 * int(mid[-1])), sr)
        for i in range(5):
            sf.write(raw / "test" / f"normal_{mid}_{i:08d}.wav", _tone(1000 + i), sr)
            sf.write(raw / "test" / f"anomaly_{mid}_{i:08d}.wav", _tone(2000 + i, extra=3000), sr)
    # aynı içerikli kopya (train içinde) ve test kaydının train'deki kopyası
    (raw / "train" / "normal_id_00_00000050.wav").write_bytes((raw / "train" / "normal_id_00_00000000.wav").read_bytes())
    (raw / "train" / "normal_id_00_00000051.wav").write_bytes((raw / "test" / "normal_id_00_00000000.wav").read_bytes())
    sf.write(raw / "train" / "normal_id_00_00000052.wav", _tone(7, dur=1.0), sr)   # kısa
    (raw / "train" / "readme.txt").write_text("not audio")                         # yok sayılmalı

    cfg = load_config()
    cfg["paths"].update({
        "raw_dir": str(tmp_path / "raw"), "processed_dir": str(tmp_path / "processed"),
        "manifest": str(tmp_path / "manifest.csv"), "split": str(tmp_path / "split.csv"),
        "checkpoints": str(tmp_path / "ckpt"), "results": str(tmp_path / "results"),
    })
    return cfg
