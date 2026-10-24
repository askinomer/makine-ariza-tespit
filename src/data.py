"""Veri envanteri (manifest) ve sızıntısız kayıt düzeyinde bölünme.

DCASE 2020 Task 2 development düzeni:
    <raw_dir>/fan/train/normal_id_00_00000000.wav
    <raw_dir>/fan/test/normal_id_00_00000000.wav
    <raw_dir>/fan/test/anomaly_id_00_00000000.wav

Dikkat: train ve test klasörlerinde AYNI dosya adı bulunabilir. Bu yüzden
bir kaydın kimliği her zaman (resmî bölüm + dosya adı) ikilisidir.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from .utils import PROJECT_ROOT

FNAME_RE = re.compile(r"^(?P<label>normal|anomaly)_(?P<mid>id_\d+)_(?P<num>\d+)\.wav$", re.IGNORECASE)
SPLITS = ("train", "val", "cal", "test")


def to_stored_path(p: Path) -> str:
    """Manifestte saklanacak yol: mümkünse proje köküne göre göreli (taşınabilirlik için)."""
    p = Path(p).resolve()
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def resolve_path(stored: str) -> Path:
    p = Path(stored)
    return p if p.is_absolute() else PROJECT_ROOT / p


def parse_filename(path: Path) -> dict | None:
    """Dosya adından etiket ve makine kimliğini; klasörden resmî bölüm ve makine türünü okur."""
    m = FNAME_RE.match(path.name)
    if not m:
        return None
    official = path.parent.name.lower()
    if official not in ("train", "test"):
        return None
    return {
        "machine_type": path.parent.parent.name.lower(),
        "machine_id": m.group("mid").lower(),
        "label": m.group("label").lower(),
        "official_split": official,
        "source_recording_id": f"{official}/{path.stem}",
    }


def discover_files(raw_dir: str | os.PathLike, machine_types=None) -> list[Path]:
    raw = Path(raw_dir)
    files = sorted(p for p in raw.rglob("*.wav") if parse_filename(p) is not None)
    if machine_types:
        wanted = {m.lower() for m in machine_types}
        found = {parse_filename(p)["machine_type"] for p in files}
        kept = [p for p in files if parse_filename(p)["machine_type"] in wanted]
        if files and not kept:
            raise FileNotFoundError(
                f"WAV dosyaları bulundu ama makine türü klasörü {sorted(found)} olarak okundu; beklenen {sorted(wanted)}. "
                "Klasör düzeni <raw_dir>/<tür>/train/... ve <raw_dir>/<tür>/test/... olmalı (ör. data/raw/fan/train)."
            )
        files = kept
    return files


def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def audio_md5(y: np.ndarray) -> str:
    """Ses içeriğinin özeti: başlıkları farklı ama örnekleri aynı kopyaları da yakalar."""
    return hashlib.md5(np.ascontiguousarray(y, dtype=np.float32).tobytes()).hexdigest()


def inspect_file(path: Path, cfg: dict) -> dict:
    """Okunabilirlik, örnekleme hızı, süre, sessizlik ve içerik özeti kontrolü."""
    info = {"sample_rate": np.nan, "channels": np.nan, "duration": np.nan, "rms": np.nan,
            "content_md5": "", "usable": False, "issue": ""}
    try:
        y, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as e:  # bozuk dosya
        info["issue"] = f"unreadable: {type(e).__name__}"
        return info
    info["sample_rate"] = sr
    info["channels"] = y.shape[1]
    y = y.mean(axis=1)
    info["duration"] = len(y) / sr if sr else np.nan
    info["rms"] = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
    info["content_md5"] = audio_md5(y)
    issues = []
    if sr != cfg["data"]["expected_sr"]:
        issues.append(f"sr={sr} (yeniden örneklenecek)")  # kullanılabilir ama not edilir
    if info["duration"] < cfg["data"]["min_duration_s"]:
        issues.append("too_short")
    if info["rms"] < cfg["data"]["silence_rms"]:
        issues.append("silent")
    info["issue"] = ";".join(issues)
    info["usable"] = not any(i in ("too_short", "silent") for i in issues)
    return info


def build_manifest(cfg: dict, progress: bool = True) -> pd.DataFrame:
    files = discover_files(cfg["paths"]["raw_dir"], cfg["data"]["machine_types"])
    if not files:
        raise FileNotFoundError(
            f"{cfg['paths']['raw_dir']} altında DCASE düzeninde WAV bulunamadı. "
            "Önce scripts/download_data.py çalıştırın veya dev_data_fan.zip dosyasını data/raw içine açın."
        )
    rows = []
    for i, p in enumerate(files):
        meta = parse_filename(p)
        meta["path"] = to_stored_path(p)
        meta.update(inspect_file(p, cfg))
        rows.append(meta)
        if progress and (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(files)} dosya incelendi")
    df = pd.DataFrame(rows)
    wanted_ids = cfg["data"].get("machine_ids")
    if wanted_ids:
        df = df[df["machine_id"].isin([w.lower() for w in wanted_ids])].reset_index(drop=True)
    df["label_int"] = (df["label"] == "anomaly").astype(int)
    cols = ["path", "machine_type", "machine_id", "label", "label_int", "official_split",
            "sample_rate", "channels", "duration", "rms", "content_md5", "source_recording_id",
            "usable", "issue"]
    return df[cols]


def assign_splits(manifest: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Kayıt düzeyinde bölünme.

    * Resmî test → 'test' (tamamı, değişmez).
    * Resmî train normal → makine kimliği içinde train/val/cal (%70/%15/%15, tohum 42).
    * Aynı içerikli kopyalar aynı grupta tutulur, farklı bölümlere düşmez.
    * Resmî testteki bir kaydın kopyası train tarafında varsa o kopya dışlanır.
    """
    df = manifest.copy()
    df["group_id"] = df["content_md5"].where(df["content_md5"] != "", df["source_recording_id"])
    df["split"] = "excluded"
    ratios = cfg["data"]["split_ratios"]
    seed = cfg["seed"]

    test_mask = (df["official_split"] == "test") & df["usable"]
    df.loc[test_mask, "split"] = "test"
    test_groups = set(df.loc[test_mask, "group_id"])

    for (mt, mid), sub in df[(df["official_split"] == "train") & df["usable"]
                             & (df["label"] == "normal")].groupby(["machine_type", "machine_id"]):
        leaked = sub["group_id"].isin(test_groups)
        df.loc[sub.index[leaked], "split"] = "excluded_dup_of_test"
        sub = sub[~leaked]
        groups = np.array(sorted(sub["group_id"].unique()))
        rng = np.random.default_rng(seed)
        rng.shuffle(groups)
        n = len(groups)
        n_train = int(round(ratios["train"] * n))
        n_val = int(round(ratios["val"] * n))
        assignment = {}
        for j, g in enumerate(groups):
            assignment[g] = "train" if j < n_train else ("val" if j < n_train + n_val else "cal")
        df.loc[sub.index, "split"] = sub["group_id"].map(assignment)

    return df


def check_no_leakage(split_df: pd.DataFrame) -> list[str]:
    """Sızıntı kontrolleri. Boş liste = sorun yok."""
    problems = []
    used = split_df[split_df["split"].isin(SPLITS)]
    # 1) Bir kayıt yalnızca bir bölümde
    dup = used.groupby("path")["split"].nunique()
    if (dup > 1).any():
        problems.append(f"{int((dup > 1).sum())} kayıt birden fazla bölümde")
    # 2) Aynı içerik farklı bölümlerde değil
    g = used.groupby("group_id")["split"].nunique()
    if (g > 1).any():
        problems.append(f"{int((g > 1).sum())} içerik grubu birden fazla bölümde")
    # 3) Eğitim/doğrulama/kalibrasyonda anomali yok
    bad = used[used["split"].isin(["train", "val", "cal"]) & (used["label"] == "anomaly")]
    if len(bad):
        problems.append(f"{len(bad)} anomali kaydı train/val/cal içinde")
    # 4) Test yalnızca resmî testten
    bad = used[(used["split"] == "test") & (used["official_split"] != "test")]
    if len(bad):
        problems.append(f"{len(bad)} resmî train kaydı testte")
    bad = used[used["split"].isin(["train", "val", "cal"]) & (used["official_split"] != "train")]
    if len(bad):
        problems.append(f"{len(bad)} resmî test kaydı eğitim tarafında")
    return problems


def load_split(cfg: dict) -> pd.DataFrame:
    path = Path(cfg["paths"]["split"])
    if not path.exists():
        raise FileNotFoundError(f"{path} yok. Önce scripts/prepare_data.py çalıştırın.")
    return pd.read_csv(path, dtype={"machine_id": str})


def machine_keys(split_df: pd.DataFrame) -> list[tuple[str, str]]:
    keys = split_df[split_df["split"].isin(SPLITS)][["machine_type", "machine_id"]].drop_duplicates()
    return sorted(map(tuple, keys.values.tolist()))


def subset(split_df: pd.DataFrame, machine_type: str, machine_id: str, split: str) -> pd.DataFrame:
    m = (split_df["machine_type"] == machine_type) & (split_df["machine_id"] == machine_id) \
        & (split_df["split"] == split)
    return split_df[m].reset_index(drop=True)
