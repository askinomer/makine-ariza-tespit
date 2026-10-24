"""Ortak yardımcılar: config okuma, tohum, yol çözümleme, deney defteri."""
from __future__ import annotations

import csv
import json
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict:
    """config.yaml dosyasını okur ve göreli yolları proje köküne göre çözer."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(Path(path).resolve())
    for key, value in cfg["paths"].items():
        p = Path(value)
        cfg["paths"][key] = str(p if p.is_absolute() else PROJECT_ROOT / p)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "no-commit"
    except Exception:
        return "unknown"


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | os.PathLike) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def load_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


RUN_LOG_FIELDS = [
    "run_id", "timestamp", "git_commit", "method", "machine_type", "machine_id", "seed",
    "feature_settings", "model", "train_time_s", "best_epoch", "val_loss", "threshold", "notes",
]


def append_run_log(cfg: dict, row: dict) -> None:
    """Deney defterine (results/runs.csv) bir satır ekler."""
    path = Path(cfg["paths"]["results"]) / "runs.csv"
    ensure_dir(path.parent)
    new = not path.exists()
    row = {k: row.get(k, "") for k in RUN_LOG_FIELDS}
    row.setdefault("timestamp", "")
    if not row["timestamp"]:
        row["timestamp"] = datetime.now().isoformat(timespec="seconds")
    if not row["git_commit"]:
        row["git_commit"] = git_commit()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RUN_LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def feature_settings_str(cfg: dict) -> str:
    f = cfg["features"]
    return (f"sr={f['sr']},n_fft={f['n_fft']},hop={f['hop_length']},n_mels={f['n_mels']},"
            f"frames={f['frames']},whop={f['window_hop']}")


def make_run_id(method: str, machine_type: str, machine_id: str, seed: int) -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}_{method}_{machine_type}_{machine_id}_s{seed}"


def resolve_device(name: str = "auto"):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
