"""DCASE 2020 Task 2 development veri setinden fan arşivini indirir ve data/raw içine açar.

Kullanım:
    python scripts/download_data.py                 # indir + aç
    python scripts/download_data.py --zip yol.zip   # elle indirdiğiniz arşivi aç

Kaynak: https://zenodo.org/records/3678171 (lisans ve atıf şartlarını bu sayfadan kontrol edin)
"""
import _bootstrap  # noqa: F401

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from src.utils import ensure_dir, load_config

URL = "https://zenodo.org/records/3678171/files/dev_data_fan.zip?download=1"


def download(url: str, out: Path) -> None:
    print(f"İndiriliyor: {url}\n → {out}")
    tmp = out.with_suffix(".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                sys.stdout.write(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB")
                sys.stdout.flush()
    print()
    tmp.rename(out)


def extract(zip_path: Path, raw_dir: Path) -> None:
    print(f"Açılıyor: {zip_path} → {raw_dir}")
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():  # zip-slip koruması
            target = (raw_dir / member).resolve()
            if not str(target).startswith(str(raw_dir.resolve())):
                raise RuntimeError(f"Şüpheli arşiv yolu: {member}")
        z.extractall(raw_dir)
    n = len(list(raw_dir.rglob("*.wav")))
    print(f"Tamam: {n} WAV dosyası bulundu.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--zip", default=None, help="Önceden indirilmiş dev_data_fan.zip")
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw_dir = ensure_dir(cfg["paths"]["raw_dir"])
    if args.zip:
        zip_path = Path(args.zip)
    else:
        zip_path = raw_dir / "dev_data_fan.zip"
        if not zip_path.exists():
            try:
                download(URL, zip_path)
            except Exception as e:
                print(f"\nİndirme başarısız: {e}\n"
                      "Tarayıcıdan https://zenodo.org/records/3678171 sayfasındaki dev_data_fan.zip "
                      "dosyasını indirip şu komutla açın:\n"
                      "  python scripts/download_data.py --zip <indirilen_dosya>")
                sys.exit(1)
    extract(zip_path, raw_dir)
    if not args.zip and not args.keep_zip:
        zip_path.unlink(missing_ok=True)
    if shutil.disk_usage(raw_dir).free < 1e9:
        print("Uyarı: diskte 1 GB'tan az boş alan kaldı.")


if __name__ == "__main__":
    main()
