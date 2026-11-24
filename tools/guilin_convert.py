"""Guilin çok koşullu akustik rulman veri seti: Mendeley'den indir → .bkc oku → 10 sn'lik WAV klipleri → .bkc'yi sil.

    python tools/guilin_convert.py --out /content/guilin_wav                       # 5 veri setinin hepsi
    python tools/guilin_convert.py --out /content/guilin_wav --labels normal outer  # yalnızca bazıları
    python tools/guilin_convert.py --out out --local-files a.bkc b.bkc             # indirmeden (test)

Çıktı: <out>/<label>/<label>_<rpm>r_<load>_f<part>_c<klip>.wav  (tek kanal, 16-bit PCM, 32.768 Hz)
       <out>/manifest_<label>.csv  (kaynak dosya, koşul, kayıt zamanı, tepe/rms, kırpılan örnek sayısı)

Genlik: kayıt başına normalizasyon YAPILMAZ. Bütün dosyalar aynı sabitle ölçeklenir: 16-bit tam ölçek = FULL_SCALE_PA.
Kaldığı yerden devam eder: dönüştürülen her kaynak için <out>/_done/<kaynak>.json yazılır.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bkc import read_info, read_samples  # noqa: E402

DATASETS = {  # etiket → Mendeley veri seti kimliği (hepsi sürüm 1, CC BY 4.0)
    "normal": "j99rf8rsrz",
    "outer": "g2kzkwnkgw",
    "inner": "w9d6g35kcy",
    "cage": "jvv7p4wfgv",
    "compound": "bbvhrygxjz",
}
PREFIX = {"nm": "normal", "Ods": "outer", "Ids": "inner", "lc": "cage", "IOds": "compound"}
FNAME_RE = re.compile(r"^(?P<prefix>nm|IOds|Ods|Ids|lc)(?P<rpm>\d+)r-(?P<load>\d+)%?-32Khz\.(?P<part>\d+)\.bkc$")
FULL_SCALE_PA = 1.0      # 16-bit tam ölçek (≈ 94 dB SPL tepe); gözlenen tepeler ~0,2 Pa
DEFAULT_CHANNEL = 1      # B&K 4966 (Pa). 0 = CRY333 (V), 2 = ikinci B&K
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; guilin-convert/1.0)"}  # varsayılan Python-urllib 403 alır
API = "https://data.mendeley.com/public-api/datasets/{id}/files?folder_id=root&version=1"
MANIFEST_FIELDS = ["path", "label", "rpm", "load", "part", "clip", "channel", "source_file", "start_utc",
                   "clip_start_s", "rms_pa", "peak_pa", "n_clipped"]


def parse_name(name: str) -> dict:
    m = FNAME_RE.match(name)
    if not m:
        raise ValueError(f"Beklenmeyen dosya adı: {name}")
    return {"label": PREFIX[m["prefix"]], "rpm": int(m["rpm"]), "load": int(m["load"]), "part": int(m["part"])}


def list_remote(label: str) -> list[dict]:
    with urllib.request.urlopen(urllib.request.Request(API.format(id=DATASETS[label]), headers=HEADERS), timeout=60) as r:
        files = json.load(r)
    return [{"name": f["filename"], "url": f["content_details"]["download_url"],
             "sha256": f["content_details"]["sha256_hash"], "size": f["size"]} for f in files]


def download(entry: dict, dest: Path, tries: int = 4) -> Path:
    for attempt in range(1, tries + 1):
        try:
            h = hashlib.sha256()
            with urllib.request.urlopen(urllib.request.Request(entry["url"], headers=HEADERS), timeout=120) as r, open(dest, "wb") as f:
                while chunk := r.read(1 << 22):
                    f.write(chunk)
                    h.update(chunk)
            if dest.stat().st_size != entry["size"] or h.hexdigest() != entry["sha256"]:
                raise IOError("boyut/sha256 uyuşmadı")
            return dest
        except Exception as e:  # ağ hatası → tekrar dene
            dest.unlink(missing_ok=True)
            if attempt == tries:
                raise
            print(f"  {entry['name']}: {e} — tekrar ({attempt}/{tries})", flush=True)
            time.sleep(5 * attempt)
    raise RuntimeError("ulaşılamaz")


def convert(bkc: Path, name: str, out: Path, channel: int, clip_s: float) -> list[dict]:
    meta = parse_name(name)
    info = read_info(bkc)
    if info.channels[channel].unit != "Pa":
        raise ValueError(f"Kanal {channel} Pa değil ({info.channels[channel].unit}); bu betik yalnızca B&K kanalları için.")
    sr = int(info.sample_rate)
    x = read_samples(bkc, channels=[channel])[0]
    n = int(round(clip_s * sr))
    d = out / meta["label"]
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in range(len(x) // n):
        seg = x[k * n:(k + 1) * n]
        q = seg / FULL_SCALE_PA
        n_clip = int((np.abs(q) >= 1.0).sum())
        fn = d / f"{meta['label']}_{meta['rpm']}r_{meta['load']}_f{meta['part']}_c{k:03d}.wav"
        sf.write(fn, np.clip(q, -1.0, 32767 / 32768), sr, subtype="PCM_16")
        rows.append({"path": f"{meta['label']}/{fn.name}", **meta, "clip": k, "channel": channel,
                     "source_file": name, "start_utc": info.start_utc, "clip_start_s": k * clip_s,
                     "rms_pa": float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))),
                     "peak_pa": float(np.abs(seg).max()), "n_clipped": n_clip})
    return rows


def process(entry: dict, out: Path, tmp: Path, channel: int, clip_s: float, local: Path | None = None) -> list[dict]:
    done = out / "_done" / f"{entry['name']}.json"
    if done.exists():
        return json.loads(done.read_text())
    src = local or download(entry, tmp / entry["name"])
    try:
        rows = convert(src, entry["name"], out, channel, clip_s)
    finally:
        if local is None:
            src.unlink(missing_ok=True)  # ham .bkc saklanmaz
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(rows))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default=None, help="İndirme klasörü (varsayılan <out>/_tmp)")
    ap.add_argument("--labels", nargs="*", default=list(DATASETS))
    ap.add_argument("--channel", type=int, default=DEFAULT_CHANNEL)
    ap.add_argument("--clip-s", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help=">0: etiket başına en fazla bu kadar kaynak dosya (deneme)")
    ap.add_argument("--local-files", nargs="*", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    tmp = Path(a.tmp) if a.tmp else out / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    if a.local_files:
        jobs = [({"name": Path(p).name}, Path(p)) for p in a.local_files]
    else:
        jobs = []
        for label in a.labels:
            files = sorted(list_remote(label), key=lambda e: e["name"])
            if len(files) != 45:
                print(f"UYARI: {label} için {len(files)} dosya listelendi (beklenen 45).")
            jobs += [(e, None) for e in (files[:a.limit] if a.limit else files)]
    print(f"{len(jobs)} kaynak dosya, kanal {a.channel}, {a.clip_s:g} sn'lik klipler → {out}", flush=True)

    by_label: dict[str, list[dict]] = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(process, e, out, tmp, a.channel, a.clip_s, loc): e["name"] for e, loc in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            rows = fut.result()
            by_label.setdefault(rows[0]["label"], []).extend(rows)
            print(f"[{i}/{len(jobs)}] {futs[fut]}: {len(rows)} klip, tepe {max(r['peak_pa'] for r in rows):.3f} Pa, "
                  f"kırpılan {sum(r['n_clipped'] for r in rows)} · {time.time() - t0:.0f} s", flush=True)

    for label, rows in by_label.items():
        rows.sort(key=lambda r: r["path"])
        with open(out / f"manifest_{label}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            w.writeheader()
            w.writerows(rows)
        clipped = sum(r["n_clipped"] for r in rows)
        print(f"{label}: {len(rows)} klip, {clipped} kırpılan örnek" + ("  ⚠️ FULL_SCALE_PA'yı artırın" if clipped else ""))


if __name__ == "__main__":
    main()
