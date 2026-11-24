"""Brüel & Kjær BK Connect `.bkc` dosyalarını okuyucu (Guilin çok koşullu akustik veri seti).

Biçim, örnek dosyaların ikili incelenmesinden çıkarıldı (resmî bir belirtim yok):

  [0:30]    üst bilgi: u32 magic 0x72E176FB (XML bu örneklerde hep 180. baytta başlar, Part0'a kadar sürer)
  [30:...]  30 baytlık kayıtlar: u16 ad uzunluğu (=6), 6 UTF-16 karakter (".Part0".."Part4"), u64 ofset, u64 boyut
  [180:]    UTF-8 (BOM'lu) XML: kanal adları, birimler, kalibrasyon hassasiyetleri
  .Part0    92 baytlık başlık (u64 @16 = kanal başına örnek sayısı, u32 @24 = blok boyutu 4096, u32 @28 = kanal sayısı)
            ardından float32 (little endian) örnekler; 4096 örneklik bloklar kanallar arasında döngüyle
            (ch0, ch1, ch2, ch0, ch1, ch2, ...). Değerler fiziksel birimde (Pa ya da V).

Doğrulama: blok adımı 3'te blok sınırlarında süreklilik var (sıçrama = dosya içi tipik fark), diğer adımlarda yok.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = 0x72E176FB
REC_START = 30
REC_SIZE = 30
XML_START = 180
PART0_HEADER = 92


@dataclass
class BkcChannel:
    name: str
    unit: str            # "Pa" ya da "V"
    sensitivity: float   # V/Pa (mikrofon) ya da V/V; birim zaten Pa ise veri dönüşüm gerektirmez
    transducer: str = ""


@dataclass
class BkcFile:
    path: Path
    sample_rate: float
    n_samples: int
    block: int
    n_channels: int
    part0_offset: int
    channels: list[BkcChannel] = field(default_factory=list)
    start_utc: str = ""


def _directory(raw: bytes) -> dict[str, tuple[int, int]]:
    if struct.unpack_from("<I", raw, 0)[0] != MAGIC:
        raise ValueError("BKC magic bulunamadı: bu bir BK Connect dosyası değil.")
    parts, pos = {}, REC_START
    while pos + REC_SIZE <= XML_START:
        n = struct.unpack_from("<H", raw, pos)[0]
        if n == 0 or n > 16:
            break
        name = raw[pos + 2:pos + 2 + 2 * n].decode("utf-16-le")
        off, size = struct.unpack_from("<QQ", raw, pos + 2 + 2 * n)
        parts[name] = (off, size)
        pos += 2 + 2 * n + 16
    return parts


def read_info(path) -> BkcFile:
    path = Path(path)
    with open(path, "rb") as f:
        head = f.read(XML_START + 64 * 1024)
        parts = _directory(head)
        if ".Part0" not in parts:
            raise ValueError(f"{path.name}: .Part0 dizinde yok ({list(parts)})")
        off0, size0 = parts[".Part0"]
        f.seek(off0)
        p0 = f.read(PART0_HEADER)
    xml_end = off0
    xml = head[XML_START:xml_end].decode("utf-8-sig", "replace")
    n_samples = struct.unpack_from("<Q", p0, 16)[0]
    block, n_ch = struct.unpack_from("<II", p0, 24)
    # XML'deki <SamplingPeriod>7056000</SamplingPeriod> birimi belgelenmemiş; hız veri setinin yayın
    # bilgisinden (32.768 Hz) alınır. Doğrulama: 19.660.800 örnek / 32.768 = tam 600 sn (10 dk'lık dosya).
    if re.search(r"<SamplingPeriod>(\d+)</SamplingPeriod>", xml) is None or n_samples != 19_660_800:
        raise ValueError(f"{path.name}: beklenmeyen üst bilgi (örnek sayısı {n_samples}).")
    sample_rate = 32768.0
    chans = []
    for m in re.finditer(r"<Signal>(.*?)</Signal>", xml, re.S):
        s = m.group(1)
        g = lambda t: (re.search(rf"<{t}[^>]*>(.*?)</{t}>", s, re.S) or [None, ""])[1].strip()
        q = g("PhysicalQuantity")
        unit = "Pa" if q.lower().startswith("sound") else "V"
        chans.append(BkcChannel(g("Name"), unit, float(g("CalibrationSensitivity") or 1.0), g("TransducerName")))
    st = re.search(r"<DateTimeUtc>(.*?)</DateTimeUtc>", xml)
    info = BkcFile(path, sample_rate, int(n_samples), int(block), int(n_ch), off0 + PART0_HEADER, chans,
                   st.group(1) if st else "")
    expected = n_samples * n_ch * 4 + PART0_HEADER
    if expected != size0:
        raise ValueError(f"{path.name}: Part0 boyutu {size0} beklenen {expected} ile uyuşmuyor.")
    return info


def read_samples(path, start: int = 0, count: int | None = None, channels=None) -> np.ndarray:
    """(kanal, örnek) float32 dizisi. `start/count` kanal başına örnek cinsindendir; bellek eşlemeli okur."""
    info = read_info(path)
    if count is None:
        count = info.n_samples - start
    B, C = info.block, info.n_channels
    if info.n_samples % B:
        raise ValueError("Örnek sayısı blok boyutunun katı değil; desteklenmiyor.")
    nb = info.n_samples // B
    mm = np.memmap(path, dtype="<f4", mode="r", offset=info.part0_offset, shape=(nb, C, B))
    b0, b1 = start // B, -(-(start + count) // B)
    sel = list(range(C)) if channels is None else list(channels)
    out = np.ascontiguousarray(mm[b0:b1][:, sel, :].transpose(1, 0, 2).reshape(len(sel), -1))
    s = start - b0 * B
    return out[:, s:s + count]
