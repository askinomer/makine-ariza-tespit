"""BK Connect .bkc → WAV (float32, 32.768 Hz, birim Pa).

    python tools/bkc_to_wav.py nm1000r-0-32Khz.1.bkc --out data/guilin_wav
    python tools/bkc_to_wav.py *.bkc --channels 1 2 --clip-s 10 --out data/guilin_wav   # 10 sn'lik klipler

Kanal numaraları 0 tabanlıdır: 0 = CRY333 (dosyada Volt; --cry-sens ile Pa'ya çevrilir),
1 ve 2 = B&K 4966 (zaten Pa). Çıktı adı: <dosya>_ch<k>[_<klip no>].wav
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bkc import read_info, read_samples  # noqa: E402

CRY333_SENS_V_PER_PA = 0.02398  # veri seti sayfasındaki değer (23,98 mV/Pa); XML'de 1 V/V görünür


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--channels", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--clip-s", type=float, default=0, help="0 = tüm dosya tek WAV; >0 = bu uzunlukta klipler")
    ap.add_argument("--cry-sens", type=float, default=CRY333_SENS_V_PER_PA,
                    help="Kanal 0 (Volt) için V/Pa; 0 verirseniz Volt olarak bırakılır")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for fp in a.files:
        info = read_info(fp)
        sr = int(info.sample_rate)
        for ch in a.channels:
            x = read_samples(fp, channels=[ch])[0]
            if info.channels[ch].unit == "V" and a.cry_sens:
                x = x / a.cry_sens
            stem = Path(fp).stem
            if a.clip_s > 0:
                n = int(a.clip_s * sr)
                for k in range(len(x) // n):
                    sf.write(out / f"{stem}_ch{ch}_{k:03d}.wav", x[k * n:(k + 1) * n], sr, subtype="FLOAT")
            else:
                sf.write(out / f"{stem}_ch{ch}.wav", x, sr, subtype="FLOAT")
            print(f"{Path(fp).name} ch{ch}: {len(x) / sr:.0f} s, rms {np.sqrt(np.mean(x.astype(np.float64) ** 2)):.4g} Pa")


if __name__ == "__main__":
    main()
