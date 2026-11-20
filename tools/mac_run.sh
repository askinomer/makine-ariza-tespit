#!/usr/bin/env bash
# macOS'ta (veya Linux'ta) tek komutla her şeyi yapar:
#   1) uv ve Python 3.13'ü kurar, 2) paketleri .venv içine kurar,
#   3) DCASE fan verisini indirir (1,4 GB) ve açar, 4) tüm hattı çalıştırır (tools/run_real.sh).
#
#   cd ~/BMT448/fan-anomaly && bash tools/mac_run.sh
#
# Her şey proje klasörünün bir üstünde kalır (ör. ~/BMT448): .tools/ (uv, Python, önbellek),
# dev_data_fan.zip ve fan-anomaly/.venv. Sistem ayarlarına ve kabuk profillerine dokunulmaz.
# İş arka planda (nohup) çalışır: Terminal penceresini kapatsan da devam eder.
# Mac'in uyumaması için caffeinate kullanılır; kapağı kapatmayın veya şarja takılı tutun.
set -uo pipefail
cd "$(dirname "$0")/.."
PROJ=$(pwd)
BASE=$(cd .. && pwd)
mkdir -p logs

# --- Ön plan: işi arka planda başlat, ilerlemeyi göster ---
if [ -z "${MAC_RUN_CHILD:-}" ]; then
  if [ -f logs/mac_run.pid ] && kill -0 "$(cat logs/mac_run.pid)" 2>/dev/null; then
    echo "Zaten çalışan bir iş var (pid $(cat logs/mac_run.pid)). İlerleme:"
  else
    MAC_RUN_CHILD=1 nohup bash "$PROJ/tools/mac_run.sh" > logs/mac_run.log 2>&1 &
    echo $! > logs/mac_run.pid
    echo "Başladı (pid $!). Bu pencereyi kapatsan da arka planda devam eder."
    echo "İzlemeyi durdurmak için Ctrl+C (iş durmaz). İlerleme:"
  fi
  echo
  exec tail -n +1 -f logs/mac_run.log
fi

# --- Arka plan işi ---
TOOLS="$BASE/.tools"
export UV_CACHE_DIR="$TOOLS/cache"
export UV_PYTHON_INSTALL_DIR="$TOOLS/python"
ZIP="${DATA_ZIP:-$BASE/dev_data_fan.zip}"
URL="https://zenodo.org/records/3678171/files/dev_data_fan.zip?download=1"
mkdir -p "$TOOLS"

say()  { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
fail() { say "HATA: $*"; echo "failed: $*" > logs/run_real.status; say "Ayrıntı: $PROJ/logs/mac_run.log ve logs/run_real.log"; exit 1; }

if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -ims -w $$ &   # bu iş bitene kadar Mac uyumasın
fi

say "Başlıyor: $(uname -sm) | proje: $PROJ"

# 1) uv (yoksa proje klasörünün yanına kurulur; kabuk profilleri değişmez)
find_uv() {
  if command -v uv >/dev/null 2>&1; then command -v uv; return; fi
  for c in "$TOOLS/uv" "$TOOLS/bin/uv"; do [ -x "$c" ] && { echo "$c"; return; }; done
}
UV=$(find_uv)
if [ -z "$UV" ]; then
  say "1/4 uv kuruluyor ($TOOLS)"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$TOOLS" sh
  UV=$(find_uv)
fi
PIP_FALLBACK=0
if [ -n "$UV" ]; then
  say "uv: $("$UV" --version)"
else
  say "uv kurulamadı; sistemdeki python3 ile devam ediliyor"
  PIP_FALLBACK=1
fi

# 2) Python 3.13 + paketler
pkg_install() {  # $@: pip argümanları
  if [ "$PIP_FALLBACK" = 1 ]; then .venv/bin/python -m pip install -q "$@"
  else "$UV" pip install --python .venv/bin/python "$@"; fi
}
if [ ! -x .venv/bin/python ]; then
  say "2/4 Python 3.13 ve sanal ortam hazırlanıyor"
  if [ "$PIP_FALLBACK" = 1 ]; then
    python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))' || fail "python3 en az 3.11 olmalı (bulunan: $(python3 --version 2>&1))"
    python3 -m venv .venv && .venv/bin/python -m pip install -q --upgrade pip || fail "sanal ortam oluşturulamadı"
  else
    "$UV" venv --python 3.13 .venv || fail "sanal ortam oluşturulamadı"
  fi
fi
say "2/4 Paketler kuruluyor (ilk seferde birkaç dakika sürebilir)"
if ! pkg_install -r requirements.txt; then
  say "Sabit sürümler bu sisteme uymadı; uyumlu sürümlerle tekrar deneniyor"
  pkg_install torch librosa soundfile numpy scipy pandas scikit-learn matplotlib PyYAML joblib streamlit pytest \
    || fail "paketler kurulamadı"
fi
if [ "$PIP_FALLBACK" = 1 ]; then .venv/bin/python -m pip freeze > logs/installed_versions.txt 2>/dev/null
else "$UV" pip freeze --python .venv/bin/python > logs/installed_versions.txt 2>/dev/null; fi
.venv/bin/python -c "import torch, librosa, sklearn, streamlit; print('torch', torch.__version__, '| librosa', librosa.__version__, '| sklearn', sklearn.__version__)" \
  || fail "paketler yüklenemedi"

# 3) Veri
n_wav=$(find data/raw -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')
if [ "${n_wav:-0}" -lt 5000 ]; then
  if [ -f "$ZIP" ] && unzip -tqq "$ZIP" >/dev/null 2>&1; then
    say "3/4 Veri arşivi hazır: $ZIP"
  else
    say "3/4 DCASE fan verisi indiriliyor (1,4 GB) → $ZIP"
    curl -L --fail --retry 5 --retry-delay 5 -C - -o "$ZIP" "$URL" \
      || { rm -f "$ZIP"; curl -L --fail --retry 5 --retry-delay 5 -o "$ZIP" "$URL"; } \
      || fail "veri indirilemedi"
    unzip -tqq "$ZIP" >/dev/null 2>&1 || fail "indirilen arşiv bozuk"
  fi
  say "3/4 Veri açılıyor"
  .venv/bin/python scripts/download_data.py --zip "$ZIP" || fail "arşiv açılamadı"
else
  say "3/4 Veri zaten açılmış ($n_wav WAV)"
fi

# 4) Hat
say "4/4 Eğitim ve değerlendirme (yaklaşık 15-30 dk). Ayrıntılı günlük: logs/run_real.log"
SKIP_DOWNLOAD=1 PY=.venv/bin/python bash tools/run_real.sh || fail "hat yarıda kaldı"

say "BİTTİ. Özet:"
sed -n '/eşit ağırlıklı ortalaması/,/^$/p' results/summary.md 2>/dev/null | head -8
echo
echo "Claude'a 'bitti' yazabilirsin. Demo için:  .venv/bin/streamlit run app/app.py"
