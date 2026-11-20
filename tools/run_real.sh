#!/usr/bin/env bash
# Gerçek DCASE fan verisiyle tüm hattı uçtan uca çalıştırır.
# Arka planda çalıştırmaya uygundur; ilerleme logs/run_real.log, durum logs/run_real.status dosyasındadır.
#
#   bash tools/run_real.sh                    # varsayılan: python, 3 AE tohumu (42 43 44)
#   PY=/yol/python SEEDS="42" bash tools/run_real.sh
#
# Protokol: config.yaml'daki ayarlar test sonucuna bakılmadan sabitlendi. Son test bir kez çalıştırılır;
# sonradan ayar değişirse raporda belirtilmelidir (results/test_access_log.csv).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
SEEDS=${SEEDS:-"42 43 44"}
NJOBS=${NJOBS:--1}
CONFIG=${CONFIG:-configs/config.yaml}
SKIP_DOWNLOAD=${SKIP_DOWNLOAD:-0}
mkdir -p logs
LOG=logs/run_real.log
STATUS=logs/run_real.status

echo "running $(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS"
echo "##### başlangıç $(date '+%Y-%m-%d %H:%M:%S') | python: $($PY --version 2>&1) | config: $CONFIG | tohumlar: $SEEDS" >> "$LOG"

step() {
  local name=$1; shift
  echo "running: $name ($(date '+%H:%M:%S'))" > "$STATUS"
  echo "=== $(date '+%H:%M:%S') $name: $*" >> "$LOG"
  echo "  > $(date '+%H:%M:%S') $name"
  local t0=$SECONDS
  if ! "$@" >> "$LOG" 2>&1; then
    echo "failed: $name" > "$STATUS"
    echo "!!! $name başarısız" >> "$LOG"
    echo "  ! $name başarısız; son satırlar:"; tail -n 15 "$LOG"
    exit 1
  fi
  echo "--- $name bitti ($((SECONDS - t0)) s)" >> "$LOG"
  echo "    tamam ($((SECONDS - t0)) s)"
}

n_wav=$(find data/raw -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')
if [ "$SKIP_DOWNLOAD" = "1" ]; then
  echo "=== indirme atlandı (SKIP_DOWNLOAD=1)" >> "$LOG"
elif [ "${n_wav:-0}" -lt 5000 ]; then
  step indirme $PY scripts/download_data.py --config "$CONFIG"
else
  echo "=== veri zaten var ($n_wav WAV), indirme atlandı" >> "$LOG"
fi

step hazirlik      $PY scripts/prepare_data.py --config "$CONFIG"
step ozellik       $PY scripts/extract_features.py --config "$CONFIG" --n-jobs "$NJOBS"
step iforest       $PY scripts/train_baseline.py --config "$CONFIG"
step overfit_test  $PY scripts/train_autoencoder.py --config "$CONFIG" --overfit-test
for s in $SEEDS; do
  step "cnn_ae_seed$s" $PY scripts/train_autoencoder.py --config "$CONFIG" --seed "$s"
done
# shellcheck disable=SC2086
step kalibrasyon   $PY scripts/calibrate.py --config "$CONFIG" --seeds $SEEDS
# shellcheck disable=SC2086
step son_test      $PY scripts/evaluate.py --config "$CONFIG" --seeds $SEEDS --exploratory-aggregations \
                   --note "ilk son test; ayarlar config.yaml başlangıç değerleri, test sonucuna göre değişiklik yapılmadı"
# shellcheck disable=SC2086
step gurultu       $PY scripts/noise_experiment.py --config "$CONFIG" --seeds $SEEDS --recalibrate

echo "done $(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS"
echo "##### bitti $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG"
