# Sesten makine arıza tespiti — BMT448 dönem projesi

Fanın **sağlıklı** çalışırken çıkardığı sesi öğrenen ve alışılmadık sesleri işaretleyen sistem.
Veri: DCASE 2020 Task 2 development, **fan** alt kümesi. İki yöntem aynı bölünmede karşılaştırılır:

| Yöntem | Ne yapar |
|---|---|
| **Isolation Forest** (referans) | Her kaydın log mel matrisinden 128 boyutlu (ortalama + std) vektör → anomali skoru |
| **CNN autoencoder** (derin model) | 64×64 log mel pencerelerini yeniden oluşturur; yüksek yeniden oluşturma hatası = anomali şüphesi |

Araştırma sorusu: *CNN autoencoder referans yöntemi geçiyor mu ve ek gürültü başarısını ne kadar düşürüyor?*

> **Kapsam sınırı.** Sistem ses anomalisini saptar; arızanın nedenini (ör. rulman) veya kalan ömrü tahmin etmez.
> Anomali skoru kalibre edilmiş bir arıza olasılığı değildir.

## Sonuçlar (gerçek veri, son test 21.11.2026, git 65ccbac)

Son test bir kez çalıştırıldı (`results/test_access_log.csv`). CNN AE değerleri üç tohumun (42, 43, 44) ortalamasıdır;
köşeli parantez = %95 bootstrap güven aralığı. Ayrıntılar: `results/summary.md`, `results/noise_summary.md`, `docs/rapor.md`.

| Kimlik | Isolation Forest AUC | CNN AE AUC | DCASE resmî baseline AUC |
|---|---|---|---|
| id_00 | 0,532 | 0,507 | 0,540 |
| id_02 | 0,681 | 0,696 | 0,722 |
| id_04 | 0,492 | 0,499 | 0,622 |
| id_06 | 0,616 | **0,851** | 0,723 |
| **Ortalama** | **0,580** | **0,638** | 0,652 |

- CNN AE − IF farkı (ortalama AUC): **+0,058 [0,036; 0,081]**; neredeyse tamamı id_06'dan. id_00 ve id_04'te iki yöntem de şans düzeyinde.
- Eşik (normal kalibrasyon p95): testte yanlış alarm ortalama %6,1 (hedef %5), yakalama (recall) CNN AE'de %22,3, IF'de %6,7.
- Beyaz gürültü (+20/+10 dB) AUC'yi düşürmedi; sabit eşikte yanlış alarm ve yakalama azaldı (CNN AE +10 dB: %3,1 ve %16,5),
  eşik gürültülü normal kayıtlarla yeniden kalibre edilince %5,7 ve %21,0'a döndü.
- DCASE baseline makine türü başına tek model ve tüm eğitim verisiyle eğitilir; karşılaştırma yaklaşıktır (bkz. rapor 7.2).

Sunum grafikleri: `results/figures/sunum_auc.png`, `sunum_gurultu.png`, `sunum_hata.png` (`python tools/make_figures.py`).

---

## 1. Hızlı başlangıç

```bash
# 1) Ortam (Python 3.11–3.13; iki kişi aynı sürümleri kullansın)
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
# GPU varsa önce torch'u https://pytorch.org/get-started/locally/ komutuyla kurun
pip install -r requirements.txt

# 2) Kod doğru çalışıyor mu? (gerçek veri olmadan, ~2 dakika)
python -m pytest -q
python run_pipeline.py --config configs/synthetic.yaml --synthetic

# 3) Gerçek veri
python scripts/download_data.py                 # dev_data_fan.zip → data/raw/fan/{train,test}
python run_pipeline.py --machine-ids id_00      # önce tek kimlik prototipi
python run_pipeline.py                          # sonra tüm fan kimlikleri
python run_pipeline.py --seeds 42 43 44         # zaman varsa 3 tohum tekrarı

# 4) Demo
streamlit run app/app.py
```

**macOS'ta tek komut (gerçek çalıştırma böyle yapıldı):** proje klasöründe `bash tools/mac_run.sh`. Betik uv ile
Python 3.13'ü ve paketleri proje klasörüne kurar, veriyi indirir (1,4 GB), ardından `tools/run_real.sh` ile tüm hattı
(hazırlık → özellik → IF → aşırı öğrenme testi → 3 tohumla CNN AE → kalibrasyon → son test → gürültü) çalıştırır.
İlerleme `logs/run_real.log` dosyasındadır. Veri zaten varsa yalnızca hat için: `SKIP_DOWNLOAD=1 PY=.venv/bin/python bash tools/run_real.sh`.
Rapor sayıları için ardından: `python tools/collect_results.py`, `python tools/report_stats.py`, `python tools/make_figures.py`.

> ⚠️ Sentetik veri yalnızca kodun çalıştığını gösterir; o sayıları raporda **kullanmayın**.
> Sentetik çıktılar ayrı klasörlere yazılır (`*_synthetic`), gerçek sonuçlarla karışmaz.

Zenodo indirmesi engellenirse: tarayıcıdan https://zenodo.org/records/3678171 sayfasındaki
`dev_data_fan.zip` dosyasını indirip `python scripts/download_data.py --zip <dosya>` çalıştırın.

---

## 2. Hat (betikler sırayla)

Her betik `configs/config.yaml` ve `data/split.csv` dosyalarını kullanır; `--config` ile başka config verilebilir.

| # | Betik | Yol haritası | Ne üretir |
|---|---|---|---|
| 1 | `scripts/download_data.py` | Adım 2 | `data/raw/fan/train`, `data/raw/fan/test` |
| 2 | `scripts/prepare_data.py` | Adım 3–5 | `data/manifest.csv`, **`data/split.csv`**, `results/data_counts.csv`, örnek görseller |
| 3 | `scripts/extract_features.py` | Adım 6 | `data/processed/logmel_<hash>/…npy` önbelleği |
| 4 | `scripts/train_baseline.py` | Adım 7 | `checkpoints/iforest/fan/<id>/model.joblib` |
| 5 | `scripts/train_autoencoder.py --overfit-test` | Adım 9 | 20 kayıtta aşırı öğrenme testi (GEÇTİ/KALDI) |
| 6 | `scripts/train_autoencoder.py` | Adım 8–9 | `checkpoints/cnn_ae/seed42/fan/<id>/best.pt`, `scaler.npz`, kayıp grafiği |
| 7 | `scripts/calibrate.py` | Adım 10 | Her modelin klasöründe `threshold.json` |
| 8 | `scripts/evaluate.py` | Adım 11 | `results/summary.md`, `metrics_per_id.csv`, ROC, dağılım, confusion matrix, hata analizi |
| 9 | `scripts/noise_experiment.py --recalibrate` | Adım 12 | `results/noise_summary.md`, `noise_robustness.png` |
| 10 | `app/app.py` | Adım 13 | Streamlit demosu |

`run_pipeline.py` hepsini sırayla çalıştırır. Ortak seçenekler: `--machine-ids id_00 id_02`, `--seeds 42 43 44`.

---

## 3. Deney protokolü — kodun garanti ettikleri

- **Kayıt düzeyinde bölünme, pencerelemeden önce.** Resmî train normal → %70 eğitim / %15 doğrulama / %15 kalibrasyon
  (her kimlik kendi içinde, tohum 42). Resmî testin tamamı → son test. Pencereler kaydının bölümünü miras alır.
- **Kopya kontrolü.** Aynı ses içeriğine sahip dosyalar aynı gruba konur; resmî testteki bir kaydın train'deki kopyası dışlanır.
  `prepare_data.py` sızıntı kontrolü başarısızsa durur.
- **`split.csv` sabittir.** Yeniden çalıştırmada üzerine yazılmaz (`--force` gerekir). Bu dosyayı Git'e ekleyin.
- **Normalizasyon yalnızca eğitimden.** Mel bandı başına ortalama/std ve IF ölçekleyicisi yalnızca eğitim kayıtlarından öğrenilir;
  doğrulama, kalibrasyon, test ve demo aynı dönüşümü yükler. Genlik normalizasyonu yapılmaz.
- **Eşik yalnızca normal kalibrasyondan.** 95. persentil. Arızalı kayıt ve test kaydı eşik için kullanılmaz.
- **Model seçimi etiketsiz.** Erken durdurma normal doğrulama kaybıyla (kayıt içi ortalama → kayıtlar arası ortalama).
- **Tek ön işleme kodu.** Eğitim, test, gürültü deneyi ve demo `src/features.py` ve `src/detectors.py` üzerinden aynı fonksiyonları çağırır
  (bunu `tests/test_pipeline_e2e.py` doğrular). Her model klasörü kendi ön işleme ayarını `meta.json` içinde taşır.
- **Test erişim kaydı.** `evaluate.py` her çalışmayı `results/test_access_log.csv` dosyasına yazar ve önceki çalıştırmaları hatırlatır.
- **Deney defteri.** Her eğitim/kalibrasyon `results/runs.csv` dosyasına run_id, git commit, tohum, özellik ayarları, süre,
  en iyi epoch, doğrulama kaybı ve eşikle eklenir.

**Ölçütler:** ROC AUC (sürekli skorlardan), pAUC (`max_fpr=0.1`, sklearn standartlaştırılmış), precision, recall, F1,
yanlış alarm oranı = FP/(FP+TN). Kimlik başına + kimliklerin eşit ağırlıklı ortalaması. Kimlik başına ~100 normal test
kaydı olduğu için **%95 bootstrap güven aralıkları** da verilir. Tek sınıflı kimlikte AUC hesaplanmaz.

**Gürültü deneyi:** aynı test kayıtlarına sabit tohumla, dosya başına deterministik beyaz gürültü (+20 dB ve +10 dB SNR;
gürültü RMS = kayıt RMS × 0,1 / ×0,316). Model ve eşik sabit. `--recalibrate` ek satırı: aynı SNR'de gürültülü **normal**
kalibrasyon kayıtlarıyla yeni eşik — düşüşün ne kadarının sıralama gücünden, ne kadarının eşik kaymasından geldiğini ayırır.

---

## 4. Klasörler

```
configs/        config.yaml (gerçek veri), synthetic.yaml (hat testi)
src/            data.py, features.py, models.py, detectors.py, metrics.py, noise.py, utils.py
scripts/        hattın betikleri (yukarıdaki tablo)
app/            app.py — Streamlit demosu
tests/          pytest: bölünme/sızıntı, özellikler, model, ölçütler, gürültü, uçtan uca + demo
tools/          mac_run.sh ve run_real.sh (tek komutla gerçek çalıştırma), collect_results.py (sonuç özeti JSON),
                report_stats.py (tohum ortalamalı AUC ve eşleştirilmiş bootstrap farkı), make_figures.py (sunum grafikleri),
                make_synthetic_dataset.py (DCASE düzeninde küçük sentetik veri)
notebooks/      colab_run.ipynb — GPU'suz ekip için Colab'da çalıştırma
docs/           rapor.md — raporun Markdown kopyası (gerçek sonuçlarla)
data/           raw/ (Git'e girmez), processed/ (Git'e girmez), manifest.csv, split.csv (Git'te)
checkpoints/    model ağırlıkları, scaler, eşik (Git'e girmez — teslimde ayrı paylaşın)
results/        metrikler, grafikler, deney defteri
```

---

## 5. Ek deneyler (ablation) nasıl yapılır

Ayarı değiştirmek için config'in bir kopyasını alın ve **çıktı klasörlerini değiştirin**, böylece ana sonuçlar ezilmez:

```bash
cp configs/config.yaml configs/ablation_b32.yaml
# ablation_b32.yaml içinde: autoencoder.bottleneck: 32
#                           paths.checkpoints: checkpoints/ablation_b32
#                           paths.results: results/ablation_b32
python scripts/train_autoencoder.py --config configs/ablation_b32.yaml
python scripts/calibrate.py --config configs/ablation_b32.yaml --methods cnn_ae
```

Önerilen ablation'lar: darboğaz 32/128/256, pencere uzunluğu (`frames: 32/64/128`), mel sayısı.
**Ayar seçimini normal doğrulama kaybına göre yapın; test AUC'ye bakıp ayar seçmeyin.** Test ancak ayarlar donduktan sonra.
Özellik ayarı değişirse önbellek otomatik olarak yeni bir `logmel_<hash>` klasörüne yazılır.

Keşif amaçlı: `evaluate.py --exploratory-aggregations` AE için pencere ortalaması yerine *max* ve *top3* toplama kurallarını
ayrı bir tabloda gösterir. Bunlar testten sonra bakıldığı için model seçimi sayılmaz; raporda "keşif" diye belirtin.

---

## 6. Ekip için iş bölümü önerisi

| Hafta | Kişi A (veri + değerlendirme) | Kişi B (model + demo) |
|---|---|---|
| 1 | `download_data.py`, `prepare_data.py` çıktıları, 5 örnek dinleme | ortam, `pytest`, sentetik hat |
| 2 | `split.csv`'yi dondur ve Git'e ekle, `extract_features.py` | `src/models.py` incele, `--overfit-test` |
| 3 | `train_baseline.py` | `train_autoencoder.py --machine-ids id_00` |
| 4 | `calibrate.py`, kalibrasyon skor dağılımları | tüm kimlikler, kayıp eğrileri |
| 5 | **ayarları dondur** → `evaluate.py` (bir kez) | demoyu gerçek modellerle dene |
| 6 | `noise_experiment.py --recalibrate` | `results/error_analysis/` dosyalarını dinle |
| 7 | rapor ve kaynakça (`docs/rapor.md`) | demo arayüzü, sunum |
| 8 | sonuçlar ve sınırlılıklar | temiz makinede README'den yeniden kurulum provası |

Süre daralırsa: tek kimlikte uçtan uca sistem (IF + AE + doğru bölünme + kalibrasyon + son test + demo) asgari teslimdir.

---

## 7. Beklenen süreler (yaklaşık)

Gerçek veriyle aynı boyutta (bir kimlik: 900 eğitim + 450 test kaydı) sentetik veriyle **2 çekirdekli CPU**'da ölçüldü:

| Adım | Süre |
|---|---|
| Özellik çıkarma (1350 kayıt) | ~6 s |
| CNN AE eğitimi (~5.700 pencere, epoch başına ~4 s, erken durdurmayla 17 epoch) | ~75 s |
| Tüm hat (bir kimlik, gürültü deneyi dahil) | ~2,5 dk |

Gerçek çalıştırma (MacBook Pro, Apple Silicon, yalnızca CPU, 4 kimlik, 3 tohum): tüm hat **~24 dakika**
(hazırlık 23 s, özellik 4 s, IF 7 s, aşırı öğrenme testi 65 s, CNN AE tohum başına ~7 dk / kimlik başına 85–135 s,
son test 47 s, gürültü deneyi 49 s); veri indirme bağlantıya göre ayrıca. GPU şart değil.

---

## 8. Sorun giderme

| Belirti | Çözüm |
|---|---|
| `DCASE düzeninde WAV bulunamadı` | Zip `data/raw/fan/train` ve `data/raw/fan/test` oluşturacak şekilde açılmalı. |
| `split.csv zaten var` | Beklenen davranış. Bilerek yenilemek için `prepare_data.py --force` (testi değiştirir!). |
| Aşırı öğrenme testi KALDI | Öğrenme oranını, config'deki `frames`/`n_mels` değerlerini ve veri yolunu kontrol edin. |
| CUDA belleği yetmiyor | `autoencoder.batch_size: 16` |
| Demo "model bulunamadı" | Önce eğitim ve `calibrate.py`; demo yalnızca eşiği olan modelleri listeler. |
| Windows'ta `extract_features.py` takılıyor | `--n-jobs 1` |

---

## 9. Raporda iddia etmeyeceklerimiz

- Resmî DCASE yarışma protokolüyle birebir eşdeğer sonuç (bölünme ve ayarlar bize ait).
- Hiç görülmemiş makinelerde aynı başarı (her kimlik kendi normal sesiyle eğitildi).
- Beyaz gürültü deneyinin gerçek fabrika gürültüsüne genellenmesi.
- Anomali skorunun arıza olasılığı olması.

Oturum bilgisi olmadığı için bölünme kayıt düzeyindedir; bağımsız oturumları garanti etmez — raporda belirtin.

## 10. Kaynaklar

1. DCASE 2020 Task 2 development veri seti — https://zenodo.org/records/3678171 (lisans ve atıf şartlarını kontrol edin)
2. DCASE 2020 Task 2 görev açıklaması — https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds
3. Resmî autoencoder baseline — https://github.com/y-kawagu/dcase2020_task2_baseline
4. scikit-learn `roc_auc_score` (`max_fpr`) — https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
5. PyTorch kurulum — https://pytorch.org/get-started/locally/
