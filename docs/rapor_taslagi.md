# Normal Seslerden Öğrenen Makine Ses Anomali Tespiti: CNN Autoencoder ile Isolation Forest Karşılaştırması

*BMT448 Derin Öğrenme — Dönem Projesi*
*[Ad Soyad 1], [Ad Soyad 2] · [Tarih]*

> Bu taslağın yöntem ve deney düzeni bölümleri koddaki uygulamayı birebir anlatır. `[DOLDUR]` ile işaretli yerler
> gerçek sonuçlarla doldurulacak. Sayıları `results/summary.md`, `results/noise_summary.md` ve
> `results/data_counts.csv` dosyalarından alın. Sentetik veri sonuçlarını kullanmayın.

---

## Özet

Endüstriyel makinelerde arızalı çalışmaya ait ses kayıtları azdır ve çeşitlidir; bu nedenle yalnızca normal seslerle
eğitilen yöntemler pratikte önemlidir. Bu çalışmada DCASE 2020 Task 2 development veri setinin fan alt kümesinde,
log mel spektrogram pencerelerini yeniden oluşturan küçük bir evrişimli autoencoder (CNN AE) ile kayıt düzeyi
istatistiklere dayalı Isolation Forest referans yöntemi aynı veri bölünmesinde karşılaştırılmıştır. Karar eşiği
yalnızca normal kalibrasyon kayıtlarından belirlenmiş, sentetik beyaz gürültünün etkisi ayrıca ölçülmüştür.
CNN AE kimliklerin ortalamasında ROC AUC [DOLDUR] (Isolation Forest: [DOLDUR]) elde etmiştir. +10 dB SNR ek gürültüde
AUC [DOLDUR] değişirken sabit eşikte yanlış alarm oranı [DOLDUR] artmıştır. [Bir cümlelik sonuç: derin model referansı
geçti mi, geçmediyse bu ne anlama geliyor?]

---

## 1. Giriş

- Problem: makinenin durumunu sesinden izlemek; arıza sesleri nadir ve önceden bilinmiyor.
- Yaklaşım: normal sesin dağılımını öğrenip ondan sapmayı anomali olarak işaretlemek (yarı denetimli / tek sınıflı öğrenme).
- Katkılar:
  1. Sızıntısız, kayıt düzeyinde bölünmüş ve tekrarlanabilir bir değerlendirme hattı.
  2. CNN autoencoder ile basit bir referans yöntemin aynı koşullarda karşılaştırılması.
  3. Etiketsiz (yalnızca normal) kalibrasyonla eşik seçimi ve yanlış alarm oranının raporlanması.
  4. Ek gürültü altında sıralama gücü ile eşik kaymasının ayrı ayrı incelenmesi.
- Kapsam sınırı: arıza nedeni veya kalan ömür tahmini yapılmaz; skor bir arıza olasılığı değildir.

## 2. İlgili çalışmalar

- DCASE 2020 Task 2: makine seslerinde denetimsiz anomali tespiti görevi [2].
- Resmî baseline: ardışık log mel çerçevelerini birleştiren tam bağlantılı autoencoder, yeniden oluşturma hatası skoru [3].
- Isolation Forest: rastgele bölmelerle izolasyon derinliğine dayalı anomali skoru.
- [İsteğe bağlı: MIMII veri seti makalesi, evrişimli/IDNN yaklaşımları — okuduğunuz kaynakları ekleyin.]

## 3. Veri

- DCASE 2020 Task 2 development, fan: ~10 s, tek kanal, 16 kHz kayıtlar; `train` yalnızca normal, `test` normal + anomali [1].
- Kimlik başına kayıt sayıları (Tablo 1): [DOLDUR — `results/data_counts.csv`]

| Kimlik | Eğitim | Doğrulama | Kalibrasyon | Test normal | Test anomali |
|---|---|---|---|---|---|
| id_00 | | | | | |
| id_02 | | | | | |
| id_04 | | | | | |
| id_06 | | | | | |

- Veri kalitesi kontrolü: bozuk/sessiz/kısa dosya ve içerik kopyası sayısı: [DOLDUR — `prepare_data.py` çıktısı].
- Kayıtlar fabrika ortam gürültüsü içerir.

## 4. Yöntem

### 4.1 Ön işleme
Ses mono okunur, gerekirse 16 kHz'e yeniden örneklenir; genlik normalizasyonu yapılmaz (seviye farkı arızayla ilişkili
olabilir). Log mel spektrogram: `n_fft=1024` (64 ms Hann pencere), `hop=512` (32 ms), 64 mel bandı (0–8 kHz),
güç spektrumu, `power_to_db(ref=1, top_db=80)`. 10 s'lik kayıt 64×313 matris verir. Model girdisi 64 zaman adımlık
(~2 s) pencerelerdir, kayma 32 adım; son pencere doldurma yapılmadan son 64 adımdan alınır (kayıt başına 9 pencere).
Mel bandı başına ortalama ve standart sapma yalnızca eğitim kayıtlarından hesaplanıp tüm bölümlere uygulanır.

### 4.2 Referans yöntem: Isolation Forest
Her kayıt için log mel matrisinin zaman boyunca bant ortalamaları ve standart sapmaları birleştirilerek 128 boyutlu
vektör elde edilir. Vektörler eğitim kayıtlarında öğrenilen `StandardScaler` ile ölçeklenir; 200 ağaçlı Isolation Forest
(`random_state=42`) yalnızca normal eğitim vektörleriyle eğitilir. Anomali skoru `−score_samples` (yüksek = daha anormal).

### 4.3 CNN autoencoder
Kodlayıcı: üç `Conv2d` bloğu (1→16→32→64 kanal, çekirdek 3, adım 2, ReLU) → 64×8×8 → doğrusal katmanla 128 boyutlu
darboğaz. Çözücü: doğrusal katman + ReLU → 64×8×8 → üç `ConvTranspose2d` bloğu (64→32→16→1, çekirdek 4, adım 2);
son katmanda aktivasyon yoktur çünkü standardize girdi negatif ve pozitif değer taşır. Toplam 1.117.361 parametre.
Kayıp: girdi ile çıktı arasındaki MSE. Adam, öğrenme oranı 0,001, batch 32, en fazla 50 epoch; normal doğrulama kaybı
7 epoch iyileşmezse durdurulur ve en iyi ağırlıklar saklanır. Her makine kimliği için ayrı model eğitilir.
Eğitimden önce 20 kayıtlık aşırı öğrenme testiyle uygulama doğrulanmıştır.

### 4.4 Kayıt skoru ve karar eşiği
Bir kaydın skoru pencere yeniden oluşturma hatalarının ortalamasıdır; bu kural testten önce sabitlenmiştir.
Her yöntem ve kimlik için eşik, normal kalibrasyon kayıtlarının skorlarının 95. persentilidir; skor eşiği aşarsa
"anomali şüphesi" kararı verilir. Bu ayar kalibrasyon verisinde ~%5 yanlış alarm hedefler; testte korunması garanti değildir.

## 5. Deney düzeni

- **Bölünme:** resmî train normal kayıtları kimlik içinde %70/%15/%15 (eğitim/doğrulama/kalibrasyon), tohum 42;
  resmî testin tamamı son test. Bölünme pencerelemeden önce, kayıt düzeyinde yapılır; aynı içerikli kopyalar aynı
  bölümde tutulur. Oturum bilgisi bulunmadığından bölünme bağımsız kayıt oturumlarını garanti etmez.
- **Model seçimi:** etiketsiz; yalnızca normal doğrulama kaybı. Test AUC değerine bakılarak ayar değiştirilmemiştir.
  [Testten sonra bir değişiklik yaptıysanız burada açıkça belirtin — `results/test_access_log.csv`.]
- **Ölçütler:** ROC AUC, pAUC (`max_fpr=0.1`, scikit-learn standartlaştırılmış değeri [4]), precision, recall, F1 ve
  yanlış alarm oranı FP/(FP+TN); kimlik başına ve kimliklerin eşit ağırlıklı ortalaması. %95 güven aralıkları sınıf içi
  bootstrap (1000 tekrar) ile.
- **Gürültü deneyi:** aynı test kayıtlarına sabit tohumla +20 dB ve +10 dB SNR beyaz gürültü (gürültü RMS = kayıt RMS ×
  0,1 ve ×0,316); model ve eşik değiştirilmeden. Ek koşul: aynı SNR'de gürültülü normal kalibrasyon kayıtlarıyla yeniden
  hesaplanan eşik.
- **Tekrarlanabilirlik:** tüm ayarlar `configs/config.yaml`; her çalıştırma `results/runs.csv` deney defterine yazılır.
  [Tohum tekrarı yaptıysanız: CNN AE 3 tohumla (42, 43, 44) eğitilmiş, ortalama ± std verilmiştir.]
- **Donanım:** [DOLDUR — CPU/GPU, eğitim süreleri `results/runs.csv`].

## 6. Sonuçlar

### 6.1 Ana karşılaştırma
Tablo 2: Kimlik başına son test sonuçları [DOLDUR — `results/summary.md` "Kimlik başına" tablosu]

Tablo 3: Kimliklerin eşit ağırlıklı ortalaması [DOLDUR]

| Yöntem | AUC | pAUC | Precision | Recall | F1 | Yanlış alarm |
|---|---|---|---|---|---|---|
| Isolation Forest | | | | | | |
| CNN autoencoder | | | | | | |

Şekil 1: ROC eğrileri (`results/figures/roc_fan_<id>.png`). Şekil 2: skor dağılımları ve eşik (`dist_*.png`).

Yorum yazarken: AUC farkı güven aralıklarıyla örtüşüyor mu? Hangi kimlikte fark büyük/küçük? Testteki yanlış alarm
oranı kalibrasyondaki %5 hedefine ne kadar yakın?

### 6.2 Gürültü dayanıklılığı
Tablo 4 [DOLDUR — `results/noise_summary.md`], Şekil 3 (`results/figures/noise_robustness.png`).

Yorum yazarken ayırın: (a) **sıralama gücü** (AUC) ne kadar düştü; (b) sabit eşikte **yanlış alarm** neden arttı —
gürültü tüm skorları yukarı iter; (c) yeniden kalibrasyon yanlış alarmı ne ölçüde geri getirdi?

### 6.3 Hata analizi
`results/error_analysis/` dosyalarındaki en yüksek skorlu yanlış alarmları ve en düşük skorlu kaçırılan anomalileri
dinleyip spektrogramlarına bakın. [DOLDUR — örüntüler: ör. kısa süreli darbeler ortalamada kayboluyor mu? Belirli bir
kimlikte normal kayıtlar arası değişkenlik yüksek mi?]

## 7. Tartışma ve sınırlılıklar

- Derin model referansı [geçti / geçmedi]; olası nedenler: [DOLDUR].
- Düşük yeniden oluşturma kaybı iyi anomali tespiti anlamına gelmez; etiketsiz model seçimi bu riski taşır.
- Kalibrasyon ve test normal kayıt sayıları küçüktür; yanlış alarm oranı geniş güven aralığına sahiptir.
- Değerlendirme eğitimde normal sesi görülen fan kimlikleriyle sınırlıdır; görülmemiş makinelere genelleme ölçülmemiştir.
- Beyaz gürültü gerçek fabrika gürültüsünün yerini tutmaz.
- Protokol resmî yarışma protokolüyle birebir aynı değildir; sonuçlar yarışma sıralamasıyla doğrudan karşılaştırılmamalıdır.

## 8. Sonuç ve gelecek çalışmalar

[DOLDUR — 1 paragraf.] Gelecek: diğer makine türleri, gürültüyle veri artırma, kısa anomaliler için farklı toplama
kuralları, görülmemiş makine kimliklerine genelleme deneyi.

## Kaynakça

[1] DCASE 2020 Challenge Task 2 Development Dataset, Zenodo, 2020. https://zenodo.org/records/3678171
    [Yazar listesini ve önerilen atıf biçimini Zenodo sayfasından kopyalayın.]
[2] DCASE 2020 Task 2: Unsupervised Detection of Anomalous Sounds for Machine Condition Monitoring.
    https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds
[3] DCASE 2020 Task 2 baseline kodu. https://github.com/y-kawagu/dcase2020_task2_baseline
[4] scikit-learn, `roc_auc_score`. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
[5] Liu F. T., Ting K. M., Zhou Z.-H., "Isolation Forest", ICDM 2008.
[Veri seti sayfasında istenen atıfları (MIMII makalesi vb.) ekleyin.]

---

## Ek: Sunum akışı (10–12 dk)

1. Problem ve neden yalnızca normal seslerden öğreniyoruz (1 dk)
2. Veri ve sızıntısız bölünme — neden kayıt düzeyinde (1,5 dk)
3. Log mel ve pencereler; CNN autoencoder fikri (2 dk)
4. Eşik: etiketsiz kalibrasyon, %5 yanlış alarm hedefi (1 dk)
5. Sonuçlar: referansla karşılaştırma, güven aralıkları (2 dk)
6. Gürültü deneyi: AUC ve yanlış alarm ayrı ayrı (1,5 dk)
7. Canlı demo: bir normal + bir anomali dosyası (`streamlit run app/app.py`, "Test setinden örnek") (2 dk)
8. Sınırlılıklar ve sonuç (1 dk)
