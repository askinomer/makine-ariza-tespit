# Hedef koşul kalibrasyonu

Eşik, dışarıda bırakılan (yeni) koşulun ilk 10 dakikalık normal kaydından (60 klip, 95. persentil) alındı; test: aynı koşulun kalan 40 dakikalık normal kaydı (240 klip) ve tüm arıza klipleri. Yeni model eğitilmedi.

## Tüm arızalar

| Yöntem | Koşul | Yakalama | Yanlış alarm | En kötü kat yanlış alarm | F1 |
|---|---|---|---|---|---|
| Isolation Forest (referans) | temiz | 0,682 | 0,162 | 0,608 | 0,714 |
| Isolation Forest (referans) | +20 dB | 0,676 | 0,131 | 0,504 | 0,718 |
| Isolation Forest (referans) | +10 dB | 0,778 | 0,112 | 0,392 | 0,850 |
| CNN autoencoder | temiz | 1,000 ± 0,000 | 0,246 ± 0,034 | 0,926 | 0,977 ± 0,003 |
| CNN autoencoder | +20 dB | 1,000 ± 0,000 | 0,182 ± 0,038 | 0,931 | 0,983 ± 0,004 |
| CNN autoencoder | +10 dB | 0,993 ± 0,012 | 0,128 ± 0,014 | 0,921 | 0,984 ± 0,006 |

## Temiz — arıza türü başına yakalama

| Yöntem | dış bilezik | iç bilezik | kafes kırığı | bileşik iç-dış |
|---|---|---|---|---|
| Isolation Forest (referans) | 0,674 | 0,694 | 0,535 | 0,826 |
| CNN autoencoder | 1,000 | 1,000 | 1,000 | 1,000 |

## Ayrım payı (temiz): en düşük arıza skoru / en yüksek normal skor, dışarıda bırakılan koşulda

1'den büyükse normal ve arızalı klipler arasında boşluk vardır; bu aralıktaki her eşik iki sınıfı kusursuz ayırır.

| Yöntem | dış bilezik | iç bilezik | kafes kırığı | bileşik iç-dış |
|---|---|---|---|---|
| Isolation Forest (referans) (medyan / en düşük) | 1,01 / 0,91 | 0,99 / 0,90 | 0,95 / 0,91 | 1,02 / 0,91 |
| CNN autoencoder (medyan / en düşük) | 1,92 / 1,35 | 1,81 / 1,38 | 1,25 / 1,07 | 2,55 / 1,84 |
