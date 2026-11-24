# Guilin akustik rulman veri seti — sonuçlar

Koşul-dışı bırakma (9 kat). CNN AE: 3 tohum (42, 43, 44) ortalaması ± tohumlar arası sd; IF: tohum 42. Eşik: eğitim koşullarının normal kalibrasyon kliplerinin 95. persentili.

Veri: 13500 klip (10 sn) — cage 2700, compound 2700, inner 2700, normal 2700, outer 2700

## Temiz test — arıza türü başına (9 kat ortalaması)

| Yöntem | Arıza | AUC | pAUC | En düşük kat AUC | Yakalama | Yanlış alarm | F1 |
|---|---|---|---|---|---|---|---|
| Isolation Forest (referans) | dış bilezik | 0,875 | 0,840 | 0,235 | 1,000 | 0,865 | 0,705 |
| Isolation Forest (referans) | iç bilezik | 0,868 | 0,850 | 0,380 | 1,000 | 0,865 | 0,705 |
| Isolation Forest (referans) | kafes kırığı | 0,838 | 0,736 | 0,388 | 0,987 | 0,865 | 0,699 |
| Isolation Forest (referans) | bileşik iç-dış | 0,945 | 0,918 | 0,553 | 1,000 | 0,865 | 0,705 |
| Isolation Forest (referans) | tüm arızalar | 0,881 | 0,836 | 0,528 | 0,997 | 0,865 | 0,902 |
| CNN autoencoder | dış bilezik | 1,000 ± 0,000 | 1,000 ± 0,000 | 1,000 | 1,000 ± 0,000 | 1,000 ± 0,000 | 0,667 ± 0,000 |
| CNN autoencoder | iç bilezik | 1,000 ± 0,000 | 1,000 ± 0,000 | 1,000 | 1,000 ± 0,000 | 1,000 ± 0,000 | 0,667 ± 0,000 |
| CNN autoencoder | kafes kırığı | 1,000 ± 0,000 | 1,000 ± 0,000 | 1,000 | 1,000 ± 0,000 | 1,000 ± 0,000 | 0,667 ± 0,000 |
| CNN autoencoder | bileşik iç-dış | 1,000 ± 0,000 | 1,000 ± 0,000 | 1,000 | 1,000 ± 0,000 | 1,000 ± 0,000 | 0,667 ± 0,000 |
| CNN autoencoder | tüm arızalar | 1,000 ± 0,000 | 1,000 ± 0,000 | 1,000 | 1,000 ± 0,000 | 1,000 ± 0,000 | 0,889 ± 0,000 |

## Ortalama AUC farkı (CNN AE − IF), eşleştirilmiş bootstrap

- IF: 0,881 [0,876; 0,886]
- CNN AE: 1,000 [1,000; 1,000]
- Fark: **0,119 [0,114; 0,124]** (%95 GA, 300 tekrar; 9 kat × 4 arıza türü ortalaması)

## Kat (dışarıda bırakılan koşul) başına — tüm arızalar

| Koşul | IF AUC | CNN AE AUC | IF yanlış alarm | CNN AE yanlış alarm | IF yakalama | CNN AE yakalama |
|---|---|---|---|---|---|---|
| 800 rpm, %0 yük | 0,652 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 |
| 800 rpm, %15 yük | 0,528 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 |
| 800 rpm, %30 yük | 0,893 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 |
| 1000 rpm, %0 yük | 0,904 | 1,000 | 0,637 | 1,000 | 0,971 | 1,000 |
| 1000 rpm, %15 yük | 0,999 | 1,000 | 0,203 | 1,000 | 1,000 | 1,000 |
| 1000 rpm, %30 yük | 0,965 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 |
| 1200 rpm, %0 yük | 0,993 | 1,000 | 0,977 | 1,000 | 1,000 | 1,000 |
| 1200 rpm, %15 yük | 1,000 | 1,000 | 0,967 | 1,000 | 1,000 | 1,000 |
| 1200 rpm, %30 yük | 0,998 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 |

## Aynı gün alt kümesi (2025-11-02: normal ve dış bilezik aynı gün kaydedildi)

| Yöntem | AUC (kat ortalaması) | Kat sayısı |
|---|---|---|
| CNN autoencoder | 1,000 | 6 |
| Isolation Forest (referans) | 1,000 | 6 |

## Gürültü dayanıklılığı (tüm arızalar)

| Yöntem | Koşul | Eşik | AUC | Yakalama | Yanlış alarm |
|---|---|---|---|---|---|
| Isolation Forest (referans) | temiz | özgün kalibrasyon | 0,881 | 0,997 | 0,865 |
| Isolation Forest (referans) | +20 dB | özgün kalibrasyon | 0,870 | 0,996 | 0,913 |
| Isolation Forest (referans) | +20 dB | gürültülü kalibrasyon | 0,870 | 0,981 | 0,814 |
| Isolation Forest (referans) | +10 dB | özgün kalibrasyon | 0,890 | 0,993 | 1,000 |
| Isolation Forest (referans) | +10 dB | gürültülü kalibrasyon | 0,890 | 0,918 | 0,624 |
| CNN autoencoder | temiz | özgün kalibrasyon | 1,000 | 1,000 | 1,000 |
| CNN autoencoder | +20 dB | özgün kalibrasyon | 1,000 | 1,000 | 1,000 |
| CNN autoencoder | +20 dB | gürültülü kalibrasyon | 1,000 | 1,000 | 0,852 |
| CNN autoencoder | +10 dB | özgün kalibrasyon | 1,000 | 1,000 | 1,000 |
| CNN autoencoder | +10 dB | gürültülü kalibrasyon | 1,000 | 0,901 | 0,191 |

Not: Normal ve arızalı kayıtlar aynı oturumda alınmamıştır; aynı gün alt kümesi bu etkiyi kısmen sınar.
