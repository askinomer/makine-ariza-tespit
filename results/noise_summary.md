# Gürültü dayanıklılığı (kimliklerin eşit ağırlıklı ortalaması)

Δ = özgün teste göre fark. Sabit eşik satırlarında model ve eşik değiştirilmedi.

| Yöntem | Eşik | Koşul | AUC | ΔAUC | Recall | ΔRecall | Yanlış alarm | ΔYanlış alarm |
|---|---|---|---|---|---|---|---|---|
| CNN autoencoder | sabit (özgün kalibrasyon) | özgün | 0.638 | +0.000 | 0.223 | +0.000 | 0.061 | +0.000 |
| CNN autoencoder | sabit (özgün kalibrasyon) | +20 dB | 0.647 | +0.009 | 0.210 | -0.013 | 0.056 | -0.005 |
| CNN autoencoder | sabit (özgün kalibrasyon) | +10 dB | 0.643 | +0.005 | 0.165 | -0.058 | 0.031 | -0.030 |
| CNN autoencoder | yeniden kalibre (gürültülü normal cal) | +20 dB | 0.647 | +0.009 | 0.225 | +0.003 | 0.060 | -0.001 |
| CNN autoencoder | yeniden kalibre (gürültülü normal cal) | +10 dB | 0.643 | +0.005 | 0.210 | -0.013 | 0.057 | -0.004 |
| Isolation Forest (referans) | sabit (özgün kalibrasyon) | özgün | 0.580 | +0.000 | 0.067 | +0.000 | 0.062 | +0.000 |
| Isolation Forest (referans) | sabit (özgün kalibrasyon) | +20 dB | 0.587 | +0.007 | 0.049 | -0.018 | 0.055 | -0.007 |
| Isolation Forest (referans) | sabit (özgün kalibrasyon) | +10 dB | 0.602 | +0.021 | 0.046 | -0.021 | 0.048 | -0.015 |
| Isolation Forest (referans) | yeniden kalibre (gürültülü normal cal) | +20 dB | 0.587 | +0.007 | 0.057 | -0.010 | 0.058 | -0.005 |
| Isolation Forest (referans) | yeniden kalibre (gürültülü normal cal) | +10 dB | 0.602 | +0.021 | 0.056 | -0.012 | 0.055 | -0.007 |
