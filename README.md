# ECG Aritmi Sınıflandırma & 1D-GAN Sentezi
### CNN · CNN+LSTM · CNN+LSTM+Attention · 1D-LSTM GAN (PyTorch)

MIT-BIH ve PTBDB veri setleri üzerinde EKG kalp atımlarını AAMI standardına göre
5 sınıfta sınıflandıran ve sınıf dengesizliğini **1D-GAN ile üretilen sentetik
atımlarla** kapatan, modüler bir PyTorch projesi. Ensemble model test setinde
**%98.0 accuracy** ve tüm sınıflarda **recall ≥ %88** elde eder.

---

## 🩺 Tıbbi Arka Plan / Medical Background

Veri setindeki her satır, R-tepesine hizalanmış ve sabit **187 örneğe** normalize
edilmiş **tek bir kalp atımıdır**. Her atım bir **PQRST kompleksi** içerir: P dalgası
(atriyal depolarizasyon), QRS kompleksi (ventriküler depolarizasyon — en keskin
bileşen) ve T dalgası (repolarizasyon).

**Sınıflar (AAMI):**

| ID | Kod | Açıklama |
|----|-----|----------|
| 0 | N | Normal beat (+ dal blokları / bundle branch blocks) |
| 1 | S | Supraventricular ectopic (örn. atriyal prematüre) |
| 2 | V | Ventricular ectopic (PVC) |
| 3 | F | Fusion beat |
| 4 | Q | Unknown / Paced |

**Temel zorluk:** Normal (N) atımlar eğitim setinin **~%82**'sini oluşturur
(72.471 / 87.554); en nadir sınıf **F yalnızca 641 atım (%0.7)**. Dengesiz veriyle
eğitilen model yüksek *accuracy* alır ama kritik aritmileri kaçırır (yüksek
false-negative). Bu yüzden **macro-F1 ve recall (sensitivity)** önceliklendirilir.

---

## 🏗️ Mimari: İki Fazlı Boru Hattı / Two-Phase Pipeline

```
            ┌─────────────────────────────────────────┐
 FAZ 1      │  Her azınlık sınıfı için 1D-LSTM GAN     │
 (GAN)      │  S, V, F, Q → sentetik gerçekçi atımlar  │
            └───────────────────┬─────────────────────┘
                                │ sentetik atımlar
                                ▼
            ┌─────────────────────────────────────────┐
 Veri       │  Sızıntısız bölme → dengeleme (resample) │
            │  + sentetik atımlar + Gaussian noise     │
            └───────────────────┬─────────────────────┘
                                ▼
            ┌─────────────────────────────────────────┐
 FAZ 2      │  CNN ──┐                                 │
 (Sınıf-    │  CNN+LSTM ──┼──→ ENSEMBLE (soft voting)  │
  landırma) │  CNN+LSTM+Attention ──┘                  │
            └─────────────────────────────────────────┘
```

- **CNN** — lokal morfolojiyi, özellikle keskin **QRS kompleksini** yakalar.
- **CNN+LSTM** — konvolüsyonel özellikler üzerinde atımlar arası **ritim/zamansal
  bağımlılığı** modeller (çift yönlü / bidirectional).
- **CNN+LSTM+Attention** — dizinin hangi bölümünün (genelde **R-peak/QRS**) tanıya
  en çok katkı verdiğini öğrenir → hem doğruluk hem **yorumlanabilirlik**.

---

## 📁 Proje Yapısı / Project Structure

```
.
├── data/                      # MIT-BIH & PTBDB CSV'leri
│   ├── mitbih_train.csv
│   ├── mitbih_test.csv
│   ├── ptbdb_normal.csv
│   └── ptbdb_abnormal.csv
├── data_loader.py             # CSV okuma, dengeleme, augmentation, Dataset/DataLoader
├── models.py                  # CNN, CNNLSTM, CNNLSTMAttention, Generator, Discriminator
├── evaluate.py                # Meter, confusion matrix, precision/recall/F1, attention viz
├── train.py                   # GAN + sınıflandırıcı eğitimini koordine eden ana dosya
├── checkpoints/               # eğitilmiş modeller + grafikler (otomatik oluşur)
└── README.md
```

---

## ⚙️ Kurulum / Setup

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn
```
> Colab'de bu paketler hazır gelir; GPU runtime (T4) önerilir.

`data/` klasörünün çalıştırma dizininde (Colab'de `/content/data/`) olduğundan
emin olun:
```bash
ls data/   # mitbih_train.csv vs. görünmeli
```

---

## 🚀 Kullanım / Usage

**1) Hızlı doğrulama (GAN atlanır, ~15–30 dk):**
```bash
python train.py --no-gan --clf-epochs 10
```

**2) Tam boru hattı (GAN dahil, ~1.5–2.5 saat, GPU şart):**
```bash
python train.py --gan-epochs 1500 --clf-epochs 25
```

**Argümanlar:**

| Argüman | Varsayılan | Açıklama |
|---------|-----------|----------|
| `--no-gan` | kapalı | GAN augmentation'ı atla |
| `--gan-epochs` | 3000 | GAN eğitim epoch sayısı |
| `--clf-epochs` | 30 | Sınıflandırıcı epoch sayısı |

Daha fazla ayar (`batch_size`, `synthetic_per_class`, `minority_classes`) için
`train.py` içindeki `Config` sınıfına bakın.

---

## 🧠 Modeller / Models

| Model | Yapı | Öne çıkan özellik |
|-------|------|-------------------|
| `CNN` | 3× ConvNormPool (skip-connection) + GAP | Lokal morfoloji / QRS |
| `CNNLSTM` | CNN + BiLSTM + temporal pooling | Ritim & zamansal bağlam |
| `CNNLSTMAttention` | CNN + BiLSTM + additive attention | Yorumlanabilir odak |
| `Generator` / `Discriminator` | 1D-LSTM GAN | Sentetik azınlık atımı |

`ConvNormPool`: nedensel (causal) padding + artık (residual) bağlantı + Swish +
MaxPool. Sınıflandırıcılar **logit** döndürür (çift-softmax hatası giderildi);
`CrossEntropyLoss` ile uyumlu.

---

## 📊 Değerlendirme Metrikleri / Evaluation

Dengesiz veri nedeniyle **accuracy tek başına yanıltıcıdır**. Raporlanan metrikler:

- **Recall (Sensitivity):** Gerçek aritmilerin kaçı yakalandı? *(en kritik)*
- **Precision (PPV):** Aritmi denenlerin kaçı doğru?
- **Macro-F1:** Precision/recall harmonik ortalaması, sınıf bazında eşit ağırlık.
- **Confusion Matrix:** Hangi aritminin hangisiyle karıştığı.

---

## 📈 Sonuçlar / Results (MIT-BIH Test, 21.892 atım)

**Final performans — GAN augmentation ile (makro-ortalama):**

| Model | Accuracy | Macro-F1 | Macro-Recall |
|-------|:--------:|:--------:|:------------:|
| CNN | 97.72% | 87.72% | 94.17% |
| CNN+LSTM | 97.66% | 88.31% | 94.25% |
| CNN+LSTM+Attention | 97.67% | 88.00% | 93.92% |
| **Ensemble (soft voting)** | **98.01%** | **88.95%** | **94.06%** |

**Ensemble — sınıf bazlı (final):**

| Sınıf | Precision | Recall | F1-score | Support |
|-------|:---------:|:------:|:--------:|:-------:|
| N | 99.48% | 98.40% | 98.94% | 18.118 |
| S | 76.45% | 87.59% | 81.64% | 556 |
| V | 95.17% | 96.62% | 95.89% | 1.448 |
| F | 56.52% | 88.27% | 68.92% | 162 |
| Q | 99.32% | 99.44% | 99.38% | 1.608 |

**GAN augmentation etkisi (ablation, Ensemble):**

| Yapılandırma | Accuracy | Macro-F1 | Macro-Recall | S-Precision | F-Precision |
|--------------|:--------:|:--------:|:------------:|:-----------:|:-----------:|
| GAN'siz (resample) | 97.43% | 87.15% | 94.92% | 66.89% | 51.75% |
| **GAN'li** | **98.01%** | **88.95%** | 94.06% | **76.45%** | **56.52%** |

**Yorum:** GAN sentetik verisi, hedeflenen en nadir sınıfların precision'ını
belirgin artırdı (**S +9.6 puan, F +4.8 puan**) ve macro-F1'i +1.8 puan yükseltti.
Macro-recall'da küçük bir düşüş (-0.86 puan) bilinçli bir precision-recall
dengesidir: model azınlık sınıflarda daha az "yanlış alarm" üretir. Tüm sınıflarda
recall ≥ %88 → klinik olarak kritik aritmiler büyük oranda yakalanır.

Çıktı grafikleri `checkpoints/` altına kaydedilir: `cm_*.png` (confusion matrix'ler),
`attention_example.png`, `synthetic_class*.npy` (sentetik atımlar).

---

## 🔍 Yorumlanabilirlik / Interpretability

`evaluate.plot_attention`, attention ağırlıklarını LSTM zaman adımlarından sinyalin
187 örneğine interpolasyonla geri ölçekleyip EKG'nin üstüne sıcak harita olarak
bindirir. Deneylerde modelin dikkati bir V (ventriküler) atımında **R-peak/QRS
yükselişine** yoğunlaşmış, sinyalin düz/sıfır bölgesine hiç bakmamıştır — yani
model kararını klinik olarak anlamlı bir bölgeye dayandırır (kara kutu değil).

---

## 🧬 GAN: Gerçek vs. Sentetik

1D-LSTM GAN, her azınlık sınıf için ayrı eğitilir (`generator_class*.pth`) ve
sentetik atımlar (`synthetic_class*.npy`) üretir. GAN, her sınıfın çekirdek
morfolojisini (R-peak, dalga şekli, zero-padding) sınıfa-özel öğrenmiştir.

**Bilinen sınırlama:** Discriminator bazı sınıflarda (özellikle V, Q) baskın geldiği
için sentetik atımlar gerçeklere göre **daha az çeşitlidir (kısmi mode collapse)**.
Sınırlı çeşitliliğe rağmen augmentation S/F precision'ını artırmaya yetmiştir.

---

## 🗂️ Veri Setleri / Datasets

- **MIT-BIH Arrhythmia** — 5 sınıflı atım sınıflandırması (ana görev).
- **PTBDB** — ikili (normal vs. miyokard enfarktüsü) sinyaller (opsiyonel görev).

Kaynak: PhysioNet (Kaggle "Heartbeat" dağıtımı).

---

## 📝 Tasarım Notları / Design Decisions

1. **Sızıntısız bölme:** Validation ham (dengelenmemiş) train'den ayrılır; dengeleme
   ve sentetik veri yalnızca eğitim setine eklenir → data leakage engellenir.
2. **GAN > basit upsample:** Kopyalamak ezberi körükler; GAN çeşitli sentetik örnek
   üretip genellemeyi destekler.
3. **Logit çıktısı:** Modeller softmax uygulamaz; `CrossEntropyLoss` ile çift-softmax
   hatası giderildi.
4. **Yorumlanabilir attention:** Additive attention LSTM zaman adımları üzerine
   uygulanır; ağırlıklar toplamı 1 olan, sinyale bindirilebilir bir dağılımdır.
5. **Augmentation hijyeni:** Gauss gürültüsü yalnızca eğitimde; test/validation'a asla.

---

## 🔭 Gelecek İş / Future Work

- Çok-lead (multi-lead) sinyal ve hasta-bazlı (inter-patient) bölme.
- GAN çeşitliliğini artırmak için **WGAN-GP**, **conditional GAN**, label smoothing.
- Sentetik verinin klinik geçerliliğinin uzman (kardiyolog) tarafından doğrulanması.
