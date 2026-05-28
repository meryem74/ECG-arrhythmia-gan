"""
models.py
================================================================
PyTorch mimarileri / PyTorch architectures.

İçerik:
  Sınıflandırıcılar (classifiers):
    - Swish, ConvNormPool  -> ortak yapı taşları
    - CNN                  -> saf konvolüsyonel
    - CNNLSTM              -> CNN + LSTM
    - CNNLSTMAttention     -> CNN + LSTM + Attention (yorumlanabilir)
  Üretici model (generative):
    - Generator, Discriminator -> 1D-LSTM GAN

EKG sinyal işleme mantığı / Signal-processing intuition:
  Bir atım PQRST kompleksidir. Conv1d katmanları kayan pencereli
  "morfolojik filtreler" gibi davranır: küçük çekirdekler (kernel)
  QRS'in keskin yükseliş-düşüşü, P ve T dalgalarının daha yayvan
  şekilleri gibi LOKAL desenleri yakalar. LSTM bu lokal özelliklerin
  zaman içindeki DİZİLİMİNİ (ritim) modeller. Attention ise dizinin
  HANGİ bölümünün tanıya en çok katkı verdiğini öğrenir.
================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

SIGNAL_LENGTH = 187
NUM_CLASSES = 5


# ----------------------------------------------------------------------
# Aktivasyon / Activation
# ----------------------------------------------------------------------
class Swish(nn.Module):
    """
    Swish: x * sigmoid(x). ReLU'ya göre türevi her yerde pürüzsüz (smooth)
    ve negatif bölgede sıfırlamaz; EKG gibi ince genlik farklarının önemli
    olduğu sinyallerde küçük negatif aktivasyonları koruyarak gradyan
    akışını iyileştirir.
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


# ----------------------------------------------------------------------
# Temel blok: Skip-connection'lı konvolüsyon / residual conv block
# ----------------------------------------------------------------------
class ConvNormPool(nn.Module):
    """
    Üç Conv1d + normalizasyon + Swish + artık (residual/skip) bağlantı + MaxPool.

    Neden skip-connection (conv1 + conv3)?
    - Derinleştikçe gradyanın kaybolmasını (vanishing gradient) önler ve
      modelin "kimlik fonksiyonunu" kolayca öğrenmesine izin verir; QRS gibi
      keskin geçişlerin bilgisi katmanlar boyunca korunur.

    Neden SOLA padding (F.pad(kernel-1, 0))?
    - Nedensel (causal) hizalama sağlar: çıktının t anındaki değeri yalnızca
      geçmiş örneklere bağlı kalır, böylece dalgaların zaman ekseni üzerindeki
      konumu (P -> QRS -> T sırası) bozulmaz.

    MaxPool(2): diziyi yarıya indirir -> 187 -> 93 -> 46. Gürültüye karşı
    dayanıklılık + hesap maliyetinin düşmesi; baskın (en yüksek genlikli,
    yani genelde R-tepesi civarı) tepkiyi korur.
    """
    def __init__(self, input_size, hidden_size, kernel_size, norm_type="batch"):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_1 = nn.Conv1d(input_size, hidden_size, kernel_size)
        self.conv_2 = nn.Conv1d(hidden_size, hidden_size, kernel_size)
        self.conv_3 = nn.Conv1d(hidden_size, hidden_size, kernel_size)
        self.swish_1, self.swish_2, self.swish_3 = Swish(), Swish(), Swish()
        if norm_type == "group":
            self.n1 = nn.GroupNorm(8, hidden_size)
            self.n2 = nn.GroupNorm(8, hidden_size)
            self.n3 = nn.GroupNorm(8, hidden_size)
        else:
            self.n1 = nn.BatchNorm1d(hidden_size)
            self.n2 = nn.BatchNorm1d(hidden_size)
            self.n3 = nn.BatchNorm1d(hidden_size)
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        c1 = self.conv_1(x)
        h = self.swish_1(self.n1(c1))
        h = F.pad(h, (self.kernel_size - 1, 0))          # causal pad
        h = self.swish_2(self.n2(self.conv_2(h)))
        h = F.pad(h, (self.kernel_size - 1, 0))
        c3 = self.conv_3(h)
        h = self.swish_3(self.n3(c1 + c3))               # skip-connection
        h = F.pad(h, (self.kernel_size - 1, 0))
        return self.pool(h)


# ----------------------------------------------------------------------
# Model 1: Saf CNN / pure CNN
# ----------------------------------------------------------------------
class CNN(nn.Module):
    """
    3 adet ConvNormPool ile hiyerarşik özellik çıkarımı, ardından
    AdaptiveAvgPool ile küresel özet ve doğrusal sınıflandırıcı.

    Klinik yorum: CNN, atımın LOKAL morfolojisinde uzmandır. Özellikle
    QRS kompleksinin genişliği ve keskinliği gibi ayırt edici şekilleri
    yakalar -> örn. geniş/bizar QRS = ventriküler (V) atım işareti.

    NOT (kaynak notebook'a göre düzeltme): Burada softmax UYGULAMIYORUZ;
    ham logit döndürüyoruz. CrossEntropyLoss zaten içeride log-softmax
    yapar, dolayısıyla çift softmax (orijinal koddaki hata) gradyanı
    zayıflatır. Logit döndürmek doğru ve savunulabilir olandır.
    """
    def __init__(self, in_channels=1, hid_size=256, kernel_size=5, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = ConvNormPool(in_channels, hid_size, kernel_size)
        self.conv2 = ConvNormPool(hid_size, hid_size // 2, kernel_size)
        self.conv3 = ConvNormPool(hid_size // 2, hid_size // 4, kernel_size)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hid_size // 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)


# ----------------------------------------------------------------------
# Model 2: CNN + LSTM
# ----------------------------------------------------------------------
class CNNLSTM(nn.Module):
    """
    CNN özellik çıkarıcı -> Bidirectional LSTM -> sınıflandırıcı.

    Klinik yorum: CNN "ne" (hangi dalga şekli) sorusunu yanıtlar; LSTM ise
    bu şekillerin zaman içindeki DİZİLİMİNİ ve aralarındaki bağımlılığı
    (ritim) modeller. Çift yönlü (bidirectional) LSTM her zaman adımını
    hem öncesi hem sonrasıyla değerlendirir; bu, bir atımın bağlamını
    (örn. erken gelen bir vuru = prematüre) yakalamak için kritiktir.

    Önemli tasarım: konv. çıktısı (batch, kanal, zaman) iken LSTM
    (batch, zaman, özellik) bekler. Bu yüzden transpose(1,2) yapıyoruz:
    zaman ekseni = atım boyunca pozisyon, özellik = kanal. (Orijinal
    notebook'taki sabit '46' yerine boyutu dinamik/doğru ele alıyoruz.)
    """
    def __init__(self, in_channels=1, hid_size=128, lstm_hidden=128, num_layers=1,
                 bidirectional=True, num_classes=NUM_CLASSES, kernel_size=5):
        super().__init__()
        self.conv1 = ConvNormPool(in_channels, hid_size, kernel_size)
        self.conv2 = ConvNormPool(hid_size, hid_size, kernel_size)
        self.lstm = nn.LSTM(hid_size, lstm_hidden, num_layers,
                            bidirectional=bidirectional, batch_first=True)
        out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)            # (batch, kanal, zaman) -> (batch, zaman, kanal)
        out, _ = self.lstm(x)
        feat = out.mean(dim=1)           # zaman ekseni boyunca ortalama (temporal pooling)
        return self.fc(feat)


# ----------------------------------------------------------------------
# Attention bloğu / Additive (Bahdanau-tarzı) attention
# ----------------------------------------------------------------------
class Attention(nn.Module):
    """
    LSTM çıktısının HER zaman adımına bir ağırlık (skor) atar; ağırlıklar
    zaman ekseni boyunca softmax ile toplamı 1 olacak şekilde normalize edilir.

    Klinik yorum (JÜRİ İÇİN ANAHTAR NOKTA):
    - Bu ağırlıklar, modelin atımın HANGİ bölümüne "baktığını" gösterir.
    - Sağlıklı bir aritmi sınıflandırıcısında ağırlıkların büyük kısmı
      QRS kompleksi / R-tepesi civarında yoğunlaşmalıdır; çünkü tanısal
      bilginin çoğu oradadır. Bu, modeli yorumlanabilir (interpretable)
      kılar: ağırlıkları sinyalin üstüne bindirip "model QRS'e odaklandı"
      diye GÖSTEREBİLİRSİN.
    - context = ağırlıklı toplam: tanıya en çok katkı veren bölgeyi öne
      çıkaran tek bir özet vektör.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out):                 # (batch, zaman, hidden_dim)
        scores = self.v(torch.tanh(self.proj(lstm_out)))   # (batch, zaman, 1)
        weights = torch.softmax(scores, dim=1)             # zaman ekseninde normalize
        context = (weights * lstm_out).sum(dim=1)          # (batch, hidden_dim)
        return context, weights.squeeze(-1)                # ağırlıkları da döndür


# ----------------------------------------------------------------------
# Model 3: CNN + LSTM + Attention
# ----------------------------------------------------------------------
class CNNLSTMAttention(nn.Module):
    """
    CNN -> Bidirectional LSTM -> Attention -> sınıflandırıcı.
    En güçlü ve en YORUMLANABİLİR model. forward(x, return_attn=True) ile
    attention ağırlıklarını alıp görselleştirebilirsin (evaluate.py'de var).
    """
    def __init__(self, in_channels=1, hid_size=128, lstm_hidden=128, num_layers=1,
                 bidirectional=True, num_classes=NUM_CLASSES, kernel_size=5):
        super().__init__()
        self.conv1 = ConvNormPool(in_channels, hid_size, kernel_size)
        self.conv2 = ConvNormPool(hid_size, hid_size, kernel_size)
        self.lstm = nn.LSTM(hid_size, lstm_hidden, num_layers,
                            bidirectional=bidirectional, batch_first=True)
        out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.attn = Attention(out_dim)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x, return_attn=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        context, weights = self.attn(out)
        logits = self.fc(context)
        if return_attn:
            return logits, weights
        return logits


# ----------------------------------------------------------------------
# GAN: Generator + Discriminator (1D-LSTM)
# ----------------------------------------------------------------------
class Generator(nn.Module):
    """
    Rastgele gürültüden (latent z) sentetik bir EKG atımı üretir.
    Input z: (batch, 1, latent_dim) -> Output: (batch, 1, 187).

    Klinik amaç: Nadir aritmi sınıfları (S, V, F, Q) için GERÇEKÇİ sentetik
    atımlar üretmek. Basit kopyalamadan farkı, üretici dağılımı öğrenip
    ÇEŞİTLİ örnekler üretmesi -> sınıflandırıcının azınlık sınıfları
    ezberlemek yerine genelleştirmesini sağlar. LSTM, bir atım boyunca
    örnekler arası ardışık bağımlılığı modelleyerek pürüzsüz, fizyolojik
    olarak akla yatkın dalga formları üretir.
    """
    def __init__(self, signal_length=SIGNAL_LENGTH, latent_dim=SIGNAL_LENGTH, hidden=128):
        super().__init__()
        self.rnn = nn.LSTM(latent_dim, hidden, 1, bidirectional=True, batch_first=True)
        d = hidden * 2
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.fc3 = nn.Linear(d, signal_length)

    def forward(self, z):
        x, _ = self.rnn(z)
        x = x.reshape(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.2, training=self.training)
        x = self.fc3(x)
        return x.unsqueeze(1)            # (batch, 1, 187)


class Discriminator(nn.Module):
    """
    Bir atımın GERÇEK mi (1) yoksa SENTETİK mi (0) olduğunu ayırt eder.
    Input: (batch, 1, 187) -> Output: (batch, 1) olasılık [0,1].

    Discriminator iyi bir "kalp doktoru eleştirmen" gibi çalışır: Generator'ı
    fizyolojik olarak tutarlı (doğru QRS genişliği, P-T ilişkisi) atımlar
    üretmeye zorlar. sigmoid + BCELoss ile eğitilir (bkz. train.py).
    """
    def __init__(self, signal_length=SIGNAL_LENGTH, hidden=256):
        super().__init__()
        self.rnn = nn.LSTM(signal_length, hidden, 1, bidirectional=True, batch_first=True)
        d = hidden * 2
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d // 2)
        self.fc3 = nn.Linear(d // 2, 1)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x.reshape(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.2, training=self.training)
        return torch.sigmoid(self.fc3(x))
