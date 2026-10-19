"""Model usulan skenario S1–S4 (ABIDE I + II, DCAN).

Arsitektur: beberapa cabang yang masing-masing menghasilkan vektor berdimensi `hidden`, lalu digabung dengan
fusi atensi antar cabang.
  - cabang graf   : GATv2 (Brody dkk., 2022) pada graf otak per subjek + node-attention pooling sadar-ROI (notebook 12b)
  - cabang linear : model linear pada FC tangent multi-atlas (pola wide & deep, Cheng dkk., 2016)
  - cabang fenotipik (S3–S4): MLP kecil
Setiap cabang juga punya kepala klasifikasi bantu (deep supervision), sehingga setiap cabang belajar
memprediksi sendiri dan kontribusinya bisa dibaca terpisah.

Semua statistik (standardisasi, PCA, bobot kelas) dihitung HANYA dari data latih setiap fold.
"""
import copy
import gc
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import dropout_edge, scatter
from torch_geometric.utils import softmax as softmax_graf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================ fitur

def _segitiga(n):
    """Indeks segitiga bawah tanpa diagonal, urutan sama dengan nilearn sym_matrix_to_vec(discard_diagonal=True)."""
    return np.tril_indices(n, -1)


def _matriks(v, n):
    """(B, n(n-1)/2) -> (B, n, n) matriks simetris dengan diagonal 0."""
    r, c = _segitiga(n)
    M = np.zeros((len(v), n, n), np.float32)
    M[:, r, c] = v
    M[:, c, r] = v
    return M


def fitur_simpul_baris(V, n_total, n_simpul, idx_latih, kelompok=128):
    """Fitur simpul = baris matriks konektivitas.

    V        : (N, n_total(n_total-1)/2) vektor tangent satu atlas (boleh memmap)
    n_simpul : simpul graf = n_simpul ROI pertama (parcel korteks); setiap simpul tetap membawa
               konektivitasnya ke SEMUA n_total ROI (termasuk subkortikal)
    Keluaran : (N, n_simpul, n_total) float32, distandardisasi per (ROI, fitur) dengan statistik data latih.
    """
    X = np.empty((len(V), n_simpul, n_total), np.float32)
    for i in range(0, len(V), kelompok):
        X[i:i + kelompok] = _matriks(np.asarray(V[i:i + kelompok], np.float32), n_total)[:, :n_simpul]
    jumlah = np.zeros((n_simpul, n_total)); jumlah2 = np.zeros((n_simpul, n_total))
    for i in range(0, len(idx_latih), kelompok):
        B = X[idx_latih[i:i + kelompok]].astype(np.float64)
        jumlah += B.sum(0); jumlah2 += (B ** 2).sum(0)
    mu = jumlah / len(idx_latih)
    sd = np.sqrt(np.maximum(jumlah2 / len(idx_latih) - mu ** 2, 0)) + 1e-6
    X -= mu.astype(np.float32)
    X /= sd.astype(np.float32)
    return X


def fitur_linear_pca(daftar_V, idx_latih, kelompok=256, kolom=8192):
    """Fitur cabang linear: FC tangent setiap atlas, distandardisasi lalu diproyeksikan ke SELURUH komponen utama data latih.

    Model linear berpenalti L2 pada fitur asli (±154 ribu dimensi untuk 3 atlas) solusinya selalu berada di ruang
    yang direntang data latih (teorema representer). Karena itu proyeksi ke seluruh komponen utama data latih
    (±n_latih komponen per atlas) TIDAK membuang informasi yang bisa dipakai model linear, tetapi dimensinya turun
    drastis sehingga muat di GPU. PCA dihitung lewat matriks Gram (n_latih x n_latih).
    Skor dibagi akar jumlah fitur supaya skalanya ±1.
    Keluaran: (N, total komponen) float32.
    """
    hasil = []
    idx_latih = np.asarray(idx_latih)
    for V in daftar_V:
        N, p = V.shape
        jumlah, jumlah2 = np.zeros(p), np.zeros(p)
        for i in range(0, len(idx_latih), kelompok):
            B = np.asarray(V[idx_latih[i:i + kelompok]], np.float64)
            jumlah += B.sum(0); jumlah2 += (B ** 2).sum(0)
        mu = jumlah / len(idx_latih)
        sd = np.sqrt(np.maximum(jumlah2 / len(idx_latih) - mu ** 2, 0)) + 1e-6
        mu, sd = mu.astype(np.float32), sd.astype(np.float32)
        Xtr = (np.asarray(V[idx_latih], np.float32) - mu) / sd                  # (n_latih, p)
        K = np.zeros((len(Xtr), len(Xtr)))
        for j in range(0, p, kolom):
            A = Xtr[:, j:j + kolom].astype(np.float64)
            K += A @ A.T
        lam, U = np.linalg.eigh(K)
        pilih = lam > lam.max() * 1e-8
        proyeksi = U[:, pilih] / np.sqrt(lam[pilih])                           # (n_latih, n_komp)
        skor = np.empty((N, int(pilih.sum())), np.float32)
        for i in range(0, N, kelompok):
            B = (np.asarray(V[i:i + kelompok], np.float32) - mu) / sd
            skor[i:i + kelompok] = ((B @ Xtr.T).astype(np.float64) @ proyeksi / np.sqrt(p)).astype(np.float32)
        hasil.append(skor)
        del Xtr, K
    return np.hstack(hasil)


def sisi_topk(V, n_total, n_simpul, k, kelompok=64):
    """Sisi graf per subjek dari matriks konektivitas (vektor segitiga n_total ROI).

    Hanya n_simpul ROI pertama yang menjadi simpul. Setiap simpul dihubungkan ke k simpul lain dengan |nilai|
    terbesar, lalu dibuat simetris (graf tak berarah). Bobot sisi = nilai konektivitasnya.
    Keluaran: list (edge_index LongTensor (2, E), edge_attr FloatTensor (E, 1)) per subjek.
    """
    hasil = []
    for i in range(0, len(V), kelompok):
        M = _matriks(np.asarray(V[i:i + kelompok], np.float32), n_total)[:, :n_simpul, :n_simpul]
        A = np.abs(M)
        A[:, np.arange(n_simpul), np.arange(n_simpul)] = -1.0
        top = np.argpartition(-A, k, axis=2)[:, :, :k]                         # (B, n, k)
        for m, t in zip(M, top):
            pilih = np.zeros((n_simpul, n_simpul), bool)
            pilih[np.repeat(np.arange(n_simpul), k), t.ravel()] = True
            pilih |= pilih.T
            s, d = np.nonzero(pilih)
            hasil.append((torch.from_numpy(np.vstack([s, d])).long(), torch.from_numpy(m[s, d])[:, None]))
    return hasil


def buat_graf(y, X_simpul, sisi, X_linear=None, X_fenotipe=None):
    """Satu objek PyG Data per subjek. x_lin dan x_fen berbentuk (1, d) sehingga saat dibatch menjadi (B, d)."""
    daftar = []
    for i in range(len(y)):
        d = Data(x=torch.from_numpy(X_simpul[i]), edge_index=sisi[i][0], edge_attr=sisi[i][1], y=torch.tensor([int(y[i])]))
        if X_linear is not None:
            d.x_lin = torch.from_numpy(X_linear[i:i + 1])
        if X_fenotipe is not None:
            d.x_fen = torch.from_numpy(X_fenotipe[i:i + 1])
        daftar.append(d)
    return daftar


# ============================================================ model

class EncoderGraf(nn.Module):
    """GATv2 + node-attention pooling sadar-ROI.

    1. proyeksi fitur simpul + embedding identitas ROI (model tahu simpul mana wilayah otak mana)
    2. n_layer GATv2 multi-head, bobot sisi sebagai edge_dim, residual + LayerNorm
    3. readout dua jalur:
       - node-attention pooling: skor per simpul -> softmax di dalam satu graf (alpha) -> jumlah berbobot.
         alpha dibaca sebagai kepentingan setiap ROI bagi prediksi.
       - jalur identitas ROI: setiap simpul dipadatkan ke d_identitas nilai, disusun berurutan menurut ROI.
    Keluaran: (B, 2*hidden).
    """

    def __init__(self, n_roi, d_in, hidden=64, n_layer=2, heads=4, dropout=0.5, dropout_input=0.2, dropedge=0.2, d_identitas=4):
        super().__init__()
        self.n_roi, self.dropout, self.dropout_input, self.dropedge = n_roi, dropout, dropout_input, dropedge
        self.proj = nn.Linear(d_in, hidden)
        self.emb_roi = nn.Embedding(n_roi, hidden)
        self.convs = nn.ModuleList([GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=1, dropout=dropout) for _ in range(n_layer)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layer)])
        self.skor = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))
        self.padat = nn.Linear(hidden, d_identitas)
        self.identitas = nn.Linear(n_roi * d_identitas, hidden)
        self.dim_keluar = 2 * hidden
        self.alpha_terakhir = None

    def representasi_simpul(self, x, ei, ea):
        """Langkah 1-2 (proyeksi + embedding ROI + GATv2): representasi setiap simpul (jumlah simpul, hidden).
        Dipakai juga oleh pretraining (rekonstruksi fitur simpul yang disembunyikan)."""
        pos = torch.arange(x.size(0), device=x.device) % self.n_roi          # indeks ROI setiap simpul
        if self.training and self.dropedge > 0 and len(self.convs):
            ei, m = dropout_edge(ei, p=self.dropedge)
            ea = ea[m]
        h = F.elu(self.proj(F.dropout(x, self.dropout_input, self.training)) + self.emb_roi(pos))
        for conv, norm in zip(self.convs, self.norms):
            h = F.dropout(F.elu(norm(h + conv(h, ei, ea))), self.dropout, self.training)
        return h

    def forward(self, x, ei, ea, n_graf):
        b = torch.arange(x.size(0), device=x.device) // self.n_roi           # indeks graf setiap simpul
        h = self.representasi_simpul(x, ei, ea)
        alpha = softmax_graf(self.skor(h).squeeze(-1), b, num_nodes=n_graf)
        self.alpha_terakhir = alpha.detach().view(n_graf, self.n_roi)
        g_atensi = scatter(alpha[:, None] * h, b, 0, dim_size=n_graf, reduce="sum")
        g_identitas = F.elu(self.identitas(F.dropout(self.padat(h).view(n_graf, -1), self.dropout, self.training)))
        return torch.cat([g_atensi, g_identitas], dim=1)


class ModelUsulan(nn.Module):
    """Gabungan cabang graf / linear / fenotipik dengan FUSI ATENSI antar cabang.

    Setiap cabang menghasilkan vektor berdimensi `hidden`. Bobot setiap cabang (beta) dihitung per subjek dari
    vektornya (skor -> softmax antar cabang), lalu vektor digabung berbobot dan masuk MLP klasifikasi.
    Setiap cabang juga punya kepala klasifikasi bantu (deep supervision, bobot loss `bobot_bantu`).

    Catatan desain: penjumlahan logit antar cabang (Wide & Deep) juga sudah dicoba (notebook 16), tetapi logit
    cabang graf ±36x lebih besar dari cabang linear sehingga mendominasi dan akurasi turun ke ±67%. Karena itu
    dipakai fusi atensi ini (desain awal).

    d_simpul=0 mematikan cabang graf, d_linear=0 mematikan cabang linear, d_fenotipe=0 mematikan cabang fenotipik.
    Saat training forward mengembalikan (logits, loss_bantu); saat evaluasi hanya logits.
    """

    def __init__(self, n_roi, d_simpul=0, d_linear=0, d_fenotipe=0, hidden=64, dropout=0.5, dropout_linear=0.2,
                 bobot_bantu=0.5, **hiper_graf):
        super().__init__()
        self.bobot_bantu = bobot_bantu
        self.register_buffer("bobot_kelas", torch.ones(2))
        self.cabang = nn.ModuleDict()
        if d_simpul:
            enc = EncoderGraf(n_roi, d_simpul, hidden=hidden, dropout=dropout, **hiper_graf)
            self.cabang["graf"] = nn.ModuleDict({"enc": enc, "proj": nn.Linear(enc.dim_keluar, hidden)})
        if d_linear:
            # sengaja tanpa aktivasi: cabang ini (beserta kepala bantunya) adalah model linear murni
            self.cabang["linear"] = nn.ModuleDict({"enc": nn.Dropout(dropout_linear), "proj": nn.Linear(d_linear, hidden)})
        if d_fenotipe:
            self.cabang["fenotipik"] = nn.ModuleDict({"enc": nn.Sequential(nn.Linear(d_fenotipe, hidden), nn.ELU(), nn.Dropout(dropout)),
                                                      "proj": nn.Linear(hidden, hidden)})
        assert len(self.cabang), "minimal satu cabang harus aktif"
        self.nama_cabang = list(self.cabang)
        self.kepala_bantu = nn.ModuleDict({n: nn.Linear(hidden, 2) for n in self.nama_cabang})
        self.norm_fusi = nn.ModuleDict({n: nn.LayerNorm(hidden) for n in self.nama_cabang})
        self.skor_fusi = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))
        self.kepala = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2))
        self.beta_terakhir, self.logit_cabang = None, {}

    def _vektor_cabang(self, batch, n_graf):
        E = {}
        if "graf" in self.cabang:
            c = self.cabang["graf"]
            E["graf"] = F.elu(c["proj"](c["enc"](batch.x, batch.edge_index, batch.edge_attr, n_graf)))
        if "linear" in self.cabang:
            c = self.cabang["linear"]
            E["linear"] = c["proj"](c["enc"](batch.x_lin))
        if "fenotipik" in self.cabang:
            c = self.cabang["fenotipik"]
            E["fenotipik"] = F.elu(c["proj"](c["enc"](batch.x_fen)))
        return E

    def forward(self, batch):
        n_graf = batch.y.size(0)
        E = self._vektor_cabang(batch, n_graf)
        self.logit_cabang = {n: self.kepala_bantu[n](E[n]) for n in self.nama_cabang}
        tumpuk = torch.stack([self.norm_fusi[n](E[n]) for n in self.nama_cabang], dim=1)   # (B, M, hidden)
        beta = torch.softmax(self.skor_fusi(tumpuk).squeeze(-1), dim=1)                    # (B, M) bobot tiap cabang
        self.beta_terakhir = beta.detach()
        logits = self.kepala((beta[..., None] * tumpuk).sum(1))
        if not self.training:
            return logits
        y = batch.y.view(-1)
        bantu = torch.stack([F.cross_entropy(l, y, weight=self.bobot_kelas) for l in self.logit_cabang.values()]).mean()
        return logits, self.bobot_bantu * bantu

    def alpha(self):
        return self.cabang["graf"]["enc"].alpha_terakhir if "graf" in self.cabang else None


# ============================================================ pelatihan

CFG_DEFAULT = {"lr": 1e-3, "weight_decay": 1e-3, "batch_size": 32, "max_epochs": 200, "min_epochs": 20, "patience": 40,
               "val_size": 0.15, "seeds": (0, 1, 2, 3, 4), "batch_eval": 128,
               "pralatih_epoch": 0, "rasio_mask": 0.25, "pralatih_kolom_awal": 0}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def split_validasi(idx_latih, y, situs, seed, val_size):
    """Potongan validasi (untuk early stopping) diambil dari data latih, bertingkat situs x diagnosis."""
    strata = np.array([f"{s}_{l}" for s, l in zip(situs[idx_latih], y[idx_latih])])
    if np.unique(strata, return_counts=True)[1].min() < 2:
        strata = y[idx_latih]
    return train_test_split(idx_latih, test_size=val_size, stratify=strata, random_state=seed)


@torch.no_grad()
def prediksi(model, data, batch_eval=128):
    """Probabilitas ASD gabungan, per cabang (kepala bantu), bobot atensi ROI (alpha) dan bobot cabang (beta)."""
    model.eval()
    hasil = {"p": [], "beta": [], "alpha": [], **{f"p_{n}": [] for n in model.nama_cabang}}
    for b in DataLoader(data, batch_size=batch_eval):
        b = b.to(DEVICE)
        hasil["p"].append(F.softmax(model(b), 1)[:, 1].cpu())
        for n, l in model.logit_cabang.items():
            hasil[f"p_{n}"].append(F.softmax(l, 1)[:, 1].cpu())
        hasil["beta"].append(model.beta_terakhir.cpu())
        if model.alpha() is not None:
            hasil["alpha"].append(model.alpha().cpu())
    return {k: torch.cat(v).numpy() for k, v in hasil.items() if len(v)}


def latih_model(model, d_latih, d_val, cfg, lapor=None):
    """AdamW + cosine annealing; bobot dengan AUC validasi terbaik (setelah min_epochs) disimpan.
    lapor(epoch, auc_val), bila diberikan, dipanggil setiap epoch (dipakai Optuna untuk menghentikan trial yang buruk lebih awal)."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["max_epochs"])
    dl = DataLoader(d_latih, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    y_val = np.array([int(d.y) for d in d_val])
    terbaik, bobot, ep_terbaik, sabar = -1.0, copy.deepcopy(model.state_dict()), 0, 0
    riwayat = []
    for ep in range(1, cfg["max_epochs"] + 1):
        model.train()
        total = 0.0
        for b in dl:
            b = b.to(DEVICE)
            opt.zero_grad()
            logits, bantu = model(b)
            loss = F.cross_entropy(logits, b.y.view(-1), weight=model.bobot_kelas) + bantu
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += loss.item() * b.y.size(0)
        sched.step()
        auc = roc_auc_score(y_val, prediksi(model, d_val, cfg["batch_eval"])["p"])
        riwayat.append((ep, total / (len(dl) * cfg["batch_size"]), auc))
        if lapor is not None:
            lapor(ep, auc)
        if ep >= cfg["min_epochs"] and auc > terbaik + 1e-4:
            terbaik, bobot, ep_terbaik, sabar = auc, copy.deepcopy(model.state_dict()), ep, 0
        elif ep >= cfg["min_epochs"]:
            sabar += 1
            if sabar >= cfg["patience"]:
                break
    model.load_state_dict(bobot)
    return model, {"epoch_terbaik": ep_terbaik, "auc_val": terbaik, "riwayat": np.array(riwayat)}


def pralatih_graf(model, data, cfg):
    """Pretraining self-supervised encoder graf dengan rekonstruksi fitur simpul yang disembunyikan
    (masked feature reconstruction, gaya GraphMAE; Hou dkk., 2022).

    Setiap epoch, sebagian simpul (cfg["rasio_mask"]) dipilih acak; fitur simpul itu diganti token mask yang
    dipelajari, lalu encoder (proyeksi + embedding ROI + GATv2) harus merekonstruksi fitur aslinya dari
    tetangganya. Encoder jadi belajar struktur graf otak TANPA label diagnosis.
    cfg["pralatih_kolom_awal"] = j berarti yang direkonstruksi hanya kolom fitur j dan seterusnya (mis. hanya fitur
    struktur di S4 lama, kini diarsipkan); masukan tetap seluruh fitur (selain simpul yang disembunyikan). Yang dipakai hanya data latih fold
    tersebut (tanpa label), data uji tidak pernah dilihat. Bobot encoder hasil pretraining lalu dilanjutkan
    (fine-tuning) bersama seluruh model pada tugas klasifikasi.
    Keluaran: riwayat loss rekonstruksi per epoch.
    """
    enc = model.cabang["graf"]["enc"]
    d_in, hidden = enc.proj.in_features, enc.proj.out_features
    token = nn.Parameter(torch.zeros(d_in, device=DEVICE))
    j0 = cfg.get("pralatih_kolom_awal", 0)
    dekoder = nn.Linear(hidden, d_in - j0).to(DEVICE)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dekoder.parameters()) + [token], lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    dl = DataLoader(data, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    riwayat = []
    for ep in range(1, cfg["pralatih_epoch"] + 1):
        enc.train()
        total, n = 0.0, 0
        for b in dl:
            b = b.to(DEVICE)
            mask = torch.rand(b.x.size(0), device=DEVICE) < cfg["rasio_mask"]
            x = torch.where(mask[:, None], token.expand_as(b.x), b.x)
            h = enc.representasi_simpul(x, b.edge_index, b.edge_attr)
            loss = F.mse_loss(dekoder(h[mask]), b.x[mask][:, j0:])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 2.0)
            opt.step()
            total += loss.item(); n += 1
        riwayat.append((ep, total / n))
    return np.array(riwayat)


def jalankan_cv_usulan(y, situs, fold, siapkan_fold, buat_model, cfg=None, cache_dir=None, verbose=True, cfg_fold=None):
    """Validasi silang dengan fold tetap.

    siapkan_fold(k, idx_latih_penuh) -> list Data untuk SEMUA subjek, dengan statistik fitur dari idx_latih_penuh saja.
    buat_model(contoh) -> ModelUsulan baru; contoh = satu objek Data fold itu (dimensi fitur, terutama jumlah
        komponen PCA cabang linear, bisa sedikit berbeda antar fold).
    Setiap fold dilatih dengan beberapa seed (bobot awal, dropout, urutan batch, dan potongan validasi berbeda).
    Seed s di seluruh fold = RUN ke-s, yaitu satu validasi silang 5-fold yang lengkap. Prediksi setiap run disimpan
    di "p_run" (N, jumlah seed) untuk dilaporkan sebagai rata-rata ± SD antar run; rata-rata antar seed ("p" dan
    kunci lain) adalah prediksi ensemble.
    Bila cfg["pralatih_epoch"] > 0, encoder graf setiap seed dipralatih dulu (pralatih_graf) pada graf data latih
    fold itu tanpa label, lalu seluruh model dilatih seperti biasa.
    Hasil per fold disimpan ke <cache_dir>/fold<k>.npz sehingga pelatihan yang terhenti bisa dilanjutkan.
    cfg_fold (opsional, dipakai S4/Optuna): {k: pengaturan pelatihan khusus fold k}. Bila diberikan, buat_model dipanggil
        sebagai buat_model(contoh, k) supaya setiap fold bisa memakai hyperparameter model hasil pencariannya sendiri.
    Keluaran: dict berisi array sepanjang jumlah subjek (p, p_run, p_<cabang>, alpha, beta) + log per seed.
    """
    cfg = {**CFG_DEFAULT, **(cfg or {})}
    cache_dir = Path(cache_dir) if cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    N, gabung, log = len(y), {}, []
    t0 = time.perf_counter()
    for k in [int(v) for v in sorted(np.unique(fold))]:
        uji, latih_penuh = np.where(fold == k)[0], np.where(fold != k)[0]
        berkas = cache_dir / f"fold{k}.npz" if cache_dir else None
        if berkas is not None and berkas.exists():
            c = dict(np.load(berkas, allow_pickle=True))
            hasil_fold, log_fold = {kk: v for kk, v in c.items() if kk != "log"}, c["log"].tolist()
            sumber = "dari cache"
        else:
            data = siapkan_fold(k, latih_penuh)
            data_uji = [data[i] for i in uji]
            per_seed, log_fold = [], []
            cfg_k = {**cfg, **(cfg_fold or {}).get(k, {})}
            for s in cfg_k["seeds"]:
                set_seed(1000 * k + s)
                tr, va = split_validasi(latih_penuh, y, situs, 1000 * k + s, cfg_k["val_size"])
                model = (buat_model(data[0]) if cfg_fold is None else buat_model(data[0], k)).to(DEVICE)
                n_c = np.bincount(y[tr], minlength=2)
                model.bobot_kelas.copy_(torch.tensor(len(tr) / (2 * n_c), dtype=torch.float32))   # class weight seimbang
                t1 = time.perf_counter()
                riwayat_pralatih = np.zeros((0, 2))
                if cfg_k.get("pralatih_epoch", 0) > 0:
                    riwayat_pralatih = pralatih_graf(model, [data[i] for i in latih_penuh], cfg_k)
                model, info = latih_model(model, [data[i] for i in tr], [data[i] for i in va], cfg_k)
                per_seed.append(prediksi(model, data_uji, cfg_k["batch_eval"]))
                log_fold.append({"fold": k, "seed": s, "epoch_terbaik": info["epoch_terbaik"], "auc_val": info["auc_val"],
                                 "detik": time.perf_counter() - t1, "riwayat": info["riwayat"], "riwayat_pralatih": riwayat_pralatih})
                if verbose:
                    print(f"    fold {k} seed {s}: epoch terbaik {info['epoch_terbaik']:3d} | AUC val {info['auc_val']*100:.2f} "
                          f"| {log_fold[-1]['detik']/60:.1f} menit", flush=True)
            hasil_fold = {kk: np.mean([h[kk] for h in per_seed], axis=0) for kk in per_seed[0]}
            hasil_fold["p_run"] = np.stack([h["p"] for h in per_seed], axis=1)                      # (n_uji, jumlah seed)
            hasil_fold["idx"] = uji
            if berkas is not None:
                np.savez(berkas, **hasil_fold, log=np.array(log_fold, dtype=object))
            del data, data_uji
            gc.collect()
            torch.cuda.empty_cache()
            sumber = f"{(time.perf_counter() - t0) / 60:.0f} menit berjalan"
        for kk, v in hasil_fold.items():
            if kk == "idx":
                continue
            if kk not in gabung:
                gabung[kk] = np.full((N,) + v.shape[1:], np.nan, dtype=np.float64)
            gabung[kk][hasil_fold["idx"]] = v
        log.extend(log_fold)
        if verbose:
            p, yy = hasil_fold["p"], y[hasil_fold["idx"]]
            print(f"  fold {k}: ACC {((p >= 0.5) == yy).mean()*100:.2f} | AUC {roc_auc_score(yy, p)*100:.2f} ({sumber})", flush=True)
    gabung["log"] = log
    return gabung
