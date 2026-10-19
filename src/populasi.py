"""Model graf populasi (simpul = subjek), gaya Parisot dkk. (2018) dan dasar DAGMNet.

Graf populasi bersifat transduktif: seluruh subjek (latih + uji) menjadi simpul, tetapi label dan
seluruh statistik (normalisasi, seleksi fitur) hanya diambil dari subjek latih pada fold tersebut.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import ChebConv, GATv2Conv

from .metrics import hitung_metrik, ringkas
from .training import DEVICE, set_seed, split_validasi


def graf_fenotipe(kohort, fitur, k_tetangga=None, ambang_umur=2.0, pakai_situs=True):
    """Bobot sisi = kemiripan fenotipe (situs sama + jenis kelamin sama + umur dekat)
    dikali kemiripan fitur (korelasi), seperti Parisot dkk. (2018).
    pakai_situs=False: situs tidak ikut menentukan sisi (uji ketahanan terhadap efek situs)."""
    situs = kohort["SITE_ID"].to_numpy()
    sex = kohort["SEX"].to_numpy()
    umur = kohort["AGE_AT_SCAN"].to_numpy()
    skor = ((situs[:, None] == situs[None, :]).astype(float) * float(pakai_situs)
            + (sex[:, None] == sex[None, :]).astype(float)
            + (np.abs(umur[:, None] - umur[None, :]) < ambang_umur).astype(float))
    kor = np.corrcoef(fitur)
    sigma = np.mean(np.sqrt(np.maximum(2 - 2 * kor, 0)))
    sim = np.exp(-(2 - 2 * kor) / (2 * sigma ** 2))
    w = skor * sim
    np.fill_diagonal(w, 0)
    if k_tetangga:
        ambang = -np.sort(-w, axis=1)[:, k_tetangga - 1:k_tetangga]
        w = np.where(w >= ambang, w, 0)
        w = np.maximum(w, w.T)
    src, dst = np.nonzero(w)
    return (torch.tensor(np.vstack([src, dst]), dtype=torch.long),
            torch.tensor(w[src, dst], dtype=torch.float32))


class GCNPopulasi(nn.Module):
    """Chebyshev GCN pada graf populasi (Parisot dkk., 2018)."""

    def __init__(self, in_dim, hidden=16, n_layer=2, K=3, dropout=0.3, n_kelas=2):
        super().__init__()
        dims = [in_dim] + [hidden] * n_layer
        self.convs = nn.ModuleList([ChebConv(dims[i], dims[i + 1], K=K) for i in range(n_layer)])
        self.out = nn.Linear(hidden, n_kelas)
        self.dropout = dropout

    def forward(self, x, ei, ew):
        for conv in self.convs:
            x = F.dropout(x, self.dropout, self.training)
            x = F.relu(conv(x, ei, ew))
        return self.out(F.dropout(x, self.dropout, self.training))


class GATPopulasi(nn.Module):
    """GATv2 pada graf populasi (komponen DAGP pada DAGMNet)."""

    def __init__(self, in_dim, hidden=32, heads=4, dropout=0.3, n_kelas=2):
        super().__init__()
        self.g1 = GATv2Conv(in_dim, hidden, heads=heads, dropout=dropout, edge_dim=1)
        self.g2 = GATv2Conv(hidden * heads, hidden, heads=1, dropout=dropout, edge_dim=1)
        self.out = nn.Linear(hidden, n_kelas)
        self.dropout = dropout

    def forward(self, x, ei, ew):
        x = F.dropout(x, self.dropout, self.training)
        x = F.elu(self.g1(x, ei, ew.view(-1, 1)))
        x = F.dropout(x, self.dropout, self.training)
        x = F.elu(self.g2(x, ei, ew.view(-1, 1)))
        return self.out(x)


def seleksi_fitur(X, y, idx_latih, n_fitur=2000):
    """RFE dengan Ridge, di-fit HANYA pada subjek latih (mencegah kebocoran)."""
    sc = StandardScaler().fit(X[idx_latih])
    Xs = sc.transform(X)
    rfe = RFE(RidgeClassifier(), n_features_to_select=n_fitur, step=0.25).fit(Xs[idx_latih], y[idx_latih])
    return Xs[:, rfe.support_].astype(np.float32)


def jalankan_cv_populasi(kohort, X, buat_model, n_fitur=2000, k_tetangga=None, cfg=None, verbose=True):
    cfg = {"lr": 5e-3, "weight_decay": 5e-4, "epochs": 300, "seeds": (0, 1, 2), "val_size": 0.15, **(cfg or {})}
    y = kohort["LABEL_BIN"].to_numpy()
    situs = kohort["SITE_ID"].to_numpy()
    fold = kohort["fold"].to_numpy()
    per_fold = []
    for k in [int(v) for v in sorted(np.unique(fold))]:
        uji = np.where(fold == k)[0]
        latih_penuh = np.where(fold != k)[0]
        Xf = seleksi_fitur(X, y, latih_penuh, n_fitur)
        ei, ew = graf_fenotipe(kohort, Xf, k_tetangga)
        x_t, ei, ew = torch.tensor(Xf).to(DEVICE), ei.to(DEVICE), ew.to(DEVICE)
        y_t = torch.tensor(y).to(DEVICE)
        p_seed = []
        for s in cfg["seeds"]:
            set_seed(1000 * k + s)
            latih, val = split_validasi(latih_penuh, y, situs, 1000 * k + s, cfg["val_size"])
            model = buat_model(Xf.shape[1]).to(DEVICE)
            opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
            terbaik, bobot = -1, None
            for ep in range(cfg["epochs"]):
                model.train()
                opt.zero_grad()
                F.cross_entropy(model(x_t, ei, ew)[latih], y_t[latih]).backward()
                opt.step()
                model.eval()
                with torch.no_grad():
                    p = F.softmax(model(x_t, ei, ew), 1)[:, 1].cpu().numpy()
                auc = roc_auc_score(y[val], p[val])
                if ep >= 20 and auc > terbaik:
                    terbaik, bobot = auc, copy.deepcopy(model.state_dict())
            model.load_state_dict(bobot)
            model.eval()
            with torch.no_grad():
                p_seed.append(F.softmax(model(x_t, ei, ew), 1)[:, 1].cpu().numpy()[uji])
        m = hitung_metrik(y[uji], np.mean(p_seed, 0))
        per_fold.append({"fold": k, **m})
        if verbose:
            print(f"  fold {k}: ACC {m['accuracy']*100:.2f} AUC {m['auc']*100:.2f}", flush=True)
    r = ringkas(per_fold)
    if verbose:
        print(f"  RATA-RATA: ACC {r['accuracy'][0]:.2f}±{r['accuracy'][1]:.2f} | AUC {r['auc'][0]:.2f}", flush=True)
    return {"per_fold": per_fold, "ringkasan": r}


# ---------------------------------------------------------------------------------------------
# Versi 2 (notebook 13b): fitur berbeda per fold (mis. tangent space yang acuannya dari data latih),
# hasil per fold disimpan agar bisa dilanjutkan, dan prediksi per subjek dikembalikan.
# ---------------------------------------------------------------------------------------------
class MLPSimpul(nn.Module):
    """Kontrol tanpa graf: MLP per subjek dengan protokol pelatihan yang sama (sisi diabaikan)."""

    def __init__(self, in_dim, hidden=32, dropout=0.3, n_kelas=2):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, hidden), nn.ELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, n_kelas))

    def forward(self, x, ei, ew):
        return self.net(x)


class DenganProyeksi(nn.Module):
    """Proyeksi linear fitur simpul sebelum model graf. Wajib untuk fitur berdimensi sangat besar pada
    ChebConv, yang menyebarkan fitur mentah lewat setiap sisi (memori ~ jumlah sisi x dimensi fitur)."""

    def __init__(self, in_dim, buat_inti, dim_proyeksi=64, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(in_dim, dim_proyeksi)
        self.inti = buat_inti(dim_proyeksi)
        self.dropout = dropout

    def forward(self, x, ei, ew):
        x = F.elu(self.proj(F.dropout(x, self.dropout, self.training)))
        return self.inti(x, ei, ew)


def seleksi_fitur_multi(blok, y, idx_latih, n_per_blok=500):
    """Standardisasi + RFE (Ridge) per blok fitur (mis. per atlas), di-fit HANYA pada subjek latih,
    lalu blok terpilih digabung. blok: dict nama -> array (subjek x fitur)."""
    hasil = []
    for nama, X in blok.items():
        sc = StandardScaler().fit(X[idx_latih])
        Xs = sc.transform(X).astype(np.float32)
        n = min(n_per_blok, Xs.shape[1])
        rfe = RFE(RidgeClassifier(), n_features_to_select=n, step=0.25).fit(Xs[idx_latih], y[idx_latih])
        hasil.append(Xs[:, rfe.support_])
    return np.hstack(hasil)


def latih_populasi(model, x_t, ei, ew, y_t, latih, val, y, cfg):
    """Pelatihan full-batch; bobot dengan AUC validasi terbaik (setelah min_epochs) dipakai."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    terbaik, bobot, ep_terbaik, sabar = -1.0, copy.deepcopy(model.state_dict()), 0, 0
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(x_t, ei, ew)[latih], y_t[latih]).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            p = F.softmax(model(x_t, ei, ew), 1)[:, 1].cpu().numpy()
        auc = roc_auc_score(y[val], p[val])
        if ep >= cfg["min_epochs"] and auc > terbaik + 1e-4:
            terbaik, bobot, ep_terbaik, sabar = auc, copy.deepcopy(model.state_dict()), ep, 0
        elif ep >= cfg["min_epochs"]:
            sabar += 1
            if sabar >= cfg["patience"]:
                break
    model.load_state_dict(bobot)
    model.eval()
    return model, {"epoch_terbaik": ep_terbaik, "auc_val": terbaik}


def jalankan_cv_populasi_v2(kohort, ambil_fitur, buat_graf, buat_model, cfg=None, cache_dir=None, verbose=True):
    """5-fold CV graf populasi dengan fold tetap dari kohort.

    ambil_fitur(k, idx_latih_penuh) -> X (subjek x fitur), seluruh statistik di-fit pada data latih.
    buat_graf(X) atau buat_graf(X, k) -> (edge_index, edge_weight) untuk seluruh subjek (transduktif;
        label tidak dipakai). Bentuk dua argumen dipakai bila graf ditentukan per fold, terlepas dari X.
    buat_model(in_dim) -> nn.Module dengan forward(x, edge_index, edge_weight).
    """
    cfg = {"lr": 5e-3, "weight_decay": 5e-4, "epochs": 300, "min_epochs": 20, "patience": 60,
           "seeds": (0, 1, 2), "val_size": 0.15, **(cfg or {})}
    y = kohort["LABEL_BIN"].to_numpy()
    situs = kohort["SITE_ID"].to_numpy()
    fold = kohort["fold"].to_numpy()
    if cache_dir is not None:
        from pathlib import Path
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    per_fold, prediksi, log = [], [], []
    for k in [int(v) for v in sorted(np.unique(fold))]:
        uji = np.where(fold == k)[0]
        latih_penuh = np.where(fold != k)[0]
        berkas = cache_dir / f"fold{k}.npz" if cache_dir is not None else None
        if berkas is not None and berkas.exists():
            c = np.load(berkas, allow_pickle=True)
            p_rata = c["p_uji"]
            log.extend(c["log"].tolist())
            sumber = " (dibaca dari cache)"
        else:
            X = ambil_fitur(k, latih_penuh)
            import inspect
            ei, ew = buat_graf(X, k) if len(inspect.signature(buat_graf).parameters) == 2 else buat_graf(X)
            x_t, ei, ew = torch.tensor(X, dtype=torch.float32).to(DEVICE), ei.to(DEVICE), ew.to(DEVICE)
            y_t = torch.tensor(y).to(DEVICE)
            p_seed, log_k = [], []
            for s in cfg["seeds"]:
                set_seed(1000 * k + s)
                latih, val = split_validasi(latih_penuh, y, situs, 1000 * k + s, cfg["val_size"])
                model, info = latih_populasi(buat_model(X.shape[1]).to(DEVICE), x_t, ei, ew, y_t, latih, val, y, cfg)
                with torch.no_grad():
                    p_seed.append(F.softmax(model(x_t, ei, ew), 1)[:, 1].cpu().numpy()[uji])
                log_k.append({"fold": k, "seed": s, **info})
            p_rata = np.mean(p_seed, axis=0)
            log.extend(log_k)
            if berkas is not None:
                np.savez(berkas, p_uji=p_rata, log=np.array(log_k, dtype=object))
            sumber = f" (epoch terbaik {[l['epoch_terbaik'] for l in log_k]})"
        m = hitung_metrik(y[uji], p_rata)
        per_fold.append({"fold": k, **m})
        prediksi.append(np.column_stack([uji, y[uji], p_rata]))
        if verbose:
            print(f"  fold {k}: ACC {m['accuracy']*100:.2f} AUC {m['auc']*100:.2f}{sumber}", flush=True)
    r = ringkas(per_fold)
    if verbose:
        print(f"  RATA-RATA: ACC {r['accuracy'][0]:.2f}±{r['accuracy'][1]:.2f} | AUC {r['auc'][0]:.2f}", flush=True)
    return {"per_fold": per_fold, "ringkasan": r, "prediksi": np.vstack(prediksi), "log": log}
