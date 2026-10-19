"""Protokol pelatihan bersama: split validasi di dalam fold, early stopping berbasis AUC validasi,
ensemble beberapa seed, dan evaluasi 5-fold memakai kolom `fold` di kohort_final.csv."""
import copy
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from .metrics import hitung_metrik, ringkas

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG_DEFAULT = {
    "lr": 1e-3, "weight_decay": 1e-3, "batch_size": 32, "max_epochs": 200,
    "min_epochs": 20, "patience": 40, "val_size": 0.15, "seeds": (0, 1, 2),
    "batch_eval": 256,          # ukuran batch saat validasi/uji (kecilkan untuk input besar, mis. volume 3D)
}


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_validasi(idx_latih, y, situs, seed, val_size=0.15):
    """Validasi diambil dari data latih saja, bertingkat menurut situs × diagnosis."""
    strata = np.array([f"{s}_{l}" for s, l in zip(situs[idx_latih], y[idx_latih])])
    _, cnt = np.unique(strata, return_counts=True)
    if cnt.min() < 2:
        strata = y[idx_latih]
    tr, va = train_test_split(idx_latih, test_size=val_size, stratify=strata, random_state=seed)
    return tr, va


def _logits(keluaran):
    """Model boleh mengembalikan logits saja, atau (logits, loss_tambahan) saat training."""
    return keluaran[0] if isinstance(keluaran, tuple) else keluaran


@torch.no_grad()
def prediksi(model, loader):
    model.eval()
    p, y = [], []
    for b in loader:
        b = b.to(DEVICE)
        p.append(F.softmax(_logits(model(b)), dim=1)[:, 1].cpu())
        y.append(b.y.cpu())
    return torch.cat(y).numpy(), torch.cat(p).numpy()


def latih(model, data_latih, data_val, cfg):
    """Melatih satu model; menyimpan bobot dengan AUC validasi terbaik (setelah min_epochs)."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["max_epochs"])
    dl_latih = DataLoader(data_latih, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    dl_val = DataLoader(data_val, batch_size=cfg.get("batch_eval", 256))
    terbaik, bobot, epoch_terbaik, sabar = -1.0, copy.deepcopy(model.state_dict()), 0, 0
    for ep in range(1, cfg["max_epochs"] + 1):
        model.train()
        for b in dl_latih:
            b = b.to(DEVICE)
            opt.zero_grad()
            keluaran = model(b)
            loss = F.cross_entropy(_logits(keluaran), b.y.view(-1))
            if isinstance(keluaran, tuple):
                loss = loss + keluaran[1]          # mis. loss adversarial situs (CAS-GNN)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()
        y_v, p_v = prediksi(model, dl_val)
        auc = roc_auc_score(y_v, p_v)
        if ep >= cfg["min_epochs"] and auc > terbaik + 1e-4:
            terbaik, bobot, epoch_terbaik, sabar = auc, copy.deepcopy(model.state_dict()), ep, 0
        elif ep >= cfg["min_epochs"]:
            sabar += 1
            if sabar >= cfg["patience"]:
                break
    model.load_state_dict(bobot)
    return model, {"epoch_terbaik": epoch_terbaik, "auc_val": terbaik}


def jalankan_cv(kohort, buat_dataset, buat_model, cfg=None, verbose=True, cache_dir=None):
    """5-fold CV dengan fold tetap dari kohort.

    buat_dataset(idx_latih_penuh, idx_latih, idx_val, idx_uji) -> (data_latih, data_val, data_uji)
        semua normalisasi/imputasi WAJIB di-fit hanya pada idx_latih_penuh atau idx_latih.
    buat_model() -> nn.Module baru.
    Probabilitas uji dirata-ratakan dari beberapa seed (ensemble).
    cache_dir (opsional): hasil setiap fold disimpan ke <cache_dir>/fold<k>.npz; fold yang sudah ada
        dibaca ulang, sehingga pelatihan yang terhenti bisa dilanjutkan tanpa mengulang fold selesai.
    """
    cfg = {**CONFIG_DEFAULT, **(cfg or {})}
    y = kohort["LABEL_BIN"].to_numpy()
    situs = kohort["SITE_ID"].to_numpy()
    fold = kohort["fold"].to_numpy()
    per_fold, prediksi_semua, log = [], [], []
    t0 = time.time()
    if cache_dir is not None:
        from pathlib import Path
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    for k in [int(v) for v in sorted(np.unique(fold))]:
        idx_uji = np.where(fold == k)[0]
        idx_latih_penuh = np.where(fold != k)[0]
        berkas = cache_dir / f"fold{k}.npz" if cache_dir is not None else None
        if berkas is not None and berkas.exists():
            c = np.load(berkas, allow_pickle=True)
            y_u, p_rata = c["y_uji"], c["p_uji"]
            log.extend(c["log"].tolist())
            m = hitung_metrik(y_u, p_rata)
            per_fold.append({"fold": k, **m})
            prediksi_semua.append(np.column_stack([idx_uji, y_u, p_rata]))
            if verbose:
                print(f"  fold {k}: ACC {m['accuracy']*100:.2f} AUC {m['auc']*100:.2f} (dibaca dari cache)", flush=True)
            continue
        p_seed = []
        for s in cfg["seeds"]:
            set_seed(1000 * k + s)
            idx_latih, idx_val = split_validasi(idx_latih_penuh, y, situs, seed=1000 * k + s, val_size=cfg["val_size"])
            d_latih, d_val, d_uji = buat_dataset(idx_latih_penuh, idx_latih, idx_val, idx_uji)
            model = buat_model().to(DEVICE)
            model, info = latih(model, d_latih, d_val, cfg)
            y_u, p_u = prediksi(model, DataLoader(d_uji, batch_size=cfg["batch_eval"]))
            p_seed.append(p_u)
            log.append({"fold": k, "seed": s, **info})
        p_rata = np.mean(p_seed, axis=0)
        if berkas is not None:
            np.savez(berkas, y_uji=y_u, p_uji=p_rata, log=np.array([l for l in log if l["fold"] == k], dtype=object))
        m = hitung_metrik(y_u, p_rata)
        per_fold.append({"fold": k, **m})
        prediksi_semua.append(np.column_stack([idx_uji, y_u, p_rata]))
        if verbose:
            ep = [l["epoch_terbaik"] for l in log if l["fold"] == k]
            print(f"  fold {k}: ACC {m['accuracy']*100:.2f} AUC {m['auc']*100:.2f} (epoch terbaik {ep})", flush=True)
    r = ringkas(per_fold)
    if verbose:
        print(f"  RATA-RATA: ACC {r['accuracy'][0]:.2f}±{r['accuracy'][1]:.2f} | AUC {r['auc'][0]:.2f} "
              f"| waktu {time.time() - t0:.0f} s", flush=True)
    return {"per_fold": per_fold, "ringkasan": r, "prediksi": np.vstack(prediksi_semua), "log": log}


def simpan_hasil(hasil, kohort, nama, folder_tabel):
    """Menyimpan metrik per fold dan prediksi per subjek ke result/tabel/<nama>_*.csv."""
    import pandas as pd
    from pathlib import Path
    folder_tabel = Path(folder_tabel)
    folder_tabel.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hasil["per_fold"]).to_csv(folder_tabel / f"{nama}_per_fold.csv", index=False)
    pred = pd.DataFrame(hasil["prediksi"], columns=["idx", "y_true", "p_asd"])
    pred["idx"] = pred["idx"].astype(int)
    pred = pred.assign(FILE_ID=kohort["FILE_ID"].to_numpy()[pred["idx"]],
                       SITE_ID=kohort["SITE_ID"].to_numpy()[pred["idx"]],
                       fold=kohort["fold"].to_numpy()[pred["idx"]]).drop(columns="idx")
    pred.to_csv(folder_tabel / f"{nama}_prediksi.csv", index=False)
    pd.DataFrame(hasil["log"]).to_csv(folder_tabel / f"{nama}_log_seed.csv", index=False)
