"""Pencarian hyperparameter model usulan dengan Optuna (Skenario 4 = S3 + Optuna).

Protokol NESTED (anti-kebocoran data): pencarian dijalankan terpisah di setiap fold luar, dan hanya memakai
data latih fold itu. Data latih dibagi sekali menjadi potongan latih-dalam dan potongan validasi-tuning (15%,
bertingkat situs x diagnosis). Setiap trial melatih satu model pada latih-dalam dan dinilai dengan AUC pada
validasi-tuning. Data uji fold itu tidak pernah dilihat selama pencarian.

- Sampler TPE (seed tetap per fold, sehingga bisa diulang).
- Trial pertama = hyperparameter S3 (titik awal), sehingga pengaturan terbaik minimal sama baiknya dengan S3
  pada validasi-tuning.
- MedianPruner: trial yang AUC validasinya di bawah median trial lain pada epoch yang sama dihentikan lebih awal.
- Hasil setiap fold disimpan di SQLite, sehingga pencarian yang terhenti bisa dilanjutkan.
"""
import gc
import time

import numpy as np
import optuna
import torch

from src.usulan import DEVICE, CFG_DEFAULT, latih_model, set_seed, split_validasi

# hyperparameter S3 (titik awal pencarian = trial pertama)
HIPER_S3 = dict(hidden=64, n_layer=2, heads=4, dropout=0.5, dropout_input=0.2, dropedge=0.2, dropout_linear=0.2, bobot_bantu=0.5,
                lr=1e-3, weight_decay=1e-3, batch_size=32)

# ruang pencarian: nama -> (jenis, pilihan / batas)
RUANG = {
    "hidden":         ("kategori", [32, 64, 128]),
    "n_layer":        ("kategori", [1, 2, 3]),
    "heads":          ("kategori", [2, 4]),
    "dropout":        ("float", (0.2, 0.6)),
    "dropout_input":  ("float", (0.0, 0.4)),
    "dropedge":       ("float", (0.0, 0.4)),
    "dropout_linear": ("float", (0.0, 0.5)),
    "bobot_bantu":    ("float", (0.1, 1.0)),
    "lr":             ("float_log", (2e-4, 3e-3)),
    "weight_decay":   ("float_log", (1e-5, 1e-2)),
    "batch_size":     ("kategori", [16, 32, 64]),
}
KUNCI_MODEL = ("hidden", "n_layer", "heads", "dropout", "dropout_input", "dropedge", "dropout_linear", "bobot_bantu")
KUNCI_LATIH = ("lr", "weight_decay", "batch_size")


def saran_hiper(trial):
    """Satu set hyperparameter dari ruang pencarian."""
    h = {}
    for nama, (jenis, batas) in RUANG.items():
        if jenis == "kategori":
            h[nama] = trial.suggest_categorical(nama, batas)
        elif jenis == "float":
            h[nama] = trial.suggest_float(nama, *batas)
        else:
            h[nama] = trial.suggest_float(nama, *batas, log=True)
    return h


def pisah_hiper(h):
    """-> (hiper model untuk ModelUsulan, pengaturan latih untuk latih_model)."""
    return {k: h[k] for k in KUNCI_MODEL}, {k: h[k] for k in KUNCI_LATIH}


def cari_hiper_fold(k, data, latih_penuh, y, situs, buat_model_hiper, n_trial, berkas_db, cfg_dasar=None, verbose=True):
    """Pencarian Optuna untuk fold luar k.

    data: list Data PyG semua subjek (statistik fitur dari latih_penuh saja, seperti pelatihan biasa).
    buat_model_hiper(contoh, hiper_model) -> ModelUsulan baru.
    Keluaran: objek study (study.best_params = hyperparameter terbaik fold k).
    """
    cfg_dasar = {**CFG_DEFAULT, **(cfg_dasar or {})}
    tr, va = split_validasi(latih_penuh, y, situs, 7000 + k, cfg_dasar["val_size"])   # potongan tetap untuk semua trial
    d_tr, d_va = [data[i] for i in tr], [data[i] for i in va]
    n_c = np.bincount(y[tr], minlength=2)
    bobot_kelas = torch.tensor(len(tr) / (2 * n_c), dtype=torch.float32)

    def objektif(trial):
        h = saran_hiper(trial)
        if h["hidden"] % h["heads"]:
            raise optuna.TrialPruned()
        hm, hl = pisah_hiper(h)
        set_seed(7000 + k)
        model = buat_model_hiper(data[0], hm).to(DEVICE)
        model.bobot_kelas.copy_(bobot_kelas)

        def lapor(ep, auc):
            trial.report(auc, ep)
            if ep >= cfg_dasar["min_epochs"] and trial.should_prune():
                raise optuna.TrialPruned()

        t0 = time.perf_counter()
        try:
            _, info = latih_model(model, d_tr, d_va, {**cfg_dasar, **hl}, lapor=lapor)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        trial.set_user_attr("epoch_terbaik", int(info["epoch_terbaik"]))
        trial.set_user_attr("menit", round((time.perf_counter() - t0) / 60, 2))
        return float(info["auc_val"])

    study = optuna.create_study(study_name=f"S4_fold{k}", storage=f"sqlite:///{berkas_db}", load_if_exists=True,
                                direction="maximize", sampler=optuna.samplers.TPESampler(seed=k),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=cfg_dasar["min_epochs"]))
    if len(study.trials) == 0:
        study.enqueue_trial(HIPER_S3)                                         # trial 0 = pengaturan S3
    selesai = [t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)]
    sisa = n_trial - len(selesai)
    if sisa > 0:
        if verbose:
            def cetak(st, t):
                status = "dihentikan awal" if t.state == optuna.trial.TrialState.PRUNED else f"AUC val {t.value * 100:.2f}"
                menit = t.user_attrs.get("menit", (t.datetime_complete - t.datetime_start).total_seconds() / 60)
                terbaik = max([x.value for x in st.trials if x.state == optuna.trial.TrialState.COMPLETE], default=float("nan"))
                print(f"    fold {k} trial {t.number:2d}: {status} | {menit:.1f} menit | terbaik sejauh ini {terbaik * 100:.2f}", flush=True)
            study.optimize(objektif, n_trials=sisa, callbacks=[cetak], gc_after_trial=True)
        else:
            study.optimize(objektif, n_trials=sisa, gc_after_trial=True)
    return study


# grid search (pembanding Optuna, notebook 21): hyperparameter yang tidak ada di grid tetap = S3
GRID_DEFAULT = {"hidden": [32, 64, 128], "n_layer": [1, 2, 3], "lr": [5e-4, 1e-3, 2e-3]}


def cari_hiper_grid_fold(k, data, latih_penuh, y, situs, buat_model_hiper, grid, berkas_db, cfg_dasar=None, verbose=True):
    """Grid search untuk fold luar k, dengan protokol yang sama persis dengan cari_hiper_fold.

    Sama: potongan latih-dalam / validasi-tuning (seed 7000 + k), seed model, bobot kelas, dan ukuran AUC validasi.
    Beda: semua kombinasi grid dicoba (GridSampler) dan tidak ada trial yang dihentikan awal (tanpa pruner).
    Hyperparameter di luar grid memakai nilai S3. Hasil disimpan di SQLite sehingga bisa dilanjutkan.
    """
    cfg_dasar = {**CFG_DEFAULT, **(cfg_dasar or {})}
    tr, va = split_validasi(latih_penuh, y, situs, 7000 + k, cfg_dasar["val_size"])   # potongan sama dengan Optuna
    d_tr, d_va = [data[i] for i in tr], [data[i] for i in va]
    n_c = np.bincount(y[tr], minlength=2)
    bobot_kelas = torch.tensor(len(tr) / (2 * n_c), dtype=torch.float32)
    n_kombinasi = int(np.prod([len(v) for v in grid.values()]))

    def objektif(trial):
        h = {**HIPER_S3, **{nama: trial.suggest_categorical(nama, nilai) for nama, nilai in grid.items()}}
        hm, hl = pisah_hiper(h)
        set_seed(7000 + k)
        model = buat_model_hiper(data[0], hm).to(DEVICE)
        model.bobot_kelas.copy_(bobot_kelas)
        t0 = time.perf_counter()
        try:
            _, info = latih_model(model, d_tr, d_va, {**cfg_dasar, **hl})
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        trial.set_user_attr("epoch_terbaik", int(info["epoch_terbaik"]))
        trial.set_user_attr("menit", round((time.perf_counter() - t0) / 60, 2))
        return float(info["auc_val"])

    study = optuna.create_study(study_name=f"S4grid_fold{k}", storage=f"sqlite:///{berkas_db}", load_if_exists=True,
                                direction="maximize", sampler=optuna.samplers.GridSampler(grid, seed=k),
                                pruner=optuna.pruners.NopPruner())
    selesai = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    sisa = n_kombinasi - len(selesai)
    if sisa > 0:
        def cetak(st, t):
            terbaik = max([x.value for x in st.trials if x.state == optuna.trial.TrialState.COMPLETE], default=float("nan"))
            print(f"    fold {k} kombinasi {len([x for x in st.trials if x.state == optuna.trial.TrialState.COMPLETE]):2d}/{n_kombinasi}: "
                  f"{t.params} AUC val {t.value * 100:.2f} | {t.user_attrs.get('menit', float('nan')):.1f} menit | terbaik {terbaik * 100:.2f}", flush=True)
        study.optimize(objektif, n_trials=sisa, callbacks=[cetak] if verbose else None, gc_after_trial=True)
    return study
