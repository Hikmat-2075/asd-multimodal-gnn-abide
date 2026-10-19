"""Memuat kohort, matriks FC, fitur sMRI, dan fenotipe; membangun graf otak per subjek."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

ATLAS = {"aal": 116, "harvard_oxford": 111, "cc200": 200, "cc400": 392}
FOLDER_FC = {"filt_global": "fc_matrices_global", "filt_noglobal": "fc_matrices"}


def muat_kohort():
    """Kohort final (987 subjek) beserta kolom fold dan label biner (1 = ASD)."""
    return pd.read_csv(PROC / "kohort_final.csv")


def muat_fc(atlas, file_ids, strategi="filt_global"):
    """Matriks FC Pearson r untuk daftar subjek: array (N, R, R) float32."""
    folder = PROC / FOLDER_FC[strategi] / atlas
    return np.stack([np.load(folder / f"{fid}_fc.npy").astype(np.float32) for fid in file_ids])


def fisher_z(fc):
    """Transformasi Fisher-z dengan diagonal 0."""
    z = np.arctanh(np.clip(fc, -0.999, 0.999)).astype(np.float32)
    idx = np.arange(fc.shape[-1])
    z[..., idx, idx] = 0.0
    return z


def segitiga_atas(mat):
    """Vektorisasi segitiga atas (tanpa diagonal) untuk model non-graf: (N, R*(R-1)/2)."""
    iu = np.triu_indices(mat.shape[-1], k=1)
    return mat[:, iu[0], iu[1]]


def bangun_graf(fc, x_node, y, top_k=0.2):
    """Graf per subjek: simpul = ROI, fitur simpul = x_node[i] (R x F),
    sisi = top_k bagian pasangan dengan |r| terbesar (dua arah), bobot sisi = r."""
    n_roi = fc.shape[-1]
    iu = np.triu_indices(n_roi, k=1)
    graf = []
    for i in range(len(fc)):
        r = fc[i][iu]
        ambang = np.quantile(np.abs(r), 1.0 - top_k)
        pilih = np.abs(r) >= ambang
        src, dst = iu[0][pilih], iu[1][pilih]
        ei = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])])
        ea = np.concatenate([r[pilih], r[pilih]])[:, None]
        graf.append(Data(x=torch.from_numpy(np.ascontiguousarray(x_node[i])),
                         edge_index=torch.from_numpy(ei).long(),
                         edge_attr=torch.from_numpy(ea.astype(np.float32)),
                         y=torch.tensor([int(y[i])])))
    return graf


def muat_smri(kohort):
    """Fitur FreeSurfer (265 kolom) yang diurutkan sesuai kohort."""
    smri = pd.read_csv(PROC / "smri_features" / "smri_freesurfer_fitur.csv")
    gabung = kohort[["SUB_ID"]].merge(smri, on="SUB_ID", how="left")
    kolom = [c for c in smri.columns if c not in ("SUB_ID", "FILE_ID")]
    return gabung[kolom].to_numpy(dtype=np.float32), kolom


def muat_fenotipe(kohort, kolom=("AGE_AT_SCAN", "SEX", "FIQ_RAW")):
    """Fenotipe mentah (nilai kosong tetap NaN, diimputasi di dalam fold)."""
    return kohort[list(kolom)].to_numpy(dtype=np.float32)
