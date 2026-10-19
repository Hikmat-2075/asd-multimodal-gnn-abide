"""Cabang citra rs-fMRI 3D: memuat peta ringkasan ABIDE PCP dan 3D CNN ringan (dipakai notebook 16).

Sama dengan rancangan notebook 15: masker otak MNI152 3 mm, volume dipotong ke kotak pembatas masker
(48 x 61 x 52), z-score per subjek di dalam masker (hanya memakai citra subjek itu sendiri), float16 di RAM.
"""
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
PETA_DIR = ROOT / "data" / "raw_pcp_derivatif"
PETA = ["reho", "alff", "falff", "degree_weighted", "vmhc"]


def muat_volume_pcp(file_ids, peta=PETA, progress=None):
    """Mengembalikan (VOL float16 [subjek, peta, 48, 61, 52], masker kotak, cakupan [subjek, peta])."""
    from nilearn import datasets, image
    contoh = nib.load(PETA_DIR / peta[0] / f"{file_ids[0]}_{peta[0]}.nii.gz")
    masker = image.resample_to_img(datasets.load_mni152_brain_mask(resolution=3), contoh,
                                   interpolation="nearest").get_fdata() > 0
    idx = np.argwhere(masker)
    kotak = tuple(slice(a, b) for a, b in zip(idx.min(0), idx.max(0) + 1))
    M = masker[kotak]
    vol = np.zeros((len(file_ids), len(peta), *M.shape), dtype=np.float16)
    cakupan = np.zeros((len(file_ids), len(peta)))
    iterasi = progress(file_ids) if progress else file_ids
    for i, f in enumerate(iterasi):
        for j, p in enumerate(peta):
            x = np.asarray(nib.load(PETA_DIR / p / f"{f}_{p}.nii.gz").dataobj, dtype=np.float32)[kotak]
            v = x[M]
            cakupan[i, j] = (v != 0).mean()
            vol[i, j] = np.where(M, (x - v.mean()) / (v.std() + 1e-6), 0).astype(np.float16)
    return vol, M, cakupan


def _blok(i, o):
    return nn.Sequential(nn.Conv3d(i, o, 3, padding=1, bias=False), nn.BatchNorm3d(o), nn.ReLU(inplace=True),
                         nn.Conv3d(o, o, 3, padding=1, bias=False), nn.BatchNorm3d(o), nn.ReLU(inplace=True),
                         nn.MaxPool3d(2))


def augmentasi(x, geser_maks=3):
    """Balik kiri-kanan (sumbu pertama ruang MNI) dan geser acak per subjek; hanya saat pelatihan."""
    keluar = []
    for v in x:
        if torch.rand(1).item() < 0.5:
            v = torch.flip(v, dims=[1])
        geser = torch.randint(-geser_maks, geser_maks + 1, (3,)).tolist()
        keluar.append(torch.roll(v, shifts=geser, dims=(1, 2, 3)))
    return torch.stack(keluar)


class EncoderCitra3D(nn.Module):
    """4 blok konvolusi 3D (16-32-64-96 kanal) + global average pooling -> vektor 96 dimensi."""

    def __init__(self, n_kanal, geser_maks=3):
        super().__init__()
        self.fitur = nn.Sequential(_blok(n_kanal, 16), _blok(16, 32), _blok(32, 64), _blok(64, 96),
                                   nn.AdaptiveAvgPool3d(1), nn.Flatten())
        self.dim = 96
        self.geser_maks = geser_maks

    def forward(self, x):
        x = x.float()
        if self.training:
            x = augmentasi(x, self.geser_maks)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            return self.fitur(x).float()
