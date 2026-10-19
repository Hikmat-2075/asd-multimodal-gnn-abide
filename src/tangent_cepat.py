"""Tangent space dari kovarians yang sudah dihitung, versi batch (banyak matriks sekaligus).

Algoritma rata-rata geometrik identik dengan nilearn `_geometric_mean` (Fletcher dkk., 2007, Algoritma 3):
gradient descent pada manifold SPD, inisialisasi rata-rata aritmetik, langkah dibagi dua bila norma naik.
Bedanya hanya cara komputasi: dekomposisi eigen dikerjakan per kelompok matriks (PyTorch CPU, float64)
alih-alih satu per satu dalam loop Python, sehingga jauh lebih cepat untuk ribuan subjek.
"""
import numpy as np


def _eigh(M):
    """Dekomposisi eigen tumpukan matriks simetris, float64. Memakai PyTorch (CPU) bila tersedia:
    ±3-4x lebih cepat daripada numpy dengan hasil identik (selisih nilai eigen ~1e-10)."""
    try:
        import torch
        w, V = torch.linalg.eigh(torch.from_numpy(np.ascontiguousarray(M, dtype=np.float64)))
        return w.numpy(), V.numpy()
    except ImportError:
        return np.linalg.eigh(M)


def _fungsi_eigen(fungsi, M):
    """Terapkan fungsi pada nilai eigen tumpukan matriks simetris (..., p, p)."""
    w, V = _eigh(M)
    return (V * fungsi(w)[..., None, :]) @ np.swapaxes(V, -1, -2)


def rata_rata_geometrik(C, max_iter=10, tol=1e-7, kelompok=128):
    """C: (n, p, p) kovarians SPD. Mengembalikan rata-rata geometrik (p, p), float64."""
    C = np.asarray(C)
    gmean = np.mean(C, axis=0, dtype=np.float64)
    norm_lama, langkah = np.inf, 1.0
    for _ in range(max_iter):
        w, V = _eigh(gmean)
        inv_akar = (V / np.sqrt(w)) @ V.T
        jumlah_log = np.zeros_like(gmean)
        for i in range(0, len(C), kelompok):
            B = np.asarray(C[i:i + kelompok], dtype=np.float64)
            jumlah_log += _fungsi_eigen(np.log, inv_akar @ B @ inv_akar).sum(axis=0)
        log_rata = jumlah_log / len(C)
        if np.any(np.isnan(log_rata)):
            raise FloatingPointError("Nilai NaN setelah logaritma matriks.")
        norm = np.linalg.norm(log_rata)
        akar = (V * np.sqrt(w)) @ V.T
        gmean = akar @ _fungsi_eigen(lambda x: np.exp(x * langkah), log_rata) @ akar
        if norm < norm_lama:
            norm_lama = norm
        elif norm > norm_lama:
            langkah /= 2.0
            norm = norm_lama
        if tol is not None and norm / gmean.size < tol:
            break
    return gmean


def tangent(C, acuan, kelompok=128):
    """Vektor tangent space (segitiga bawah tanpa diagonal, urutan sama dengan nilearn sym_matrix_to_vec)."""
    from nilearn.connectome import sym_matrix_to_vec
    w, V = _eigh(acuan)
    inv_akar = (V / np.sqrt(w)) @ V.T
    hasil = []
    for i in range(0, len(C), kelompok):
        B = np.asarray(C[i:i + kelompok], dtype=np.float64)
        hasil.append(sym_matrix_to_vec(_fungsi_eigen(np.log, inv_akar @ B @ inv_akar), discard_diagonal=True).astype(np.float32))
    return np.vstack(hasil)
