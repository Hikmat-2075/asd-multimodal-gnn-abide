"""Preprocessing sMRI mandiri dengan ANTsPy untuk seluruh subjek (pipeline sama dengan notebook 07a).

Tahap per subjek: N4 -> registrasi SyN ke ICBM152 2009 (1 mm) -> skull stripping berbasis template ->
segmentasi Atropos (prior CSF/GM/WM) -> atlas fMRI (AAL, HO, CC200, CC400) ditarik ke ruang subjek ->
volume GM per wilayah. Fungsi `proses_subjek` sengaja berada di modul (bukan di notebook) agar bisa
dijalankan paralel dengan multiprocessing di Windows.
"""
import json
import os
import shutil
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed" / "smri_antspy"

ATLAS_NII = {
    "aal": ROOT / "data" / "atlas" / "aal" / "aal_roi_atlas.nii.gz",
    "harvard_oxford": ROOT / "data" / "atlas" / "harvard_oxford" / "ho_roi_atlas.nii.gz",
    "cc200": ROOT / "data" / "atlas" / "cc200" / "cc200_roi_atlas.nii.gz",
    "cc400": ROOT / "data" / "atlas" / "cc400" / "cc400_roi_atlas.nii.gz",
}
DERIV_1D = {"aal": "aal", "harvard_oxford": "ho", "cc200": "cc200", "cc400": "cc400"}
MIN_DISK_GB = 20        # di bawah ini subjek tidak dimulai (status "disk_penuh") agar tidak gagal massal


def ruang_disk_gb(path=None):
    """Sisa ruang disk (GB) pada drive folder Temp (tempat ANTs menulis berkas sementara)."""
    return shutil.disk_usage(path or tempfile.gettempdir()).free / 1e9


def path_t1(file_id):
    """Lokasi T1 mentah: data/raw/<folder situs>/<id 7 digit>/session_1/anat_1/mprage.nii.gz."""
    situs, sub = file_id.rsplit("_", 1)
    return ROOT / "data" / "raw" / situs / sub / "session_1" / "anat_1" / "mprage.nii.gz"


def urutan_roi(atlas, contoh_file_id):
    """Urutan label ROI persis seperti kolom time series PCP (= urutan simpul matriks FC)."""
    f = ROOT / "data" / "processed" / "time_series_global" / atlas / f"{contoh_file_id}_rois_{DERIV_1D[atlas]}.1D"
    header = f.open().readline().strip().lstrip("#").split()
    return [int(h.lstrip("#")) for h in header]


def sudah_selesai(file_id, out_dir=OUT_DIR):
    return (Path(out_dir) / "per_subjek" / f"{file_id}.npz").exists()


def init_worker(n_thread, seed=42):
    """Dipanggil sekali di setiap proses anak sebelum ANTs dimuat."""
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(n_thread)
    os.environ["ANTS_RANDOM_SEED"] = str(seed)


def _header_f32(img):
    """Bulatkan origin/spacing/direction ke presisi float32.

    ANTsPy menulis citra prior Atropos ke berkas NIfTI sementara (header float32), sedangkan citra
    intensitas tetap di memori (double). Pada citra yang sedikit miring (oblique; mis. Trinity, Leuven,
    Olin) selisih pembulatan ~1e-6 mm membuat ITK menolak dengan "Inputs do not occupy the same physical
    space". Dengan header float32 di semua citra, keduanya identik.
    """
    img = img.clone()
    img.set_origin(tuple(float(v) for v in np.float32(img.origin)))
    img.set_spacing(tuple(float(v) for v in np.float32(img.spacing)))
    img.set_direction(np.float32(img.direction).astype(float))
    return img


def _gambar_qc(n4, masker, label, file_id, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img, msk, lab = n4.numpy(), masker.numpy() > 0, label
    sumbu = int(np.argmax(img.shape))                    # potong di 3 posisi sepanjang sumbu terpanjang
    idx = np.where(msk.any(axis=tuple(a for a in range(3) if a != sumbu)))[0]
    posisi = [int(idx[0] + (idx[-1] - idx[0]) * f) for f in (0.35, 0.55, 0.75)]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, p in zip(axes, posisi):
        sl = [slice(None)] * 3
        sl[sumbu] = p
        ax.imshow(np.rot90(img[tuple(sl)]), cmap="gray")
        ax.imshow(np.rot90(np.ma.masked_where(lab[tuple(sl)] != 2, lab[tuple(sl)])), cmap="autumn", alpha=0.35)
        ax.contour(np.rot90(msk[tuple(sl)]), levels=[0.5], colors="c", linewidths=0.8)
        ax.axis("off")
    fig.suptitle(f"{file_id}  (merah = GM, biru = batas masker otak)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def proses_subjek(file_id, template_paths, roi_order, out_dir=OUT_DIR, interp_atlas="nearestNeighbor",
                  resolusi_mm=None, template_reg_mm=None, simpan_citra=False):
    """Memproses satu subjek dan menyimpan <out_dir>/per_subjek/<file_id>.npz + .json.

    template_paths : dict berkas ICBM152 2009 (kunci t1, mask, gm, wm, csf) dari nilearn.
    roi_order      : dict atlas -> list label ROI (urutan simpul FC).
    interp_atlas   : interpolasi label atlas 3 mm ke grid subjek ("nearestNeighbor" cepat;
                     "genericLabel" ~8x lebih lambat, 98.5% voxel identik).
    resolusi_mm    : ukuran voxel isotropik tempat T1 diproses (None = resolusi asli ±1 mm, DIPAKAI).
    template_reg_mm: resolusi template saat registrasi (None = template asli 1 mm, DIPAKAI).
                     Percobaan 1.5 mm / template 2 mm ditolak: volume GM per ROI hanya r = 0.89
                     terhadap versi 1 mm.
    Mengembalikan dict ringkas (status, durasi, ukuran QC). Error tidak menghentikan batch.
    """
    import ants

    out_dir = Path(out_dir)
    (out_dir / "per_subjek").mkdir(parents=True, exist_ok=True)
    (out_dir / "qc_gambar").mkdir(parents=True, exist_ok=True)
    waktu, t_mulai = {}, time.perf_counter()
    sisa_gb = ruang_disk_gb()
    if sisa_gb < MIN_DISK_GB:
        return {"FILE_ID": file_id, "status": "disk_penuh", "detik_total": 0.0,
                "pesan": f"sisa disk {sisa_gb:.1f} GB < {MIN_DISK_GB} GB, subjek tidak dimulai"}
    # Semua berkas sementara ANTsPy (warp ±180 MB, prior/probabilitas Atropos, hasil apply_transforms)
    # diarahkan ke satu folder khusus subjek ini, lalu folder itu dihapus utuh di blok finally.
    # Aman karena setiap proses anak hanya mengerjakan satu subjek pada satu waktu.
    tmp_lama = tempfile.tempdir
    tmp_subjek = tempfile.mkdtemp(prefix=f"ants_{file_id}_")
    tempfile.tempdir = tmp_subjek
    try:
        t0 = time.perf_counter()
        t1 = ants.image_read(str(path_t1(file_id)))
        if t1.dimension == 4:                            # beberapa berkas tersimpan 4D dengan 1 volume
            t1 = ants.slice_image(t1, axis=3, idx=0)
        dimensi_asli, voxel_asli = t1.shape, t1.spacing
        if resolusi_mm is not None:                      # interpolasi linear ke grid isotropik
            t1 = ants.resample_image(t1, (resolusi_mm,) * 3, use_voxels=False, interp_type=0)
        t1 = _header_f32(t1)
        vox_ml = float(np.prod(t1.spacing)) / 1000
        n4 = ants.n4_bias_field_correction(t1, shrink_factor=4)
        waktu["n4"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        template = ants.image_read(template_paths["t1"])
        masker_tpl = ants.image_read(template_paths["mask"])
        if template_reg_mm is not None:
            template = ants.resample_image(template, (template_reg_mm,) * 3, use_voxels=False, interp_type=0)
            masker_tpl = ants.resample_image(masker_tpl, (template_reg_mm,) * 3, use_voxels=False, interp_type=1)
        reg = ants.registration(fixed=template, moving=n4, type_of_transform="SyN")
        m_tpl = masker_tpl.numpy() > 0
        korelasi_tpl = float(np.corrcoef(reg["warpedmovout"].numpy()[m_tpl], template.numpy()[m_tpl])[0, 1])
        waktu["registrasi"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        masker = ants.apply_transforms(fixed=n4, moving=masker_tpl, transformlist=reg["invtransforms"],
                                       interpolator="nearestNeighbor")
        masker = ants.iMath(masker, "FillHoles")
        icv_L = float(masker.numpy().sum() * vox_ml / 1000)
        # potong ke kotak pembatas otak (+ margin 5 voxel) agar segmentasi & penarikan atlas lebih cepat
        n4 = _header_f32(ants.crop_image(n4, ants.iMath(masker, "MD", 5)))
        masker = ants.copy_image_info(n4, ants.resample_image_to_target(masker, n4, interp_type="nearestNeighbor"))
        otak = n4 * masker
        waktu["skull_stripping"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        prior = [ants.copy_image_info(n4, ants.apply_transforms(fixed=n4, moving=ants.image_read(template_paths[k]),
                                                                transformlist=reg["invtransforms"])) * masker
                 for k in ("csf", "gm", "wm")]
        seg = ants.atropos(a=otak, x=masker, i=prior, m="[0.2,1x1x1]", c="[5,0]", priorweight=0.25)
        label = seg["segmentation"].numpy().astype(np.int16)
        prob = {n: seg["probabilityimages"][k].numpy() * (label > 0)
                for k, n in enumerate(("csf", "gm", "wm"))}
        waktu["segmentasi"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        hasil = {}
        n_roi_kosong = {}
        for atlas, ids in roi_order.items():
            lab_atlas = ants.apply_transforms(fixed=n4, moving=ants.image_read(str(ATLAS_NII[atlas])),
                                              transformlist=reg["invtransforms"], interpolator=interp_atlas)
            a = lab_atlas.numpy().astype(np.int64).ravel()
            n_bin = max(max(ids), int(a.max())) + 1
            vol = np.bincount(a, minlength=n_bin) * vox_ml
            gm_soft = np.bincount(a, weights=prob["gm"].ravel(), minlength=n_bin) * vox_ml
            gm_hard = np.bincount(a, weights=(label == 2).ravel().astype(float), minlength=n_bin) * vox_ml
            ids = np.asarray(ids)
            hasil[f"gm_{atlas}"] = gm_soft[ids].astype(np.float32)
            hasil[f"gm_keras_{atlas}"] = gm_hard[ids].astype(np.float32)
            hasil[f"vol_{atlas}"] = vol[ids].astype(np.float32)
            n_roi_kosong[atlas] = int((vol[ids] == 0).sum())
        waktu["volume_roi"] = time.perf_counter() - t0

        wm = label == 3
        img_n4 = n4.numpy()
        ringkas = {
            "FILE_ID": file_id, "status": "ok", "pesan": "",
            "dimensi": "x".join(map(str, dimensi_asli)),
            "voxel_mm": "x".join(f"{s:.2f}" for s in voxel_asli),
            "resolusi_proses_mm": resolusi_mm if resolusi_mm is not None else "asli",
            "icv_L": icv_L,
            "gm_L": float(prob["gm"].sum() * vox_ml / 1000),
            "wm_L": float(prob["wm"].sum() * vox_ml / 1000),
            "csf_L": float(prob["csf"].sum() * vox_ml / 1000),
            "korelasi_template": korelasi_tpl,
            "cv_wm_pct": float(img_n4[wm].std() / img_n4[wm].mean() * 100) if wm.any() else float("nan"),
            **{f"roi_kosong_{a}": n for a, n in n_roi_kosong.items()},
            **{f"detik_{k}": round(v, 1) for k, v in waktu.items()},
            "detik_total": round(time.perf_counter() - t_mulai, 1),
        }
        _gambar_qc(n4, masker, label, file_id, out_dir / "qc_gambar" / f"{file_id}.png")
        if simpan_citra:
            d = out_dir / "citra" / file_id
            d.mkdir(parents=True, exist_ok=True)
            for nama, img in [("t1_n4", n4), ("masker_otak", masker), ("segmentasi", seg["segmentation"])]:
                ants.image_write(img, str(d / f"{nama}.nii.gz"))

        # tulis json dulu, npz terakhir (npz = penanda selesai), lewat berkas sementara agar tidak setengah jadi
        (out_dir / "per_subjek" / f"{file_id}.json").write_text(json.dumps(ringkas, indent=1))
        tmp = out_dir / "per_subjek" / f"{file_id}.tmp.npz"
        np.savez_compressed(tmp, **hasil)
        os.replace(tmp, out_dir / "per_subjek" / f"{file_id}.npz")
        (out_dir / "per_subjek" / f"{file_id}.gagal.txt").unlink(missing_ok=True)   # sisa kegagalan lama
        return ringkas
    except Exception as e:                               # noqa: BLE001 - satu subjek gagal tidak boleh menghentikan batch
        ringkas = {"FILE_ID": file_id, "status": "gagal", "pesan": f"{type(e).__name__}: {e}",
                   "detik_total": round(time.perf_counter() - t_mulai, 1)}
        (out_dir / "per_subjek" / f"{file_id}.gagal.txt").write_text(traceback.format_exc())
        return ringkas
    finally:
        # ANTs tidak menghapus berkas sementaranya sendiri; tanpa ini disk penuh setelah beberapa ratus
        # subjek (terjadi pada run 9 Oktober 2026: ±86 GB sisa berkas di Temp).
        tempfile.tempdir = tmp_lama
        shutil.rmtree(tmp_subjek, ignore_errors=True)
