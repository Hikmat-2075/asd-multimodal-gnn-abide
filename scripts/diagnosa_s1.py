"""Diagnosa skenario S1 (rs-fMRI, 1 atlas): membandingkan baseline klasik, MLP, dan GNN
dengan protokol evaluasi yang sama (fold tetap, normalisasi di dalam fold).

Pemakaian:  python scripts/diagnosa_s1.py --atlas aal --eksperimen lr mlp gatv2 gcn
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import ATLAS, bangun_graf, fisher_z, muat_fc, muat_kohort, segitiga_atas  # noqa: E402
from src.metrics import hitung_metrik, ringkas, teks_ringkas  # noqa: E402
from src.models import GNNGraf, MLP  # noqa: E402
from src.training import jalankan_cv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "result" / "metrics"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="aal")
    ap.add_argument("--eksperimen", nargs="+", default=["lr", "mlp", "gatv2", "gcn"])
    ap.add_argument("--top_k", type=float, default=0.2)
    args = ap.parse_args()

    kohort = muat_kohort()
    y = kohort["LABEL_BIN"].to_numpy()
    fc = muat_fc(args.atlas, kohort["FILE_ID"])
    z = fisher_z(fc)
    vec = segitiga_atas(z)
    n_roi = ATLAS[args.atlas]
    hasil = {}

    if "lr" in args.eksperimen:
        print(f"[{args.atlas}] Logistic Regression (L2) pada segitiga atas FC", flush=True)
        per_fold = []
        for k in range(5):
            tr, te = kohort["fold"] != k, kohort["fold"] == k
            sc = StandardScaler().fit(vec[tr])
            clf = LogisticRegression(C=0.01, max_iter=2000).fit(sc.transform(vec[tr]), y[tr])
            per_fold.append({"fold": k, **hitung_metrik(y[te], clf.predict_proba(sc.transform(vec[te]))[:, 1])})
        hasil["lr"] = ringkas(per_fold)
        print("  ", teks_ringkas(hasil["lr"]), flush=True)

    def dataset_vektor(i_penuh, i_latih, i_val, i_uji):
        sc = StandardScaler().fit(vec[i_penuh])
        v = sc.transform(vec).astype(np.float32)
        buat = lambda idx: [Data(x=torch.from_numpy(v[i][None]), y=torch.tensor([int(y[i])])) for i in idx]
        return buat(i_latih), buat(i_val), buat(i_uji)

    def dataset_graf(i_penuh, i_latih, i_val, i_uji):
        mu, sd = z[i_penuh].mean(0), z[i_penuh].std(0) + 1e-6
        x_node = ((z - mu) / sd).astype(np.float32)
        semua = bangun_graf(fc, x_node, y, top_k=args.top_k)
        ambil = lambda idx: [semua[i] for i in idx]
        return ambil(i_latih), ambil(i_val), ambil(i_uji)

    if "mlp" in args.eksperimen:
        print(f"[{args.atlas}] MLP pada segitiga atas FC", flush=True)
        r = jalankan_cv(kohort, dataset_vektor, lambda: MLP(vec.shape[1], hidden=128, dropout=0.5),
                        cfg={"lr": 1e-3, "weight_decay": 1e-2, "max_epochs": 100, "min_epochs": 10, "patience": 30})
        hasil["mlp"] = r["ringkasan"]

    for conv in ("gatv2", "gcn"):
        if conv in args.eksperimen:
            print(f"[{args.atlas}] GNN {conv} (top_k={args.top_k})", flush=True)
            r = jalankan_cv(kohort, dataset_graf,
                            lambda: GNNGraf(n_roi, hidden=64, n_layer=2, heads=2, conv=conv, readout="meanmax",
                                            dropout=0.5, dropout_input=0.2, dropedge=0.2),
                            cfg={"lr": 1e-3, "weight_decay": 1e-2, "max_epochs": 120, "min_epochs": 15, "patience": 30})
            hasil[conv] = r["ringkasan"]

    (OUT / f"diagnosa_s1_{args.atlas}.json").write_text(json.dumps(hasil, indent=2))
    print("\nRINGKASAN")
    for k, v in hasil.items():
        print(f"  {k:6s}: {teks_ringkas(v)}")


if __name__ == "__main__":
    main()
