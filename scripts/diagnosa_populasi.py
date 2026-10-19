"""Diagnosa model graf populasi (Parisot dkk. 2018 / dasar DAGMNet) pada FC satu atlas.

Pemakaian:  python scripts/diagnosa_populasi.py --atlas aal --model cheb gat
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import fisher_z, muat_fc, muat_kohort, segitiga_atas  # noqa: E402
from src.metrics import teks_ringkas  # noqa: E402
from src.populasi import GATPopulasi, GCNPopulasi, jalankan_cv_populasi  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "result" / "metrics"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="aal")
    ap.add_argument("--model", nargs="+", default=["cheb", "gat"])
    ap.add_argument("--n_fitur", type=int, default=2000)
    ap.add_argument("--k", type=int, default=0, help="k tetangga terdekat (0 = graf penuh berbobot)")
    args = ap.parse_args()

    kohort = muat_kohort()
    X = segitiga_atas(fisher_z(muat_fc(args.atlas, kohort["FILE_ID"])))
    hasil = {}
    pembuat = {"cheb": lambda d: GCNPopulasi(d, hidden=16, n_layer=2, K=3, dropout=0.3),
               "gat": lambda d: GATPopulasi(d, hidden=32, heads=4, dropout=0.3)}
    for nama in args.model:
        print(f"[{args.atlas}] graf populasi {nama} (n_fitur={args.n_fitur}, k={args.k or 'penuh'})", flush=True)
        r = jalankan_cv_populasi(kohort, X, pembuat[nama], n_fitur=args.n_fitur, k_tetangga=args.k or None)
        hasil[nama] = r["ringkasan"]
    (OUT / f"diagnosa_populasi_{args.atlas}.json").write_text(json.dumps(hasil, indent=2))
    print("\nRINGKASAN")
    for k, v in hasil.items():
        print(f"  {k:5s}: {teks_ringkas(v)}")


if __name__ == "__main__":
    main()
