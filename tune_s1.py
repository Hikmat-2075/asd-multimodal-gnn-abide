# -*- coding: utf-8 -*-
"""Seleksi konfigurasi Skenario 1 HANYA dengan validasi internal (fold uji tidak disentuh)."""
import copy, random, time, json
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.utils import dropout_edge
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score

DEV = torch.device("cuda")
df = pd.read_csv("data/processed/kohort_final.csv")
y_all = (df["DX_GROUP"] == 1).astype(int).values
folds = df["fold"].values

def load(atlas):
    R = np.stack([np.load(f"data/processed/fc_matrices_global/{atlas}/{f}_fc.npy") for f in df["FILE_ID"]]).astype(np.float32)
    Z = np.arctanh(np.clip(R, -0.999999, 0.999999))
    for z in Z: np.fill_diagonal(z, 0.0)
    n = R.shape[1]; iu = np.triu_indices(n, 1); E, W = [], []
    for r in R:
        a = np.abs(r[iu]); keep = a >= np.quantile(a, 0.9)
        s, d = iu[0][keep], iu[1][keep]
        E.append(torch.tensor(np.vstack([np.r_[s, d], np.r_[d, s]]), dtype=torch.long))
        W.append(torch.tensor(np.r_[r[s, d], r[s, d]], dtype=torch.float32).unsqueeze(1))
    return Z, E, W

class Net(nn.Module):
    def __init__(s, n, hid, heads, drop, dropedge):
        super().__init__()
        s.c1 = GATv2Conv(n, hid, heads=heads, edge_dim=1, dropout=drop); s.b1 = nn.BatchNorm1d(hid * heads)
        s.c2 = GATv2Conv(hid * heads, hid, heads=heads, edge_dim=1, dropout=drop); s.b2 = nn.BatchNorm1d(hid * heads)
        s.pool = AttentionalAggregation(nn.Sequential(nn.Linear(hid * heads, 16), nn.Tanh(), nn.Linear(16, 1)))
        s.head = nn.Sequential(nn.Dropout(drop), nn.Linear(hid * heads, 2))
        s.drop, s.de = drop, dropedge
    def forward(s, d):
        ei, ea = d.edge_index, d.edge_attr
        if s.training and s.de > 0:
            ei, m = dropout_edge(ei, p=s.de, force_undirected=False); ea = ea[m]
        x = F.dropout(d.x, s.drop, s.training)
        x = F.elu(s.b1(s.c1(x, ei, ea))); x = F.dropout(x, s.drop, s.training)
        x = F.elu(s.b2(s.c2(x, ei, ea)))
        return s.head(s.pool(x, d.batch))

CONFIGS = {
 "A_aal_raw_kecil":   dict(atlas="aal",   std=False, hid=16, heads=2, drop=0.5, de=0.2, lr=1e-4, wd=1e-2),
 "B_aal_std_kecil":   dict(atlas="aal",   std=True,  hid=16, heads=2, drop=0.5, de=0.2, lr=1e-4, wd=1e-2),
 "C_aal_raw_sedang":  dict(atlas="aal",   std=False, hid=32, heads=4, drop=0.5, de=0.2, lr=1e-4, wd=1e-2),
 "D_cc200_raw_kecil": dict(atlas="cc200", std=False, hid=16, heads=2, drop=0.5, de=0.2, lr=1e-4, wd=1e-2),
 "E_cc200_raw_sedang":dict(atlas="cc200", std=False, hid=32, heads=4, drop=0.5, de=0.2, lr=1e-4, wd=1e-2),
}
cache = {}
rows = []
for name, c in CONFIGS.items():
    if c["atlas"] not in cache: cache[c["atlas"]] = load(c["atlas"])
    Z, E, W = cache[c["atlas"]]; n = Z.shape[1]
    t0 = time.time()
    for f in range(5):
        seed = 42 + f; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        trf = np.where(folds != f)[0]                       # fold uji f TIDAK dipakai sama sekali
        a, b = next(StratifiedShuffleSplit(1, test_size=0.15, random_state=seed).split(trf, y_all[trf]))
        tr, va = trf[a], trf[b]
        mu = Z[trf].mean(0) if c["std"] else 0.0; sd = Z[trf].std(0) + 1e-6 if c["std"] else 1.0
        mk = lambda idx: [Data(x=torch.tensor((Z[i] - mu) / sd, dtype=torch.float32), edge_index=E[i], edge_attr=W[i],
                               y=torch.tensor([y_all[i]])) for i in idx]
        Ltr = DataLoader(mk(tr), batch_size=32, shuffle=True); Lva = DataLoader(mk(va), batch_size=256)
        m = Net(n, c["hid"], c["heads"], c["drop"], c["de"]).to(DEV)
        opt = torch.optim.Adam(m.parameters(), lr=c["lr"], weight_decay=c["wd"])
        best = (1e9, 0, 0, 0); wait = 0
        for ep in range(1, 201):
            m.train()
            for d in Ltr:
                d = d.to(DEV); opt.zero_grad(); F.cross_entropy(m(d), d.y).backward(); opt.step()
            m.eval(); P, Y, L = [], [], 0
            with torch.no_grad():
                for d in Lva:
                    d = d.to(DEV); o = m(d); L += F.cross_entropy(o, d.y, reduction="sum").item()
                    P.append(F.softmax(o, 1)[:, 1].cpu()); Y.append(d.y.cpu())
            P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy(); L /= len(Y)
            if L < best[0] - 1e-4: best, wait = (L, accuracy_score(Y, P >= .5), roc_auc_score(Y, P), ep), 0
            else:
                wait += 1
                if wait >= 30: break
        rows.append(dict(config=name, fold=f, val_loss=best[0], val_acc=best[1], val_auc=best[2], best_ep=best[3]))
    r = pd.DataFrame([x for x in rows if x["config"] == name])
    print(f"{name:20s} | val ACC {r.val_acc.mean()*100:5.2f} | val AUC {r.val_auc.mean()*100:5.2f} | val loss {r.val_loss.mean():.3f} | best ep {r.best_ep.tolist()} | {time.time()-t0:.0f}s", flush=True)

out = pd.DataFrame(rows); out.to_csv("result/tabel/10_seleksi_konfigurasi_validasi_internal.csv", index=False)
print(out.groupby("config")[["val_acc", "val_auc", "val_loss"]].mean().sort_values("val_auc", ascending=False))
