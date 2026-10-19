"""Metrik klasifikasi biner (1 = ASD)."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def hitung_metrik(y_true, p_asd, ambang=0.5):
    y_pred = (np.asarray(p_asd) >= ambang).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "auc": roc_auc_score(y_true, p_asd),
    }


def ringkas(per_fold):
    """Rata-rata ± simpangan baku antar fold (dalam persen)."""
    df = pd.DataFrame(per_fold)
    kolom = ["accuracy", "sensitivity", "specificity", "precision", "f1", "auc"]
    return {k: (df[k].mean() * 100, df[k].std(ddof=0) * 100) for k in kolom}


def teks_ringkas(r):
    return " | ".join(f"{k[:3].upper()} {m:.2f}±{s:.2f}" for k, (m, s) in r.items())
