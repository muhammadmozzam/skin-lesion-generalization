import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, brier_score_loss)


def ece(y, p, n_bins=15):
    """Expected calibration error for the predicted class's confidence."""
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    correct = (pred == y).astype(float)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def all_metrics(y, p, thr=0.5):
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    tp = ((pred == 1) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    return {
        "n": int(len(y)),
        "acc": accuracy_score(y, pred),
        "bal_acc": balanced_accuracy_score(y, pred),
        "auc": roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"),
        "f1": f1_score(y, pred, zero_division=0),
        "sensitivity": tp / max(1, (y == 1).sum()),
        "specificity": tn / max(1, (y == 0).sum()),
        "ece": ece(y, p),
        "brier": brier_score_loss(y, p),
    }
