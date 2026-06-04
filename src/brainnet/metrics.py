"""Metriche di valutazione corrette.

Corregge l'errore del notebook originale, dove l'assegnazione di TP/TN/FP/FN
dalla matrice di confusione era incoerente tra le celle e non rispettava la
convenzione sklearn (riga = vero, colonna = predetto).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


@dataclass
class ClassificationMetrics:
    accuracy: float
    sensitivity: float        # recall sulla classe positiva (Parkinson)
    specificity: float
    precision: float
    auc: float
    auc_ci95: tuple[float, float]
    tn: int
    fp: int
    fn: int
    tp: int


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    n_boot: int = 2000, seed: int = 0) -> ClassificationMetrics:
    """y_true: {0,1}; y_prob: probabilita' della classe positiva."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= 0.5).astype(int)

    # labels=[0,1] fissa l'ordine: convenzione sklearn riga=vero, colonna=predetto.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    lo, hi = _bootstrap_auc_ci(y_true, y_prob, n_boot=n_boot, seed=seed)

    return ClassificationMetrics(
        accuracy=acc, sensitivity=sens, specificity=spec, precision=prec,
        auc=auc, auc_ci95=(lo, hi), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def _bootstrap_auc_ci(y_true, y_prob, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if not aucs:
        return (float("nan"), float("nan"))
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))


def class_weights_from_labels(labels, n_classes: int = 2):
    """Pesi inversamente proporzionali alla frequenza di classe (per la loss).

    w[c] = N / (n_classes * count[c]); classi assenti -> peso neutro.
    Calcolare SEMPRE solo sul training (per-fold), mai su validation/test.
    """
    import numpy as np
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    return counts.sum() / (n_classes * counts)
