"""Report e grafici della cross-validation.

Salva su disco (backend Agg, nessuna finestra):
  * confusion_matrix.png  — matrice di confusione sulle predizioni out-of-fold
  * roc_curve.png         — curva ROC out-of-fold con AUC e CI95
  * training_curves.png   — loss e AUC di validazione per epoca, per ogni fold
  * metrics.json          — metriche per fold + aggregate OOF + media CV

"Out-of-fold" (OOF): poiche' la StratifiedGroupKFold partiziona i pazienti,
ogni paziente e' predetto dal modello del fold in cui sta in validazione.
Concatenando le predizioni dei fold si ottiene una stima onesta su tutti i casi.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

_NEG, _POS = "Neg (N/T)", "Pos (P)"


def plot_confusion_matrix(tn, fp, fn, tp, path, title="Matrice di confusione (OOF)"):
    cm = np.array([[tn, fp], [fn, tp]], dtype=int)
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], [_NEG, _POS])
    ax.set_yticks([0, 1], [_NEG, _POS])
    ax.set_xlabel("Predetto"); ax.set_ylabel("Reale")
    thr = cm.max() / 2 if cm.max() else 0.5
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=15)
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ax.set_title(f"{title}\nSens={sens:.2f}  Spec={spec:.2f}")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_roc(y_true, y_prob, auc, ci, path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    label = f"AUC = {auc:.3f}"
    if ci and not any(np.isnan(ci)):
        label += f"  (CI95 {ci[0]:.3f}-{ci[1]:.3f})"
    ax.plot(fpr, tpr, lw=2, label=label)
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("1 - Specificita' (FPR)"); ax.set_ylabel("Sensibilita' (TPR)")
    ax.set_title("Curva ROC (out-of-fold)")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_training_curves(histories, path):
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 3.4), squeeze=False)
    for k, h in enumerate(histories):
        ax = axes[0][k]
        ep = range(1, len(h["val_auc"]) + 1)
        ax.plot(ep, h["train_loss"], label="train loss", color="tab:blue")
        ax.plot(ep, h["val_loss"], label="val loss", color="tab:orange")
        ax.set_xlabel("epoca"); ax.set_ylabel("loss"); ax.set_title(f"Fold {k}")
        ax2 = ax.twinx()
        ax2.plot(ep, h["val_auc"], label="val AUC", color="tab:green")
        ax2.set_ylabel("AUC"); ax2.set_ylim(0, 1.02)
        if k == 0:
            lines = ax.get_lines() + ax2.get_lines()
            ax.legend(lines, [l.get_label() for l in lines], loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_cv_report(fold_results, histories, oof_true, oof_prob, out_dir, compute_metrics):
    """Genera tutte le figure e il metrics.json. Ritorna le metriche OOF."""
    out_dir = Path(out_dir)
    oof = compute_metrics(np.asarray(oof_true), np.asarray(oof_prob))

    plot_confusion_matrix(oof.tn, oof.fp, oof.fn, oof.tp, out_dir / "confusion_matrix.png")
    plot_roc(oof_true, oof_prob, oof.auc, oof.auc_ci95, out_dir / "roc_curve.png")
    if histories:
        plot_training_curves(histories, out_dir / "training_curves.png")

    aucs = [r["metrics"].auc for r in fold_results]
    report = {
        "per_fold": [{"fold": r["fold"], **asdict(r["metrics"])} for r in fold_results],
        "cv_auc_mean": float(np.nanmean(aucs)),
        "cv_auc_std": float(np.nanstd(aucs)),
        "out_of_fold": asdict(oof),
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    return oof
