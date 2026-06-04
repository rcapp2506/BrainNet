"""Report e grafici della cross-validation.

Salva su disco (backend Agg, nessuna finestra):
  * confusion_matrix.png         — matrice di confusione OOF a soglia 0.5
  * confusion_matrix_youden.png  — matrice di confusione OOF alla soglia ottimale
  * roc_curve.png                — ROC OOF con AUC, CI95 e i due punti operativi
  * training_curves.png          — loss e AUC val per epoca, asse x comune, best marcato
  * metrics.json                 — metriche per fold + OOF (0.5 e Youden) + media CV

"Out-of-fold" (OOF): poiche' la StratifiedGroupKFold partiziona i pazienti,
ogni paziente e' predetto dal modello del fold in cui sta in validazione.
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

from .metrics import youden_threshold

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


def plot_roc(y_true, y_prob, auc, ci, path, points=None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    label = f"AUC = {auc:.3f}"
    if ci and not any(np.isnan(ci)):
        label += f"  (CI95 {ci[0]:.3f}-{ci[1]:.3f})"
    ax.plot(fpr, tpr, lw=2, label=label, color="tab:blue")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    for name, (x, y), color in (points or []):
        ax.scatter([x], [y], s=50, color=color, zorder=5, label=name)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("1 - Specificita' (FPR)"); ax.set_ylabel("Sensibilita' (TPR)")
    ax.set_title("Curva ROC (out-of-fold)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_training_curves(histories, path):
    n = len(histories)
    max_ep = max((len(h["val_auc"]) for h in histories), default=1)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 3.4), squeeze=False)
    for k, h in enumerate(histories):
        ax = axes[0][k]
        ep = range(1, len(h["val_auc"]) + 1)
        l1, = ax.plot(ep, h["train_loss"], label="train loss", color="tab:blue")
        l2, = ax.plot(ep, h["val_loss"], label="val loss", color="tab:orange")
        ax.set_xlim(1, max_ep)                       # asse x comune a tutti i fold
        ax.set_xlabel("epoca"); ax.set_ylabel("loss"); ax.set_title(f"Fold {k}")
        ax2 = ax.twinx()
        l3, = ax2.plot(ep, h["val_auc"], label="val AUC", color="tab:green")
        ax2.set_ylabel("AUC"); ax2.set_ylim(0, 1.02)
        if h["val_auc"]:                             # epoca del best (modello salvato)
            best = int(np.argmax(h["val_auc"])) + 1
            ax.axvline(best, color="gray", ls="--", lw=1)
            ax.text(best, ax.get_ylim()[1], f" best {best}", color="gray",
                    fontsize=7, va="top")
        if k == 0:
            ax.legend([l1, l2, l3], [l1.get_label(), l2.get_label(), l3.get_label()],
                      loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _operating_point(m):
    fpr = m.fp / (m.fp + m.tn) if (m.fp + m.tn) else 0.0
    tpr = m.tp / (m.tp + m.fn) if (m.tp + m.fn) else 0.0
    return fpr, tpr


def save_cv_report(fold_results, histories, oof_true, oof_prob, out_dir, compute_metrics):
    """Genera figure e metrics.json. Ritorna le metriche OOF a soglia 0.5."""
    out_dir = Path(out_dir)
    yt, yp = np.asarray(oof_true), np.asarray(oof_prob)

    oof = compute_metrics(yt, yp)                                  # soglia 0.5
    thr = youden_threshold(yt, yp)                                 # soglia ottimale
    oof_opt = compute_metrics(yt, yp, threshold=thr)               # metriche al punto Youden

    plot_confusion_matrix(oof.tn, oof.fp, oof.fn, oof.tp,
                          out_dir / "confusion_matrix.png",
                          title="Matrice di confusione (OOF, soglia 0.5)")
    plot_confusion_matrix(oof_opt.tn, oof_opt.fp, oof_opt.fn, oof_opt.tp,
                          out_dir / "confusion_matrix_youden.png",
                          title=f"Matrice di confusione (OOF, soglia Youden={thr:.2f})")
    plot_roc(yt, yp, oof.auc, oof.auc_ci95, out_dir / "roc_curve.png",
             points=[(f"soglia 0.5: sens={oof.sensitivity:.2f} spec={oof.specificity:.2f}",
                      _operating_point(oof), "tab:red"),
                     (f"Youden {thr:.2f}: sens={oof_opt.sensitivity:.2f} spec={oof_opt.specificity:.2f}",
                      _operating_point(oof_opt), "tab:green")])
    if histories:
        plot_training_curves(histories, out_dir / "training_curves.png")

    aucs = [r["metrics"].auc for r in fold_results]
    report = {
        "per_fold": [{"fold": r["fold"], **asdict(r["metrics"])} for r in fold_results],
        "cv_auc_mean": float(np.nanmean(aucs)),
        "cv_auc_std": float(np.nanstd(aucs)),
        "out_of_fold": asdict(oof),
        "out_of_fold_youden": asdict(oof_opt),
        "youden_threshold": thr,
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))

    print(f"  Soglia ottimale (Youden) = {thr:.3f}")
    print(f"  A 0.5:    sens={oof.sensitivity:.3f} spec={oof.specificity:.3f} acc={oof.accuracy:.3f}")
    print(f"  A Youden: sens={oof_opt.sensitivity:.3f} spec={oof_opt.specificity:.3f} acc={oof_opt.accuracy:.3f}")
    return oof
