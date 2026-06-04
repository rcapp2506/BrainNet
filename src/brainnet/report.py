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
from sklearn.metrics import roc_curve, roc_auc_score

from .metrics import youden_threshold, threshold_for_sensitivity

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


def save_cv_report(fold_results, histories, oof_true, oof_prob, out_dir, compute_metrics,
                   target_sensitivity: float = 0.95):
    """Genera figure e metrics.json. Ritorna le metriche OOF a soglia 0.5."""
    out_dir = Path(out_dir)
    yt, yp = np.asarray(oof_true), np.asarray(oof_prob)

    oof = compute_metrics(yt, yp)                                  # soglia 0.5
    thr_y = youden_threshold(yt, yp)                               # bilanciata (Youden)
    oof_y = compute_metrics(yt, yp, threshold=thr_y)
    thr_c = threshold_for_sensitivity(yt, yp, target_sensitivity)  # clinica (max sensibilita')
    oof_c = compute_metrics(yt, yp, threshold=thr_c)

    plot_confusion_matrix(oof.tn, oof.fp, oof.fn, oof.tp,
                          out_dir / "confusion_matrix.png",
                          title="Matrice di confusione (OOF, soglia 0.5)")
    plot_confusion_matrix(oof_c.tn, oof_c.fp, oof_c.fn, oof_c.tp,
                          out_dir / "confusion_matrix_clinica.png",
                          title=f"OOF, punto clinico (sens>={target_sensitivity:.2f}, soglia={thr_c:.2f})")
    plot_roc(yt, yp, oof.auc, oof.auc_ci95, out_dir / "roc_curve.png",
             points=[(f"soglia 0.5: sens={oof.sensitivity:.2f} spec={oof.specificity:.2f}",
                      _operating_point(oof), "tab:red"),
                     (f"Youden {thr_y:.2f}: sens={oof_y.sensitivity:.2f} spec={oof_y.specificity:.2f}",
                      _operating_point(oof_y), "tab:orange"),
                     (f"clinico {thr_c:.2f}: sens={oof_c.sensitivity:.2f} spec={oof_c.specificity:.2f}",
                      _operating_point(oof_c), "tab:green")])
    if histories:
        plot_training_curves(histories, out_dir / "training_curves.png")

    aucs = [r["metrics"].auc for r in fold_results]
    report = {
        "per_fold": [{"fold": r["fold"], **asdict(r["metrics"])} for r in fold_results],
        "cv_auc_mean": float(np.nanmean(aucs)),
        "cv_auc_std": float(np.nanstd(aucs)),
        "out_of_fold": asdict(oof),
        "out_of_fold_youden": asdict(oof_y),
        "out_of_fold_clinical": asdict(oof_c),
        "youden_threshold": thr_y,
        "clinical_threshold": thr_c,
        "target_sensitivity": target_sensitivity,
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))

    print(f"  Soglia Youden={thr_y:.3f} | Soglia clinica (sens>={target_sensitivity:.2f})={thr_c:.3f}")
    print(f"  A 0.5:     sens={oof.sensitivity:.3f} spec={oof.specificity:.3f} acc={oof.accuracy:.3f}")
    print(f"  A Youden:  sens={oof_y.sensitivity:.3f} spec={oof_y.specificity:.3f} acc={oof_y.accuracy:.3f}")
    print(f"  Clinico:   sens={oof_c.sensitivity:.3f} spec={oof_c.specificity:.3f} acc={oof_c.accuracy:.3f}")
    return oof


# ─────────────────────── analisi multi-seed (varianza + fasce) ───────────────────────

def plot_roc_band(oof_list, path):
    """oof_list: lista di (y_true, y_prob), una per seed. Banda media ± std della ROC."""
    grid = np.linspace(0, 1, 101)
    tprs, aucs = [], []
    for yt, yp in oof_list:
        fpr, tpr, _ = roc_curve(yt, yp)
        ti = np.interp(grid, fpr, tpr); ti[0] = 0.0
        tprs.append(ti); aucs.append(roc_auc_score(yt, yp))
    tprs = np.array(tprs)
    mean_tpr, std_tpr = tprs.mean(0), tprs.std(0)
    mean_tpr[-1] = 1.0
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    ax.plot(grid, mean_tpr, color="tab:blue", lw=2,
            label=f"ROC media (AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f})")
    ax.fill_between(grid, np.clip(mean_tpr - std_tpr, 0, 1), np.clip(mean_tpr + std_tpr, 0, 1),
                    color="tab:blue", alpha=0.2, label="± 1 dev. std")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("1 - Specificita' (FPR)"); ax.set_ylabel("Sensibilita' (TPR)")
    ax.set_title(f"ROC media su {len(oof_list)} seed (out-of-fold)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_auc_distribution(aucs, path):
    aucs = np.asarray(aucs, float)
    fig, ax = plt.subplots(figsize=(4.0, 4.4))
    ax.boxplot(aucs, vert=True, widths=0.5, showmeans=True)
    ax.scatter(np.random.default_rng(0).normal(1, 0.04, len(aucs)), aucs,
               color="tab:blue", alpha=0.6, zorder=3)
    ax.set_xticks([1], ["OOF AUC"])
    ax.set_ylabel("AUC")
    ax.set_title(f"Distribuzione AUC su {len(aucs)} seed\n"
                 f"media={aucs.mean():.3f} ± {aucs.std():.3f}")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_metric_bands(metric_dict, path):
    """metric_dict: {nome: [valori per seed]} -> barre media ± dev. std."""
    names = list(metric_dict)
    means = [float(np.mean(metric_dict[n])) for n in names]
    stds = [float(np.std(metric_dict[n])) for n in names]
    fig, ax = plt.subplots(figsize=(1.5 * len(names) + 1.5, 4.0))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=5, color="tab:blue", alpha=0.75)
    for xi, m, s in zip(x, means, stds):
        ax.text(xi, min(m + s + 0.03, 1.04), f"{m:.2f}±{s:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, names, rotation=15)
    ax.set_ylim(0, 1.08); ax.set_ylabel("valore")
    ax.set_title("Metriche OOF: media ± dev. std sui seed")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_multiseed_report(per_seed, out_dir, compute_metrics, youden_threshold,
                          target_sensitivity: float = 0.95):
    """per_seed: lista di {seed, oof_true, oof_prob}. Salva varianza e grafici a fasce."""
    out_dir = Path(out_dir)
    oof_list = [(np.asarray(s["oof_true"]), np.asarray(s["oof_prob"])) for s in per_seed]

    rows = []
    for s, (yt, yp) in zip(per_seed, oof_list):
        m05 = compute_metrics(yt, yp)
        thr_y = youden_threshold(yt, yp)
        my = compute_metrics(yt, yp, threshold=thr_y)
        rows.append({"seed": s["seed"], "auc": m05.auc,
                     "sens_05": m05.sensitivity, "spec_05": m05.specificity,
                     "youden_threshold": thr_y,
                     "sens_youden": my.sensitivity, "spec_youden": my.specificity})

    def stats(key):
        v = np.array([r[key] for r in rows], float)
        return {"mean": float(np.mean(v)), "std": float(np.std(v)),
                "min": float(np.min(v)), "max": float(np.max(v)),
                "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}

    summary = {k: stats(k) for k in
               ("auc", "sens_05", "spec_05", "sens_youden", "spec_youden")}

    plot_roc_band(oof_list, out_dir / "roc_band.png")
    plot_auc_distribution([r["auc"] for r in rows], out_dir / "auc_distribution.png")
    plot_metric_bands({"AUC": [r["auc"] for r in rows],
                       "sens@0.5": [r["sens_05"] for r in rows],
                       "spec@0.5": [r["spec_05"] for r in rows],
                       "sens@You": [r["sens_youden"] for r in rows],
                       "spec@You": [r["spec_youden"] for r in rows]},
                      out_dir / "metrics_band.png")

    (out_dir / "multiseed_metrics.json").write_text(
        json.dumps({"n_seeds": len(rows), "per_seed": rows, "summary": summary}, indent=2))
    return summary
