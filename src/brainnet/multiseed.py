"""Campagna multi-seed sul classico (analisi della varianza + grafici a fasce).

Ripete l'intera 5-fold CV con R seed diversi (ogni seed cambia split, init pesi
e ordine dati), poi aggrega:
  * varianza di AUC e dei punti operativi (media, dev. std, min/max, CI95);
  * roc_band.png        — ROC media ± 1 dev. std sui seed;
  * auc_distribution.png — boxplot/strip dell'AUC sui seed;
  * metrics_band.png    — barre media ± std (AUC, sens/spec a 0.5 e a Youden);
  * multiseed_metrics.json — tutti i numeri (per-seed + riepilogo).

Uso:
    python -m brainnet.multiseed --seeds 10
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from .config import Config
from .data import build_dataframe
from .metrics import compute_metrics, youden_threshold
from .train import _run_cv
from .report import save_multiseed_report


def run_multiseed(cfg: Config, n_seeds: int = 10, base_seed: int = 0) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(cfg.output_dir / "config.json")

    df = build_dataframe(cfg.data)
    seeds = [base_seed + 100 * i + 1 for i in range(n_seeds)]   # 1, 101, 201, ...
    print(f"Device: {device.type.upper()}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}")
    print(f"Campagna multi-seed: {n_seeds} run | seed = {seeds}")
    print(f"Output in: {cfg.output_dir}/\n")

    per_seed = []
    for i, seed in enumerate(seeds):
        print(f"\n===== SEED {seed}  ({i + 1}/{n_seeds}) =====", flush=True)
        _, _, oof_true, oof_prob = _run_cv(cfg, df, seed, checkpoint_dir=None)
        oof = compute_metrics(np.asarray(oof_true), np.asarray(oof_prob))
        print(f"[seed {seed}] OOF AUC={oof.auc:.3f} "
              f"sens={oof.sensitivity:.3f} spec={oof.specificity:.3f}", flush=True)
        per_seed.append({"seed": seed, "oof_true": oof_true, "oof_prob": oof_prob})

    summary = save_multiseed_report(per_seed, cfg.output_dir, compute_metrics,
                                    youden_threshold,
                                    target_sensitivity=cfg.train.target_sensitivity)

    a, sy, py = summary["auc"], summary["sens_youden"], summary["spec_youden"]
    print(f"\n===== RIEPILOGO {n_seeds} SEED =====")
    print(f"AUC OOF:  {a['mean']:.3f} ± {a['std']:.3f}  "
          f"(min {a['min']:.3f}, max {a['max']:.3f}, CI95 {a['ci95'][0]:.3f}-{a['ci95'][1]:.3f})")
    print(f"A Youden: sens {sy['mean']:.3f} ± {sy['std']:.3f} | spec {py['mean']:.3f} ± {py['std']:.3f}")
    print(f"Grafici a fasce in {cfg.output_dir}/: "
          f"roc_band.png, auc_distribution.png, metrics_band.png")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Campagna multi-seed sul classico (varianza + grafici a fasce).")
    ap.add_argument("--seeds", type=int, default=10, help="numero di seed (default 10)")
    ap.add_argument("--base-seed", type=int, default=0)
    args = ap.parse_args()
    run_multiseed(Config(), n_seeds=args.seeds, base_seed=args.base_seed)
