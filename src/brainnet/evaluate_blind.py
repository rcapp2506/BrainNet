"""Valutazione sul test cieco (golden set indipendente).

Disciplina del test cieco: questa coorte NON entra mai nella CV, nel training o
nella selezione del modello. Si valuta UNA SOLA VOLTA, alla fine, sul modello
gia' scelto. Nessun tuning sulle sue metriche.

Strategia: ensemble dei modelli dei fold della CV. Per ogni fold si caricano i
pesi salvati durante il training, si predice sul test cieco, e si MEDIANO le
probabilita' tra i fold (ensemble). Le metriche finali si calcolano una volta
sola sulle probabilita' mediate.

Modalita':
  classical  CNN 3D (build_model + ModelConfig); pesi  fold{f}_best.pt
  quantum    rete ibrida quanvoluzionale; pesi          qfold{f}_best.pt
             (input slice-come-canali, come in train_quantum)

Uso:
    python -m brainnet.evaluate_blind --run-dir runs --mode classical
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data import build_dataframe, make_eval_dataset
from .metrics import compute_metrics
from .progress import pbar


def _test_config(cfg: Config, quantum: bool) -> Config:
    """Config che punta al set di test, con il layout coerente alla modalita'."""
    data = dataclasses.replace(
        cfg.data,
        labels_csv=cfg.data.test_labels_csv,
        dicom_subdir=cfg.data.test_dicom_subdir,
        channels_as_slices=quantum,
        img_size=cfg.quantum.img_size if quantum else cfg.data.img_size,
    )
    return dataclasses.replace(cfg, data=data)


def _build_model(cfg: Config, quantum: bool, seed: int = 42):
    if quantum:
        from .quantum.engine import BackendManager
        from .quantum.hybrid import HybridQuanvNet
        bm = BackendManager(cfg.quantum.backend_type, seed=seed).initialize()
        return HybridQuanvNet(cfg.quantum, bm)
    from .model import build_model
    return build_model(cfg.model)


@torch.no_grad()
def _predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval().to(device)
    probs, labels = [], []
    for batch in pbar(loader, desc="test cieco"):
        x = batch["image"].to(device)
        logits = model(x)
        probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
        labels.append(batch["label"].numpy())
    return np.concatenate(probs), np.concatenate(labels)


def evaluate_blind(cfg: Config, run_dir: Path, mode: str = "classical") -> dict:
    quantum = (mode == "quantum")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_test = _test_config(cfg, quantum)
    df = build_dataframe(cfg_test.data)
    ds = make_eval_dataset(df, cfg_test)
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.train.batch_size,
                                         shuffle=False, num_workers=cfg.train.num_workers)

    prefix = "qfold" if quantum else "fold"
    ckpts = sorted(run_dir.glob(f"{prefix}*_best.pt"))
    if not ckpts:
        raise FileNotFoundError(
            f"Nessun checkpoint '{prefix}*_best.pt' in {run_dir}. "
            f"Esegui prima il training ({'quantistico' if quantum else 'classico'}).")

    # Ensemble: media delle probabilita' sui fold. Le etichette sono identiche
    # tra i fold (stesso loader deterministico), quindi le prendo una volta.
    fold_probs, labels = [], None
    for ck in ckpts:
        model = _build_model(cfg, quantum)
        model.load_state_dict(torch.load(ck, map_location=device))
        p, y = _predict(model, loader, device)
        fold_probs.append(p)
        labels = y
        print(f"  caricato {ck.name}")

    ens = np.mean(np.stack(fold_probs, axis=0), axis=0)
    m = compute_metrics(labels, ens)

    report = {
        "mode": mode,
        "n_test": int(len(labels)),
        "n_folds_ensembled": len(ckpts),
        "checkpoints": [c.name for c in ckpts],
        "accuracy": m.accuracy, "sensitivity": m.sensitivity,
        "specificity": m.specificity, "precision": m.precision,
        "auc": m.auc, "auc_ci95": list(m.auc_ci95),
        "confusion": {"tn": m.tn, "fp": m.fp, "fn": m.fn, "tp": m.tp},
    }
    (run_dir / f"blind_test_{mode}.json").write_text(json.dumps(report, indent=2))

    print(f"\n{'='*60}\n  TEST CIECO ({mode}) — valutazione unica\n{'='*60}")
    print(f"  n = {report['n_test']} pazienti | ensemble di {len(ckpts)} fold")
    print(f"  AUC  = {m.auc:.3f}  (CI95 {m.auc_ci95[0]:.3f}-{m.auc_ci95[1]:.3f})")
    print(f"  Acc  = {m.accuracy:.3f} | Sens = {m.sensitivity:.3f} | Spec = {m.specificity:.3f}")
    print(f"  Matrice: TN={m.tn} FP={m.fp} FN={m.fn} TP={m.tp}")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, default=Path("runs"),
                   help="cartella con i checkpoint dei fold")
    p.add_argument("--mode", choices=["classical", "quantum"], default="classical")
    args = p.parse_args()

    cfg = Config()
    cfg_json = args.run_dir / "config.json"
    if cfg_json.exists():
        print(f"(uso la config salvata in {cfg_json})")
    evaluate_blind(cfg, args.run_dir, args.mode)


if __name__ == "__main__":
    main()
