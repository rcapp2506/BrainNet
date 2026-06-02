"""Addestramento con cross-validation a livello di paziente.

Punti centrali della reingegnerizzazione:
  * StratifiedGroupKFold: ogni paziente (GUID) sta in un solo fold; le classi
    restano bilanciate tra i fold. Niente leakage tra train e validation.
  * augmentation solo nel DataLoader di training.
  * early stopping sull'AUC di validazione, AdamW + cosine annealing.
  * seeding deterministico.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold

from .config import Config
from .data import build_dataframe, make_datasets
from .model import build_model
from .metrics import compute_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _run_epoch(model, loader, device, optimizer=None, scaler=None, amp=False):
    train = optimizer is not None
    model.train(train)
    crit = nn.CrossEntropyLoss()
    losses, probs, labels = [], [], []

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast(
            device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(x)
            loss = crit(logits, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
        losses.append(loss.item())
        probs.append(torch.softmax(logits.detach().float(), dim=1)[:, 1].cpu().numpy())
        labels.append(y.cpu().numpy())

    return float(np.mean(losses)), np.concatenate(probs), np.concatenate(labels)


def train_cv(cfg: Config) -> list[dict]:
    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(cfg.output_dir / "config.json")

    df = build_dataframe(cfg.data)
    skf = StratifiedGroupKFold(n_splits=cfg.train.n_folds, shuffle=True,
                               random_state=cfg.train.seed)

    fold_results = []
    for fold, (tr, va) in enumerate(skf.split(df, df["label"], groups=df["group"])):
        df_tr, df_va = df.iloc[tr], df.iloc[va]
        ds_tr, ds_va = make_datasets(df_tr, df_va, cfg)

        dl_tr = DataLoader(ds_tr, batch_size=cfg.train.batch_size, shuffle=True,
                           num_workers=cfg.train.num_workers, drop_last=False)
        dl_va = DataLoader(ds_va, batch_size=cfg.train.batch_size, shuffle=False,
                           num_workers=cfg.train.num_workers)

        model = build_model(cfg.model).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                weight_decay=cfg.train.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train.max_epochs)
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.amp and device.type == "cuda")

        best_auc, best_state, patience = -1.0, None, 0
        for epoch in range(cfg.train.max_epochs):
            _run_epoch(model, dl_tr, device, optimizer=opt, scaler=scaler, amp=cfg.train.amp)
            sched.step()
            _, vp, vy = _run_epoch(model, dl_va, device, amp=cfg.train.amp)
            m = compute_metrics(vy, vp, n_boot=0)
            if m.auc > best_auc:
                best_auc, best_state, patience = m.auc, {
                    k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
            else:
                patience += 1
                if patience >= cfg.train.early_stop_patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
            torch.save(best_state, cfg.output_dir / f"fold{fold}_best.pt")

        _, vp, vy = _run_epoch(model, dl_va, device, amp=cfg.train.amp)
        final = compute_metrics(vy, vp)
        fold_results.append({"fold": fold, "metrics": final})
        print(f"[fold {fold}] AUC={final.auc:.3f} "
              f"CI95={final.auc_ci95[0]:.3f}-{final.auc_ci95[1]:.3f} "
              f"sens={final.sensitivity:.3f} spec={final.specificity:.3f}")

    aucs = [r["metrics"].auc for r in fold_results]
    print(f"\nAUC media CV: {np.nanmean(aucs):.3f} +/- {np.nanstd(aucs):.3f}")
    return fold_results


if __name__ == "__main__":
    train_cv(Config())
