"""Addestramento e confronto appaiato ibrido-vs-classico sul dataset PET.

Riproduce sul nuovo dominio la metodologia del Cap. 3 della tesi:
  * CV a livello di PAZIENTE (StratifiedGroupKFold): nessun leakage tra fold;
  * campagna MULTI-SEED (R run per fold) per stimare la variabilita';
  * confronto APPAIATO HybridQuanvNet vs MatchedClassicalNet — stesso fold,
    stesso seed, stesso input slice-come-canali, trunk e testa identici, cosi'
    ogni differenza e' attribuibile al blocco centrale (quanv vs conv);
  * test di Wilcoxon appaiato sulle accuratezze finali, e cattura separata del
    vantaggio all'EPOCA 1 (l'osservazione chiave della tesi: efficienza di
    apprendimento precoce, non accuratezza asintotica superiore).

Backend quantistico: Qiskit 2.2 (statevector di default; "aer" per velocita').
NB: il forward quantistico via parameter-shift e' costoso; i default sono
conservativi (poche epoche, batch piccolo). Per campagne complete conviene
backend "aer", PUB splitting (n_parallel_chunks) e piu' epoche.
"""
from __future__ import annotations

import dataclasses
import json
import random
import time
from pathlib import Path

import os
# Cap dei thread BLAS PRIMA di importare numpy/torch: con i job in parallelo
# evita l'oversubscription (ogni processo userebbe altrimenti tutti i core).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import wilcoxon

from .config import Config
from .data import build_dataframe, make_datasets
from .metrics import compute_metrics, class_weights_from_labels
from .quantum.engine import BackendManager
from .quantum.hybrid import HybridQuanvNet, MatchedClassicalNet
from .progress import pbar


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _make_config_2d(cfg: Config) -> Config:
    """Forza il layout slice-come-canali coerente con la rete quantistica."""
    n_slices = cfg.data.slice_end - cfg.data.slice_start
    if cfg.quantum.in_channels != n_slices:
        raise ValueError(
            f"quantum.in_channels ({cfg.quantum.in_channels}) != numero di slice "
            f"({n_slices}). Allinea slice_start/slice_end o in_channels.")
    data2d = dataclasses.replace(
        cfg.data, channels_as_slices=True, img_size=cfg.quantum.img_size)
    return dataclasses.replace(cfg, data=data2d)


def _run_epoch(model, loader, device, optimizer=None, desc="", criterion=None):
    train = optimizer is not None
    model.train(train)
    crit = criterion if criterion is not None else nn.CrossEntropyLoss()
    probs, labels, losses = [], [], []
    correct = total = 0
    bar = pbar(loader, desc=desc)
    for batch in bar:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = crit(logits, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        probs.append(torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy())
        labels.append(y.cpu().numpy())
        correct += int((logits.detach().argmax(1) == y).sum().item())
        total += int(y.numel())
        if hasattr(bar, "set_postfix"):
            bar.set_postfix(loss=f"{np.mean(losses):.3f}", acc=f"{correct/total:.3f}")
    return np.concatenate(probs), np.concatenate(labels)


def _train_one(model, dl_tr, dl_va, device, max_epochs, lr, weight_decay, patience, tag="", criterion=None):
    """Allena un modello; ritorna le metriche per-epoca e quelle finali."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

    epoch_acc, epoch_auc = [], []
    best_auc, best_state, since_best = -1.0, None, 0
    for ep in range(max_epochs):
        t0 = time.time()
        _run_epoch(model, dl_tr, device, optimizer=opt, desc=f"{tag} ep{ep+1} train", criterion=criterion)
        sched.step()
        vp, vy = _run_epoch(model, dl_va, device, desc=f"{tag} ep{ep+1} val", criterion=criterion)
        m = compute_metrics(vy, vp, n_boot=0)
        epoch_acc.append(m.accuracy)
        epoch_auc.append(m.auc)
        print(f"    [{tag} | epoca {ep+1}/{max_epochs}] "
              f"val_acc={m.accuracy:.3f} val_AUC={m.auc:.3f} ({time.time()-t0:.1f}s)", flush=True)
        if m.auc > best_auc:
            best_auc = m.auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            since_best = 0
        else:
            since_best += 1
            if since_best >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    vp, vy = _run_epoch(model, dl_va, device)
    final = compute_metrics(vy, vp)
    return {
        "epoch_acc": epoch_acc,
        "epoch_auc": epoch_auc,
        "epoch1_acc": epoch_acc[0] if epoch_acc else float("nan"),
        "final_acc": final.accuracy,
        "final_auc": final.auc,
        "final_auc_ci95": final.auc_ci95,
    }


def _run_job(payload):
    """Allena la coppia (ibrido, classico) per un singolo (fold, seed).
    Eseguibile in un processo worker: argomenti e ritorno sono picklabili."""
    cfg, fold, seed, df_tr, df_va, max_epochs, batch_size, parallel = payload
    if cfg.quantum.torch_threads:
        torch.set_num_threads(cfg.quantum.torch_threads)
    # In parallelo si usa la CPU: la GPU singola sarebbe contesa tra i processi,
    # e il forward quanvoluzionale e' comunque su CPU (Qiskit).
    device = torch.device("cpu") if parallel else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    ds_tr, ds_va = make_datasets(df_tr, df_va, cfg)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0)

    weight = None
    if cfg.train.class_weights:
        w = class_weights_from_labels(df_tr["label"].values, cfg.quantum.num_classes)
        weight = torch.tensor(w, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    # in parallelo Aer usa pochi thread per processo (budget = n_job x thread/job)
    aer_par = cfg.quantum.omp_threads_per_job if parallel else cfg.quantum.aer_parallel

    set_seed(seed)
    bm = BackendManager(cfg.quantum.backend_type, seed=seed, aer_parallel=aer_par).initialize()
    hybrid = HybridQuanvNet(cfg.quantum, bm)
    h = _train_one(hybrid, dl_tr, dl_va, device, max_epochs, cfg.train.lr,
                   cfg.train.weight_decay, cfg.train.early_stop_patience,
                   tag=f"ibrido f{fold} s{seed}", criterion=criterion)
    hybrid_state = {k: v.cpu() for k, v in hybrid.state_dict().items()}

    set_seed(seed)
    classical = MatchedClassicalNet(cfg.quantum)
    c = _train_one(classical, dl_tr, dl_va, device, max_epochs, cfg.train.lr,
                   cfg.train.weight_decay, cfg.train.early_stop_patience,
                   tag=f"classico f{fold} s{seed}", criterion=criterion)

    return {"fold": fold, "seed": seed, "hybrid": h, "classical": c,
            "hybrid_state": hybrid_state, "hybrid_final_auc": h.get("final_auc")}


def train_quantum_cv(cfg: Config, n_seeds: int = 5, max_epochs: int = 15,
                     batch_size: int = 4) -> dict:
    cfg = _make_config_2d(cfg)
    if cfg.quantum.torch_threads:
        torch.set_num_threads(cfg.quantum.torch_threads)
    n_jobs = max(1, cfg.quantum.n_parallel_jobs)
    parallel = n_jobs > 1
    print(f"[quantum] backend={cfg.quantum.backend_type} | job_paralleli={n_jobs} "
          f"thread/job={cfg.quantum.omp_threads_per_job} "
          f"(seriale: aer_parallel={cfg.quantum.aer_parallel}, chunks={cfg.quantum.n_parallel_chunks})")
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataframe(cfg.data)
    skf = StratifiedGroupKFold(n_splits=cfg.train.n_folds, shuffle=True,
                               random_state=cfg.train.seed)

    # griglia di job (fold, seed) — porting dello schema multi-seed della tesi
    jobs = []
    for fold, (tr, va) in enumerate(skf.split(df, df["label"], groups=df["group"])):
        df_tr, df_va = df.iloc[tr], df.iloc[va]
        for s in range(n_seeds):
            seed = cfg.train.seed + 111 * s
            jobs.append((cfg, fold, seed, df_tr, df_va, max_epochs, batch_size, parallel))

    results = []
    if parallel:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"[quantum] avvio {len(jobs)} job su {n_jobs} processi...")
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = [ex.submit(_run_job, j) for j in jobs]
            for fut in as_completed(futs):
                results.append(fut.result())
    else:
        for j in jobs:
            results.append(_run_job(j))

    # checkpoint dell'ibrido migliore per fold (ensemble per il test cieco)
    from collections import defaultdict
    by_fold = defaultdict(list)
    for r in results:
        by_fold[r["fold"]].append(r)
    for f, rs in by_fold.items():
        valid = [r for r in rs if r["hybrid_final_auc"] is not None]
        if valid:
            best = max(valid, key=lambda r: r["hybrid_final_auc"])
            torch.save(best["hybrid_state"], out_dir / f"qfold{f}_best.pt")

    pairs = [{"fold": r["fold"], "seed": r["seed"],
              "hybrid": r["hybrid"], "classical": r["classical"]} for r in results]
    pairs.sort(key=lambda p: (p["fold"], p["seed"]))
    for p in pairs:
        h, c = p["hybrid"], p["classical"]
        print(f"[fold {p['fold']} seed {p['seed']}] "
              f"ibrido: acc1={h['epoch1_acc']:.3f} accF={h['final_acc']:.3f} AUC={h['final_auc']:.3f} | "
              f"classico: acc1={c['epoch1_acc']:.3f} accF={c['final_acc']:.3f} AUC={c['final_auc']:.3f}")

    summary = _summarize(pairs)
    (out_dir / "quantum_results.json").write_text(
        json.dumps({"pairs": pairs, "summary": summary}, default=str, indent=2))
    _print_summary(summary)
    return {"pairs": pairs, "summary": summary}


def _summarize(pairs: list[dict]) -> dict:
    hf = np.array([p["hybrid"]["final_acc"] for p in pairs])
    cf = np.array([p["classical"]["final_acc"] for p in pairs])
    h1 = np.array([p["hybrid"]["epoch1_acc"] for p in pairs])
    c1 = np.array([p["classical"]["epoch1_acc"] for p in pairs])

    def _wilcoxon(a, b):
        d = a - b
        if np.allclose(d, 0) or len(d) < 1:
            return {"statistic": None, "p_value": None}
        try:
            st, p = wilcoxon(a, b)
            return {"statistic": float(st), "p_value": float(p)}
        except ValueError:
            return {"statistic": None, "p_value": None}

    return {
        "n_pairs": len(pairs),
        "final_acc": {
            "hybrid_mean": float(hf.mean()), "hybrid_std": float(hf.std()),
            "classical_mean": float(cf.mean()), "classical_std": float(cf.std()),
            "delta_mean_pp": float((hf - cf).mean() * 100),
            "wilcoxon": _wilcoxon(hf, cf),
        },
        "epoch1_acc": {
            "hybrid_mean": float(np.nanmean(h1)), "classical_mean": float(np.nanmean(c1)),
            "delta_mean_pp": float(np.nanmean(h1 - c1) * 100),
            "wilcoxon": _wilcoxon(h1, c1),
        },
    }


def _print_summary(s: dict) -> None:
    fa, e1 = s["final_acc"], s["epoch1_acc"]
    print(f"\n{'='*64}\n  Riepilogo ({s['n_pairs']} coppie fold×seed)\n{'='*64}")
    print(f"Accuratezza FINALE:")
    print(f"  ibrido   {fa['hybrid_mean']:.4f} ± {fa['hybrid_std']:.4f}")
    print(f"  classico {fa['classical_mean']:.4f} ± {fa['classical_std']:.4f}")
    print(f"  Δ = {fa['delta_mean_pp']:+.2f} pp | Wilcoxon p = {fa['wilcoxon']['p_value']}")
    print(f"Accuratezza EPOCA 1 (efficienza precoce):")
    print(f"  Δ = {e1['delta_mean_pp']:+.2f} pp | Wilcoxon p = {e1['wilcoxon']['p_value']}")
    print("\nNota: l'ipotesi da testare e' se il vantaggio precoce dell'ibrido,\n"
          "irrilevante all'asintoto su EuroSAT, diventi utile nel regime small-n.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Training quantistico ibrido vs classico (CV multi-seed).")
    ap.add_argument("--jobs", type=int, default=None,
                    help="processi paralleli sui job (fold,seed); override di n_parallel_jobs. Su 14 core: 6")
    ap.add_argument("--serial", action="store_true", help="forza esecuzione seriale (= --jobs 1)")
    ap.add_argument("--seeds", type=int, default=5, help="numero di seed per fold")
    ap.add_argument("--epochs", type=int, default=15, help="epoche massime per modello")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    cfg = Config()
    n_jobs = 1 if args.serial else args.jobs
    if n_jobs is not None:
        cfg = dataclasses.replace(cfg, quantum=dataclasses.replace(cfg.quantum, n_parallel_jobs=n_jobs))
    train_quantum_cv(cfg, n_seeds=args.seeds, max_epochs=args.epochs, batch_size=args.batch_size)
