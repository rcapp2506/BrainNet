"""Modelli per il confronto appaiato ibrido-vs-classico (late fusion).

Trunk e testa IDENTICI tra i due modelli; differiscono solo nel blocco centrale:
  * HybridQuanvNet   — blocco quanvoluzionale (QuantumConvLayer).
  * MatchedClassicalNet — Conv2d(ch->ch, k=3) strutturalmente isomorfo, per il
    controllo classico a capacita' appaiata (come il "Level 2" della tesi).

Mantenere trunk/testa identici garantisce che ogni differenza misurata sia
attribuibile al blocco centrale, non a un classificatore piu' forte.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import QuantumConfig
from .engine import BackendManager
from .layer import QuantumConvLayer


def _trunk(cfg: QuantumConfig) -> nn.Sequential:
    ch, ks, pad = cfg.conv_channels, cfg.conv_kernel_size, cfg.conv_padding
    return nn.Sequential(
        nn.Conv2d(cfg.in_channels, ch, ks, padding=pad), nn.BatchNorm2d(ch),
        nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(ch, ch, ks, padding=pad), nn.BatchNorm2d(ch),
        nn.ReLU(), nn.MaxPool2d(2),
    )


def _quanv_out_size(cfg: QuantumConfig) -> int:
    fm = cfg.img_size // 4                      # dopo due MaxPool2d
    return (fm - cfg.kernel_size) // cfg.stride + 1


def _head(cfg: QuantumConfig) -> nn.Sequential:
    flat = cfg.conv_channels * _quanv_out_size(cfg) ** 2
    hidden = max(flat // 3, 64)
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(flat, hidden), nn.ReLU(),
        nn.Linear(hidden, cfg.num_classes),
    )


class HybridQuanvNet(nn.Module):
    """Conv -> Conv -> Quanv -> testa lineare (late fusion)."""

    def __init__(self, cfg: QuantumConfig, backend_manager: BackendManager):
        super().__init__()
        self.trunk = _trunk(cfg)
        self.quanv = QuantumConvLayer(
            cfg.num_qubits, cfg.kernel_size, cfg.stride, backend_manager,
            ansatz=cfg.ansatz, measure_qubit=cfg.measure_qubit,
            n_parallel_chunks=cfg.n_parallel_chunks)
        self.head = _head(cfg)

    def forward(self, x):
        return self.head(self.quanv(self.trunk(x)))


class MatchedClassicalNet(nn.Module):
    """Controllo classico isomorfo: Conv2d(ch->ch, k=3) al posto del quanv."""

    def __init__(self, cfg: QuantumConfig):
        super().__init__()
        ch = cfg.conv_channels
        self.trunk = _trunk(cfg)
        self.mid = nn.Sequential(
            nn.Conv2d(ch, ch, cfg.kernel_size, padding=0), nn.ReLU())
        self.head = _head(cfg)

    def forward(self, x):
        return self.head(self.mid(self.trunk(x)))


def build_quantum_models(cfg: QuantumConfig, backend_manager: BackendManager):
    """Ritorna (ibrido, controllo_classico_appaiato) con trunk/testa identici."""
    return HybridQuanvNet(cfg, backend_manager), MatchedClassicalNet(cfg)
