"""Definizione dei modelli 3D.

Due opzioni:
  * "smallcnn" (DEFAULT): erede modernizzato della CNN del paper 2021 (blocchi
    Conv3D + BatchNorm + GELU + pooling), con AdaptiveAvgPool finale. Adatta a
    volumi anisotropi a poche slice (qui 130x130x10) e a pochi pazienti.
  * "densenet": MONAI DenseNet121 3D. ATTENZIONE: esegue ~5 dimezzamenti, quindi
    l'asse delle slice (10) collassa sotto il kernel di pooling -> errore. Usabile
    solo ridimensionando il volume a una profondita' molto maggiore (interpolando
    slice, sconsigliato) e comunque sovradimensionata per ~98 pazienti.
"""
from __future__ import annotations

import torch
from torch import nn
from monai.networks.nets import DenseNet121

from .config import ModelConfig


class SmallCNN3D(nn.Module):
    """Baseline: erede modernizzato della CNN originale (Conv3D x4)."""

    def __init__(self, n_classes: int = 2, dropout: float = 0.2):
        super().__init__()

        def block(ci, co):
            return nn.Sequential(
                nn.Conv3d(ci, co, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(co),
                nn.GELU(),
                nn.MaxPool3d(2, ceil_mode=True),   # ceil: l'asse slice (10) non collassa a 0
                nn.Dropout3d(dropout),
            )

        self.features = nn.Sequential(
            block(1, 16), block(16, 32), block(32, 64),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(64, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.arch == "densenet":
        return DenseNet121(
            spatial_dims=3, in_channels=1, out_channels=cfg.n_classes,
            dropout_prob=cfg.dropout,
        )
    if cfg.arch == "smallcnn":
        return SmallCNN3D(n_classes=cfg.n_classes, dropout=cfg.dropout)
    raise ValueError(f"arch sconosciuta: {cfg.arch!r}")
