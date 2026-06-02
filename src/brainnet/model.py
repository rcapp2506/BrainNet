"""Definizione dei modelli 3D.

Due opzioni:
  * "densenet": MONAI DenseNet121 a 3 dimensioni. E' la rete che oggi userei come
    default per volumi PET/CT: profonda ma regolarizzata, robusta su dataset piccoli.
  * "smallcnn": l'equivalente svecchiato della rete del notebook (4 blocchi conv),
    riscritta in PyTorch con BatchNorm e GELU, utile come baseline leggera.
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
                nn.MaxPool3d(2),
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
