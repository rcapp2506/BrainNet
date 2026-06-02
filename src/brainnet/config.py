"""Configurazione centralizzata e tipizzata.

Tutti i "numeri magici" che nel notebook originale erano sparsi tra le celle
(e spesso sovrascritti a runtime) vivono qui, in un unico oggetto serializzabile.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass(frozen=True)
class DataConfig:
    # Layout su disco: <root>/CDOPA/<GUID>/IM-0001-XXXX.dcm
    data_root: Path = Path("data")
    dicom_subdir: str = "CDOPA"
    labels_csv: str = "labels.csv"           # CSV PSEUDONIMIZZATO (solo GUID, LABEL)
    guid_col: str = "GUID"
    label_col: str = "LABEL"
    positive_label: str = "P"                # P -> 1 ; tutto il resto -> 0

    # Region of interest: nel notebook erano xmin/xmax/ymin/ymax + irange*
    # Centro striatale: 10 slice, crop 130x130. Reso esplicito e documentato.
    slice_start: int = 20
    slice_end: int = 30                      # exclusive -> 10 slice
    crop_x: tuple[int, int] = (33, 163)      # 130 px
    crop_y: tuple[int, int] = (73, 203)      # 130 px

    spatial_size: tuple[int, int, int] = (130, 130, 10)  # H, W, D


@dataclass(frozen=True)
class TrainConfig:
    n_folds: int = 5
    seed: int = 42
    batch_size: int = 4
    max_epochs: int = 150
    early_stop_patience: int = 20            # su AUC di validazione
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    amp: bool = True                         # mixed precision se CUDA disponibile


@dataclass(frozen=True)
class ModelConfig:
    # "densenet" (MONAI DenseNet121 3D, raccomandato) oppure "smallcnn"
    arch: str = "densenet"
    dropout: float = 0.2
    n_classes: int = 2


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output_dir: Path = Path("runs")

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), default=str, indent=2))
