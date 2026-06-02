"""Caricamento dati e pipeline leak-free.

Differenze chiave rispetto al notebook originale:
  * un paziente = un GUID = un volume; lo split avviene PER PAZIENTE.
  * l'augmentation e' una transform MONAI applicata on-the-fly SOLO al training;
    non genera mai copie persistenti che possano finire in validation.
  * la normalizzazione di intensita' e' per-volume (nessuna statistica globale
    calcolata anche sulla validation).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

from monai.data import Dataset
from monai.transforms import (
    Compose, EnsureChannelFirst, ScaleIntensity, RandAffine,
    RandFlip, RandGaussianNoise, ToTensor, Lambda,
)

from .config import Config


def load_volume(guid_dir: Path, cfg) -> np.ndarray:
    """Legge le slice DICOM di un paziente, le ordina per posizione assiale,
    ritaglia la ROI e impila il sotto-volume di interesse.

    Ritorna un array float32 di forma (H, W, D).
    """
    slices = [pydicom.dcmread(str(p)) for p in guid_dir.glob("*.dcm")]
    # Ordinamento robusto: ImagePositionPatient[2], fallback su InstanceNumber.
    def _z(s):
        try:
            return float(s.ImagePositionPatient[2])
        except Exception:
            return float(getattr(s, "InstanceNumber", 0))

    slices.sort(key=_z)
    volume = np.stack([s.pixel_array.astype(np.float32) for s in slices])  # (D_full, H, W)

    x0, x1 = cfg.crop_x
    y0, y1 = cfg.crop_y
    sub = volume[cfg.slice_start:cfg.slice_end, x0:x1, y0:y1]               # (D, H, W)
    return np.transpose(sub, (1, 2, 0))                                     # (H, W, D)


def build_dataframe(cfg: DataConfig) -> pd.DataFrame:  # type: ignore[name-defined]
    """Costruisce il DataFrame (filepath, label, group) a partire dal CSV
    PSEUDONIMIZZATO. Verifica che ogni GUID abbia una cartella DICOM."""
    csv_path = cfg.data_root / cfg.labels_csv
    df = pd.read_csv(csv_path, dtype={cfg.guid_col: str})
    root = cfg.data_root / cfg.dicom_subdir

    rows = []
    for _, r in df.iterrows():
        guid = str(r[cfg.guid_col])
        d = root / guid
        if not d.is_dir():
            continue
        label = 1 if str(r[cfg.label_col]).strip().upper() == cfg.positive_label else 0
        rows.append({"guid_dir": d, "label": label, "group": guid})

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"Nessun paziente trovato sotto {root}. Verifica il CSV e i dati.")
    return out


def _train_transforms(cfg: Config):
    """Augmentation applicata SOLO al training. Affine 3D leggera (rotazione +
    traslazione), flip, rumore gaussiano: sostituisce l'augmentation a sessioni
    TF1 dell'originale, in modo vettorizzato e leak-free."""
    return Compose([
        Lambda(lambda x: load_volume(x["guid_dir"], cfg.data) if isinstance(x, dict) else x),
        EnsureChannelFirst(channel_dim="no_channel"),
        ScaleIntensity(),  # per-volume in [0, 1]
        RandAffine(prob=0.5, translate_range=(8, 8, 0),
                   rotate_range=(0.0, 0.0, 0.1), padding_mode="zeros"),
        RandFlip(prob=0.5, spatial_axis=0),
        RandGaussianNoise(prob=0.2, std=0.02),
        ToTensor(),
    ])


def _eval_transforms(cfg: Config):
    return Compose([
        Lambda(lambda x: load_volume(x["guid_dir"], cfg.data) if isinstance(x, dict) else x),
        EnsureChannelFirst(channel_dim="no_channel"),
        ScaleIntensity(),
        ToTensor(),
    ])


def make_datasets(df_train: pd.DataFrame, df_val: pd.DataFrame, cfg: Config):
    train_items = df_train[["guid_dir", "label"]].to_dict("records")
    val_items = df_val[["guid_dir", "label"]].to_dict("records")

    class _DS(Dataset):
        def __init__(self, items, tf):
            self.items, self.tf = items, tf

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            it = self.items[i]
            return {"image": self.tf(it), "label": int(it["label"])}

    return _DS(train_items, _train_transforms(cfg)), _DS(val_items, _eval_transforms(cfg))
