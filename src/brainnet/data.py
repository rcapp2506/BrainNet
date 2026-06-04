"""Caricamento dati e pipeline di augmentation medicalmente fedele.

Principio guida (vedi discussione clinica): un'augmentation e' lecita solo se
riproduce una variabilita' reale dell'acquisizione, indipendente dalla diagnosi.
Per la PET striatale C-DOPA cio' significa:

  LECITO     traslazione rigida piccola (posizionamento nel gantry)
             rotazione in-plane piccola (inclinazione testa / AC-PC imperfetto)
             rumore di conteggio Poisson (statistica di acquisizione)
             flip sinistra-destra SOLO per il task binario PD/controllo
  VIETATO    scaling/zoom (i gangli della base hanno dimensione conservata;
             lo zoom falsa l'estensione della captazione, che e' diagnostica)
             deformazioni elastiche (falsano la morfologia del nucleo)
             gamma/contrasto forti (falsano il rapporto di binding = il segnale)

Lo stesso loader serve due layout, scelti via `channels_as_slices`:
  * 3D  (1, H, W, D)  -> CNN 3D classica
  * 2D  (D, H, W)     -> pipeline 2D slice-come-canali (ibrida quanv + controllo
                          classico appaiato), con resize in-plane a img_size.

Lo split e' SEMPRE a livello di paziente (un GUID = un volume = un gruppo).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pydicom
import torch

from monai.data import Dataset
from monai.transforms import Compose, RandAffine, RandFlip, ScaleIntensity, Resize

from .config import Config, DataConfig


# ─────────────────────────────────────────────────────────────────────────────
#  Lettura DICOM
# ─────────────────────────────────────────────────────────────────────────────

def load_volume(guid_dir, cfg: DataConfig) -> np.ndarray:
    """Legge le slice DICOM di un paziente, le ordina per posizione assiale,
    ritaglia la ROI peri-striatale e impila il sotto-volume. Ritorna (H, W, D)."""
    slices = [pydicom.dcmread(str(p)) for p in sorted(guid_dir.glob("*.dcm"))]

    def _z(s):
        try:
            return float(s.ImagePositionPatient[2])
        except Exception:
            return float(getattr(s, "InstanceNumber", 0))

    slices.sort(key=_z)
    volume = np.stack([s.pixel_array.astype(np.float32) for s in slices])  # (D_full,H,W)
    x0, x1 = cfg.crop_x
    y0, y1 = cfg.crop_y
    sub = volume[cfg.slice_start:cfg.slice_end, x0:x1, y0:y1]               # (D,H,W)
    return np.transpose(sub, (1, 2, 0))                                     # (H,W,D)


# ─────────────────────────────────────────────────────────────────────────────
#  Transform custom: rumore di conteggio Poisson (fedele alla fisica PET)
# ─────────────────────────────────────────────────────────────────────────────

class RandPoissonCounts:
    """Simula un'acquisizione a conteggi piu' bassi (dose/tempo minori).

    L'immagine normalizzata in [0,1] e' trattata come tasso atteso; la si scala
    a un numero di conteggi equivalenti, si campiona Poisson e si rinormalizza.
    Preserva anatomia e pattern di captazione, varia solo la realizzazione del
    rumore: l'unica augmentation che riproduce la vera sorgente di variabilita'
    della PET.
    """

    def __init__(self, prob, counts_range, seed=None):
        self.prob = prob
        self.counts_range = counts_range
        self.rng = np.random.default_rng(seed)

    def __call__(self, img):
        arr = img.numpy() if isinstance(img, torch.Tensor) else np.asarray(img)
        if self.rng.random() > self.prob:
            return img
        lam = self.rng.uniform(*self.counts_range)
        scaled = np.clip(arr.astype(np.float64), 0, None) * lam
        noisy = (self.rng.poisson(scaled).astype(np.float32) / lam)
        return torch.as_tensor(noisy) if isinstance(img, torch.Tensor) else noisy


# ─────────────────────────────────────────────────────────────────────────────
#  Costruzione delle pipeline di transform
# ─────────────────────────────────────────────────────────────────────────────

class _LoadLayout:
    """Carica il volume e lo dispone nel layout richiesto.

    Implementata come classe (non closure) per essere picklabile: su Windows i
    worker del DataLoader avviano processi separati e devono serializzare la
    trasformazione, e le funzioni annidate non sono picklabili.
    """

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg

    def __call__(self, item):
        v = load_volume(item["guid_dir"], self.cfg)
        if self.cfg.channels_as_slices:
            return np.transpose(v, (2, 0, 1)).astype(np.float32)   # (D,H,W)
        return v[np.newaxis, ...].astype(np.float32)               # (1,H,W,D)


def _to_layout(cfg: DataConfig):
    """Lambda di ingresso: legge il volume e lo dispone nel layout richiesto."""
    return _LoadLayout(cfg)


def _spatial_resize(cfg: DataConfig):
    if cfg.channels_as_slices:
        return Resize(spatial_size=(cfg.img_size, cfg.img_size))       # (D,img,img)
    return Resize(spatial_size=cfg.spatial_size)                        # (1,H,W,D)


def _affine(cfg: DataConfig):
    """Solo rigido in-plane: traslazione + rotazione piccole, NIENTE scaling."""
    rot = float(np.deg2rad(cfg.aug_rotate_deg))
    t = cfg.aug_translate_vox
    if cfg.channels_as_slices:                                          # 2D: spatial (H,W)
        return RandAffine(prob=0.5, translate_range=(t, t),
                          rotate_range=(rot,), padding_mode="zeros")
    return RandAffine(prob=0.5, translate_range=(t, t, 0.0),            # 3D: solo in-plane
                      rotate_range=(0.0, 0.0, rot), padding_mode="zeros")


def build_transforms(cfg: DataConfig, train: bool) -> Compose:
    steps = [_to_layout(cfg), _spatial_resize(cfg), ScaleIntensity()]  # per-volume [0,1]
    if train:
        steps.append(_affine(cfg))
        if cfg.aug_flip_lr:
            steps.append(RandFlip(prob=0.5, spatial_axis=cfg.lr_axis))
        if cfg.aug_poisson:
            steps.append(RandPoissonCounts(0.3, cfg.aug_poisson_counts))
    return Compose(steps)


# ─────────────────────────────────────────────────────────────────────────────
#  DataFrame e Dataset
# ─────────────────────────────────────────────────────────────────────────────

def build_dataframe(cfg: DataConfig) -> pd.DataFrame:
    """(guid_dir, label, group) dal CSV pseudonimizzato; verifica le cartelle.

    Segnala in modo ESPLICITO quanti pazienti del CSV hanno effettivamente la
    cartella DICOM: i mancanti verrebbero altrimenti scartati in silenzio.
    """
    df = pd.read_csv(cfg.data_root / cfg.labels_csv, dtype={cfg.guid_col: str})
    root = cfg.data_root / cfg.dicom_subdir
    rows, missing = [], []
    for _, r in df.iterrows():
        guid = str(r[cfg.guid_col])
        d = root / guid
        if not d.is_dir():
            missing.append(guid)
            continue
        label = 1 if str(r[cfg.label_col]).strip().upper() == cfg.positive_label else 0
        rows.append({"guid_dir": d, "label": label, "group": guid})
    out = pd.DataFrame(rows)

    total, found = len(df), len(out)
    print(f"[dati] {root}: {found}/{total} pazienti con cartella DICOM"
          + ("" if missing else " (completo)"))
    if found:
        n_pos = int((out["label"] == 1).sum())
        print(f"[dati] bilanciamento classi: positivi(P)={n_pos}, negativi(N/T)={found - n_pos}")
    if missing:
        ex = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        print(f"[ATTENZIONE] {len(missing)} GUID sono nel CSV ma NON hanno cartella in {root}.")
        print(f"[ATTENZIONE] Il run userebbe solo {found} pazienti su {total}. "
              f"Controlla la de-identificazione o il percorso 'dicom_subdir'.")
        print(f"[ATTENZIONE] Esempi di mancanti: {ex}")
    if out.empty:
        raise RuntimeError(f"Nessun paziente trovato sotto {root}. Verifica CSV e dati.")
    return out


class _VolumeDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        x = self.transform(it)
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        return {"image": x.float(), "label": int(it["label"])}


def make_datasets(df_train: pd.DataFrame, df_val: pd.DataFrame, cfg: Config):
    tr = df_train[["guid_dir", "label"]].to_dict("records")
    va = df_val[["guid_dir", "label"]].to_dict("records")
    return (_VolumeDataset(tr, build_transforms(cfg.data, train=True)),
            _VolumeDataset(va, build_transforms(cfg.data, train=False)))


def make_eval_dataset(df: pd.DataFrame, cfg: Config):
    """Dataset di sola valutazione: trasformazioni deterministiche (train=False),
    nessuna augmentation. Usato per il test cieco e, in generale, per inferenza."""
    items = df[["guid_dir", "label"]].to_dict("records")
    return _VolumeDataset(items, build_transforms(cfg.data, train=False))
