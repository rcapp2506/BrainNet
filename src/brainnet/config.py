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
    dicom_subdir: str = "CDOPA_DEID"        # cartella DICOM de-identificata (training)
    labels_csv: str = "labels.csv"           # CSV PSEUDONIMIZZATO (solo GUID, LABEL)
    guid_col: str = "GUID"
    label_col: str = "LABEL"
    positive_label: str = "P"                # P -> 1 ; tutto il resto -> 0

    # ── Test cieco (golden set indipendente: carbidopa-test.csv) ──
    # Coorte separata, valutata UNA SOLA VOLTA a fine lavoro.
    test_labels_csv: str = "labels_test.csv"
    test_dicom_subdir: str = "CDOPA-TEST_DEID"  # cartella DICOM de-identificata (test cieco)

    # Region of interest peri-striatale (esplicita e documentata; nel notebook
    # originale erano xmin/xmax/ymin/ymax + irange* sovrascritti a runtime).
    slice_start: int = 20
    slice_end: int = 30                      # exclusive -> 10 slice
    crop_x: tuple[int, int] = (33, 163)      # 130 px
    crop_y: tuple[int, int] = (73, 203)      # 130 px

    # ── Layout di output ──
    # False -> volume 3D (1, H, W, D) per la CNN 3D classica.
    # True  -> slice-come-canali (D, H, W) per la pipeline 2D (ibrida e
    #          controllo classico appaiato), con resize in-plane a img_size.
    cache_volumes: bool = True               # tieni i volumi in RAM (letti una volta sola)
    channels_as_slices: bool = False
    img_size: int = 64                       # lato in-plane in modalita' 2D
    spatial_size: tuple[int, int, int] = (130, 130, 10)  # H, W, D (modalita' 3D)

    # ── Augmentation FEDELE (vedi note mediche) ──
    # Asse L-R anatomico nel tensore channel-first: per (D,H,W) e (1,H,W,D)
    # la colonna (W) ha indice spaziale 1. DA VERIFICARE su ImageOrientationPatient.
    lr_axis: int = 1
    aug_translate_vox: float = 6.0           # traslazione rigida in-plane (voxel)
    aug_rotate_deg: float = 6.0              # rotazione in-plane (gradi)
    aug_flip_lr: bool = True                 # solo per task binario PD/controllo
    aug_poisson: bool = True                 # rumore di conteggio realistico
    aug_poisson_counts: tuple[float, float] = (1e3, 1e4)  # range conteggi equiv.
    # NESSUNO scaling/zoom, NESSUNA deformazione elastica, NESSUN gamma/contrasto.


@dataclass(frozen=True)
class TrainConfig:
    n_folds: int = 5
    seed: int = 42
    batch_size: int = 16     # con GPU e volumi piccoli c'e' molto margine; rialzabile
    max_epochs: int = 150
    early_stop_patience: int = 20            # su AUC di validazione
    class_weights: bool = False              # ricalibrazione pesi opzionale (default OFF: non aiutava l'AUC e destabilizzava i fold)
    target_sensitivity: float = 0.95         # punto operativo clinico: max sensibilita' (evitare falsi negativi)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0      # 0 = nessun processo worker (sicuro su Windows); rialzabile ora che i transform sono picklabili
    amp: bool = True                         # mixed precision se CUDA disponibile


@dataclass(frozen=True)
class ModelConfig:
    # "densenet" (MONAI DenseNet121 3D, raccomandato) oppure "smallcnn"
    arch: str = "smallcnn"   # smallcnn: adatta a 130x130x10 e a pochi pazienti (vedi model.py)
    dropout: float = 0.2
    n_classes: int = 2


@dataclass(frozen=True)
class QuantumConfig:
    """Rete ibrida quanvoluzionale (late fusion), adattata dalla tesi.

    Capacita' VOLUTAMENTE bassa: il vincitore EuroSAT (C16-Q64) e' tarato su
    ~200 immagini; su ~46 pazienti partiamo dalla variante piu' leggera per
    non ricadere nell'overfitting gia' diagnosticato nel notebook 2019.
    """
    # ── Circuito quanvoluzionale ──
    num_qubits: int = 9                      # = kernel_size**2 (patch 3x3)
    kernel_size: int = 3
    stride: int = 1
    measure_qubit: int = 0
    # backend Qiskit 2.2 primitives V2: "statevector" | "aer"
    ansatz: str = "rzrx"                     # "rzrx" (tesi) | "ry" (lean) | "rz" (frozen)
    # ── Parallelismo CPU del simulatore (calibrato su 14 core fisici) ──
    # Il forward quanvoluzionale gira su CPU (Qiskit). Con backend "aer" gli
    # esperimenti (i circuiti delle patch) vengono eseguiti in parallelo.
    backend_type: str = "aer"                # "aer" (parallelo, veloce) | "statevector" (esatto, per check)
    aer_parallel: int = 12                   # thread/esperimenti Aer in parallelo (14 core - 2 di margine)
    n_parallel_chunks: int = 8               # split dei PUB in K blocchi
    torch_threads: int = 2                   # thread per la parte torch (evita oversubscription con Aer)
    # ── Parallelismo sui job (fold, seed): porting dello schema della tesi ──
    n_parallel_jobs: int = 1                 # 1 = seriale; su 14 core usare 6 (6 job x ~2 thread = 12)
    omp_threads_per_job: int = 2             # thread BLAS/Aer per processo worker

    # ── Trunk classico (late fusion: Conv -> Conv -> Quanv -> testa) ──
    in_channels: int = 10                    # 10 slice peri-striatali = canali
    conv_channels: int = 6                   # variante leggera (C6-Q6)
    conv_kernel_size: int = 5
    conv_padding: int = 2
    dropout_rate: float = 0.0
    img_size: int = 64
    num_classes: int = 2


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    output_dir: Path = Path("runs")

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), default=str, indent=2))
