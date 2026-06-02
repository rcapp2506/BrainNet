# BrainNet

CNN 3D per la diagnosi precoce della malattia di Parkinson da imaging PET con tracciante C-DOPA (captazione striatale dopaminergica).

Questa è la reingegnerizzazione del notebook di ricerca originale (`CDOPA-GOLD-STANDARD-01`) secondo prassi attuali di deep learning per imaging medico: split a livello di paziente, pipeline leak-free, cross-validation, metriche corrette, separazione netta tra codice e dati sensibili.

## ⚠️ Dati sanitari — leggere prima di tutto

I dati grezzi (immagini DICOM e CSV pazienti) **non sono e non devono essere** in questo repository. Il CSV originale conteneva nomi reali di pazienti associati alla diagnosi: è un dato di categoria particolare (art. 9 GDPR). Prima di qualsiasi uso o condivisione:

1. Tieni i dati grezzi fuori dal repo (vedi `.gitignore`).
2. Genera il CSV pseudonimizzato con soli `GUID,LABEL`:
   ```bash
   python scripts/anonymize_csv.py data/carbidopa.csv data/labels.csv
   ```
3. I DICOM andrebbero ulteriormente de-identificati (rimozione dei tag PHM: `PatientName`, `PatientID`, date di nascita, ecc.) e gestiti secondo l'approvazione etica dello studio.

## Layout dei dati atteso (in locale, non versionato)

```
data/
├── labels.csv            # solo GUID,LABEL  (pseudonimizzato)
└── CDOPA/
    └── <GUID>/
        └── IM-0001-XXXX.dcm
```

## Installazione

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Addestramento

```bash
python -m brainnet.train
```

Esegue una 5-fold `StratifiedGroupKFold` (raggruppamento per paziente), salva i pesi migliori per fold in `runs/` e stampa AUC media ± deviazione con intervalli di confidenza bootstrap.

La configurazione è centralizzata e tipizzata in `src/brainnet/config.py`.

## Struttura

| File | Ruolo |
|------|-------|
| `config.py`  | Tutti i parametri (ROI, training, modello) in un'unica dataclass serializzabile |
| `data.py`    | Lettura DICOM, raggruppamento per paziente, transform MONAI leak-free |
| `model.py`   | DenseNet121-3D (default) o `SmallCNN3D` (baseline svecchiata dell'originale) |
| `train.py`   | Cross-validation, AdamW + cosine annealing, early stopping su AUC |
| `metrics.py` | Matrice di confusione corretta, sensibilità/specificità/AUC + IC95 |
| `scripts/anonymize_csv.py` | Pseudonimizzazione del CSV pazienti |

Vedi `MIGRATION_NOTES.md` per la mappa dettagliata dei problemi corretti rispetto al notebook originale.

## Nota metodologica

Con ~46 pazienti l'obiettivo realistico non è "accuracy 1.0" (nel codice originale era un artefatto di data leakage) ma una stima onesta e con incertezza dell'AUC in cross-validation. Per risultati robusti servono più soggetti e, idealmente, una validazione esterna su una coorte indipendente.
