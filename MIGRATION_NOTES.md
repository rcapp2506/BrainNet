# Note di migrazione — dal notebook alla pipeline reingegnerizzata

Mappa dei problemi individuati nel notebook `CDOPA-GOLD-STANDARD-01` e di come sono stati risolti.

## Bug e criticità corrette

| # | Problema nell'originale | Gravità | Soluzione |
|---|--------------------------|---------|-----------|
| 1 | **Data leakage da augmentation**: `train_test_split` rieseguito sul dataset già aumentato (celle 19→33/37) → copie traslate dello stesso paziente in train e validation | Critico | Augmentation come transform on-the-fly nel solo DataLoader di training (`data.py`); nessuna copia persistente |
| 2 | **Split a livello di campione, non di paziente** | Critico | `StratifiedGroupKFold` con `groups=GUID` (`train.py`) |
| 3 | **Loop sui seed che continua ad addestrare lo stesso modello** (cella 37) contaminando la validation | Critico | Un modello reinizializzato per fold; nessun riuso di stato tra split |
| 4 | **Matrice di confusione incoerente** tra celle (TP/TN invertiti) e non conforme a sklearn | Grave | `confusion_matrix(..., labels=[0,1]).ravel()` → `tn,fp,fn,tp` espliciti (`metrics.py`) |
| 5 | **Normalizzazione con statistiche globali** (media/max su tutto il set, validation inclusa) | Grave | `ScaleIntensity` per-volume |
| 6 | **Dataset minuscolo + singolo split** | Grave | Cross-validation + AUC con IC95 bootstrap |
| 7 | `np.int` deprecato (funzione `suv`) | Tecnico | Rimosso; tipi espliciti |
| 8 | Augmentation a `tf.compat.v1.Session` + `extract_glimpse` in loop Python | Tecnico | `RandAffine`/`RandFlip` MONAI, vettorizzate |
| 9 | `Convolution3D`, `keras.wrappers.scikit_learn`, `Adadelta(lr=...)` obsoleti/rimossi | Tecnico | PyTorch + MONAI; `AdamW(learning_rate→lr)` |
| 10 | `threshold=7000` hardcoded che sovrascrive il parametro; decine di valori commentati | Qualità | Parametri unici in `config.py` |
| 11 | Mascheratura che muta gli array in place (`slicecut[True]=0.`) | Qualità | Pipeline funzionale, nessuna mutazione degli input |
| 12 | Seeding disattivato | Riproducibilità | `set_seed()` deterministico |
| 13 | Import morti (`pandas_datareader`, `azure...BlobService`) | Pulizia | Rimossi |
| 14 | **Nomi pazienti reali nel CSV** | Privacy | `scripts/anonymize_csv.py` + `.gitignore` |

## Decisione di framework

L'originale è TF/Keras. La pipeline nuova è **PyTorch + MONAI**, che oggi è lo standard di fatto per imaging medico 3D (loader DICOM, transform spaziali leak-free, reti 3D pronte). È un cambio sostanziale e reversibile: se si preferisce restare su Keras, gli stessi principi (split per paziente, augmentation solo in training, metriche corrette, CV) si applicano identici.

## Possibile estensione (da valutare insieme)

Una variante quantistica/ibrida (encoder classico → classificatore variazionale quantistico) è esplorabile a fini di ricerca, ma su ~46 pazienti non offrirebbe vantaggi pratici rispetto alla CNN 3D classica: la annoto come direzione opzionale, non come raccomandazione.
