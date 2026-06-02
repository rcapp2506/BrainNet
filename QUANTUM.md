# Rete ibrida quanvoluzionale — adattamento a BrainNet

Adattamento dell'architettura quanvoluzionale (late fusion) al dataset PET
C-DOPA. Riusa lo stesso circuito (angle encoding + ansatz hardware-efficient +
misura Z, gradienti parameter-shift) cambiando solo cio' che il dominio impone.

## Modulo `src/brainnet/quantum/`
| File | Ruolo |
|------|-------|
| `circuit.py` | Circuito quanvoluzionale parametrico (Qiskit 2.2), ansatz selezionabile |
| `engine.py`  | BackendManager (statevector/Aer) + motore parameter-shift con PUB splitting |
| `layer.py`   | `QuantumConvLayer`: patch k×k → ⟨Z⟩, autograd PyTorch |
| `hybrid.py`  | `HybridQuanvNet` (Conv→Conv→Quanv→testa) + `MatchedClassicalNet` appaiato |

## Adattamenti rispetto alla versione EuroSAT
- **Input**: 10 slice peri-striatali come canali 2D (`in_channels=10`), ROI
  ridimensionata a 64×64. Riusa la pipeline 2D senza toccare il circuito.
- **Capacità**: si parte dalla variante leggera (C6, quanv 6 canali); il
  vincitore EuroSAT C16–Q64 (~4.6·10⁵ parametri) overfitta su ~46 pazienti.
- **Augmentation**: policy medicalmente fedele (vedi `data.py`), non i flip
  verticali / color jitter di EuroSAT.
- **Split**: a livello di paziente (`StratifiedGroupKFold`), non per immagine.

## Correzione di trainabilità (codice vs teoria della tesi)
Lo strato variazionale del repo originale usa solo `RZ(w)`. Con osservabile `Z`,
ogni `RZ` commuta con l'osservabile propagato a ritroso: **gradiente dei pesi
nullo → feature map fissa, pesi non addestrabili**. Verificato numericamente.

L'Eq. hardware-efficient della tesi prevede invece rotazioni `Rz·Rx` per filo.
Allineando il codice alla teoria (`ansatz="rzrx"`, default) i pesi tornano
addestrabili; parameter-shift verificato contro differenze finite (err ~1e-10).
Opzioni: `"rzrx"` (tesi, 2n pesi), `"ry"` (lean, n pesi tutti attivi),
`"rz"` (riproduce il repo ma con pesi congelati, solo per riproducibilità).
