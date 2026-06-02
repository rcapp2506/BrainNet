"""Pseudonimizzazione del CSV pazienti.

Il file originale `carbidopa.csv` contiene NOME e COGNOME reali dei pazienti
accanto alla diagnosi: e' un dato sanitario identificabile e NON deve entrare
nel repository ne' essere condiviso.

Questo script produce un CSV ridotto ai soli campi necessari all'addestramento
(GUID pseudonimo + LABEL), scartando nome e ID progressivo. Va eseguito una
sola volta, in locale, sui dati grezzi.

Uso:
    python scripts/anonymize_csv.py data/carbidopa.csv data/labels.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def anonymize(src: Path, dst: Path) -> None:
    df = pd.read_csv(src, dtype=str)
    keep = [c for c in ("GUID", "LABEL") if c in df.columns]
    if "GUID" not in keep or "LABEL" not in keep:
        raise SystemExit(f"Colonne GUID/LABEL non trovate in {src}. Colonne: {list(df.columns)}")
    out = df[keep].copy()
    out.to_csv(dst, index=False)
    print(f"Scritto {dst} con {len(out)} righe; nessun nome/identificativo diretto.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    anonymize(Path(sys.argv[1]), Path(sys.argv[2]))
