"""Barre di avanzamento opzionali.

Usa tqdm se installato (in requirements.txt); altrimenti degrada a no-op senza
rompere nulla, restituendo l'iterabile cosi' com'e'.
"""
from __future__ import annotations

try:
    from tqdm.auto import tqdm  # type: ignore
    HAS_TQDM = True

    def pbar(iterable, desc: str = "", leave: bool = False):
        return tqdm(iterable, desc=desc, leave=leave, dynamic_ncols=True)

except Exception:  # tqdm non disponibile
    HAS_TQDM = False

    def pbar(iterable, desc: str = "", leave: bool = False):
        if desc:
            print(f"  ... {desc}", flush=True)
        return iterable
