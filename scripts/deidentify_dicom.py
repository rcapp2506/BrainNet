"""De-identificazione degli header DICOM.

ATTENZIONE: anonimizzare il NOME della cartella (GUID) non basta. I file .dcm
contengono PHI negli header (PatientName, PatientID, PatientBirthDate,
InstitutionName, ReferringPhysicianName, StudyDate, AccessionNumber, ...).
Questo script li ripulisce, scrivendo una copia de-identificata.

Strategia (allineata in spirito al profilo DICOM PS3.15 Annex E):
  * gli identificatori diretti vengono rimossi o sostituiti con il GUID di cartella;
  * le date vengono azzerate (o, opzionalmente, troncate all'anno);
  * i tag privati vengono rimossi;
  * i pixel e la geometria necessari al modello restano intatti.

Per una pipeline pienamente conforme allo standard considera la libreria `deid`
(https://pydicom.github.io/deid/) che implementa il profilo completo.

Uso:
    python scripts/deidentify_dicom.py data/CDOPA data/CDOPA_DEID
    python scripts/deidentify_dicom.py data/CDOPA data/CDOPA_DEID --keep-year
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pydicom

# Tag da svuotare completamente (identificatori diretti o quasi).
BLANK_TAGS = [
    "PatientBirthDate", "PatientBirthTime", "PatientAddress",
    "OtherPatientIDs", "OtherPatientNames", "PatientTelephoneNumbers",
    "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName",
    "ReferringPhysicianName", "PerformingPhysicianName", "OperatorsName",
    "PhysiciansOfRecord", "RequestingPhysician", "NameOfPhysiciansReadingStudy",
    "AccessionNumber", "StationName", "DeviceSerialNumber",
    "StudyTime", "SeriesTime", "AcquisitionTime", "ContentTime",
    "StudyID", "PerformedProcedureStepID", "ScheduledProcedureStepID",
]

# Date: azzerate, oppure troncate all'anno (YYYY0101) con --keep-year.
DATE_TAGS = ["StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate"]


def deidentify_file(path: Path, out_path: Path, pseudo_id: str, keep_year: bool) -> None:
    ds = pydicom.dcmread(str(path))

    # Identificatori paziente -> pseudonimo (il GUID di cartella).
    ds.PatientName = pseudo_id
    ds.PatientID = pseudo_id

    for tag in BLANK_TAGS:
        if tag in ds:
            ds.data_element(tag).value = ""

    for tag in DATE_TAGS:
        if tag in ds and str(ds.data_element(tag).value):
            v = str(ds.data_element(tag).value)
            ds.data_element(tag).value = (v[:4] + "0101") if keep_year and len(v) >= 4 else ""

    ds.remove_private_tags()

    # Marca la de-identificazione (richiesto dal profilo).
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = "BrainNet curated tag scrub"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(out_path))


def main() -> None:
    ap = argparse.ArgumentParser(description="De-identifica gli header DICOM.")
    ap.add_argument("src", type=Path, help="cartella radice con sottocartelle <GUID>/")
    ap.add_argument("dst", type=Path, help="cartella di output de-identificata")
    ap.add_argument("--keep-year", action="store_true",
                    help="conserva l'anno nelle date (YYYY0101) invece di azzerarle")
    args = ap.parse_args()

    n = 0
    for guid_dir in sorted(p for p in args.src.iterdir() if p.is_dir()):
        pseudo = guid_dir.name  # il GUID e' gia' lo pseudonimo
        for dcm in guid_dir.glob("*.dcm"):
            deidentify_file(dcm, args.dst / pseudo / dcm.name, pseudo, args.keep_year)
            n += 1
    print(f"De-identificati {n} file in {args.dst}. "
          f"Verifica con: python scripts/check_phi.py {args.dst}")


if __name__ == "__main__":
    main()
