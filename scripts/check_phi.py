"""Verifica PHI: controlla che gli header DICOM siano de-identificati.

Da eseguire DOPO deidentify_dicom.py e prima di qualsiasi condivisione.
Esce con codice 1 se trova ancora tag identificativi valorizzati.

Uso:
    python scripts/check_phi.py data/CDOPA_DEID
"""
from __future__ import annotations

import sys
from pathlib import Path

import pydicom

# Tag che NON devono risultare valorizzati con dati reali dopo la de-identificazione.
SUSPECT = [
    "PatientBirthDate", "PatientAddress", "OtherPatientIDs", "OtherPatientNames",
    "InstitutionName", "InstitutionAddress", "ReferringPhysicianName",
    "PerformingPhysicianName", "OperatorsName", "AccessionNumber",
    "StationName", "DeviceSerialNumber", "StudyDate",
]


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/CDOPA_DEID")
    findings: dict[str, int] = {}
    checked = 0

    for dcm in root.rglob("*.dcm"):
        ds = pydicom.dcmread(str(dcm), stop_before_pixels=True)
        checked += 1
        for tag in SUSPECT:
            if tag in ds and str(ds.data_element(tag).value).strip() not in ("", "None"):
                findings[tag] = findings.get(tag, 0) + 1
        # Il nome paziente deve coincidere con lo pseudonimo (GUID = nome cartella).
        if "PatientName" in ds and str(ds.PatientName) != dcm.parent.name:
            findings["PatientName!=GUID"] = findings.get("PatientName!=GUID", 0) + 1

    print(f"Controllati {checked} file in {root}")
    if findings:
        print("ATTENZIONE: residui PHI rilevati:")
        for k, v in sorted(findings.items()):
            print(f"  {k}: {v} file")
        return 1
    print("OK: nessun residuo PHI nei tag controllati.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
