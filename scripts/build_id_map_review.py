"""Phase 1.1c — one row per OASIS MRI scan folder on disk, for Sheetal's
ID-map review (spec real_dataset_setup.md Phase 1.4).

Output: data/_id_map_review.csv
  mri_id, oasis_source, oasis_subject_id, folder_path, in_clinical_xlsx, n_sessions

Run:  .venv/Scripts/python.exe scripts/build_id_map_review.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUT = DATA / "_id_map_review.csv"

MRI_ID_RE = re.compile(r"^OAS[12]_\d{4}_MR\d+$")


def subject_of(mri_id: str) -> str:
    """OAS2_0001_MR2 -> OAS2_0001"""
    return "_".join(mri_id.split("_")[:2])


def scan_folders(root: Path, source: str) -> list[dict]:
    rows: list[dict] = []
    if not root.exists():
        print(f"  WARNING: {root} does not exist — skipping {source}")
        return rows
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or not MRI_ID_RE.match(entry.name):
            if entry.name not in {".", ".."}:
                print(f"  skipped non-scan entry: {entry.name}")
            continue
        rows.append(
            {
                "mri_id": entry.name,
                "oasis_source": source,
                "oasis_subject_id": subject_of(entry.name),
                "folder_path": str(entry.relative_to(REPO)).replace("\\", "/"),
            }
        )
    return rows


def main() -> int:
    rows: list[dict] = []
    rows += scan_folders(DATA / "OAS2_RAW_PART1", "oasis2")
    rows += scan_folders(DATA / "oasis1_MRI_scans", "oasis1")

    df = pd.DataFrame(rows)

    # sessions per subject, from what is actually on disk
    df["n_sessions"] = df.groupby("oasis_subject_id")["mri_id"].transform("count")

    # clinical-xlsx membership
    o2 = pd.read_excel(DATA / "oasis2_mri_clinical.xlsx")
    o1 = pd.read_excel(DATA / "oasis1_clinical data.xlsx")
    o2_ids = set(o2["MRI ID"].astype(str))
    o1_ids = set(o1["ID"].astype(str))

    def in_xlsx(r: pd.Series) -> str:
        pool = o2_ids if r["oasis_source"] == "oasis2" else o1_ids
        return "Y" if r["mri_id"] in pool else "N"

    df["in_clinical_xlsx"] = df.apply(in_xlsx, axis=1)

    df = df.sort_values(["oasis_source", "mri_id"]).reset_index(drop=True)
    df = df[
        [
            "mri_id",
            "oasis_source",
            "oasis_subject_id",
            "folder_path",
            "in_clinical_xlsx",
            "n_sessions",
        ]
    ]
    df.to_csv(OUT, index=False)

    print(f"\nwrote {OUT.relative_to(REPO)}  ({len(df)} scan folders)")
    for src, g in df.groupby("oasis_source"):
        print(
            f"  {src}: {len(g)} sessions / {g['oasis_subject_id'].nunique()} subjects"
            f"  |  not in xlsx: {(g['in_clinical_xlsx'] == 'N').sum()}"
        )
    missing = df[df["in_clinical_xlsx"] == "N"]
    if not missing.empty:
        print("\n  scan folders with NO clinical row (Sheetal: investigate):")
        for mid in missing["mri_id"]:
            print(f"    {mid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
