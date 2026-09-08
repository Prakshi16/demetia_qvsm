"""Guardrail — assert the Track A output CSVs match Appendix A of real_dataset_setup.md
exactly, so schema drift is caught before Phase 1.5 / 1.6 consume them.

Run:  .venv/Scripts/python.exe scripts/assert_contracts.py
"""

from __future__ import annotations

import sys

import pandas as pd

from _speech_common import DATA, SPEECH_FEATURES

CONTRACTS = {
    "speech_clips.csv": ["speaker_id", "speaker_folder", "clip_path", "label", "orig_split"],
    "speech_windows_raw.csv": ["speaker_id", "clip_path", "window_idx", *SPEECH_FEATURES],
    "speech_windows_clean.csv": ["speaker_id", "clip_path", "window_idx", *SPEECH_FEATURES],
    "speech_model_agesex.csv": ["speaker_id", "age_model", "sex_model", "n_clips"],
}

problems: list[str] = []
for name, cols in CONTRACTS.items():
    path = DATA / name
    if not path.exists():
        print(f"  - {name}: not produced yet (skip)")
        continue
    got = list(pd.read_csv(path, nrows=0).columns)
    if got != cols:
        problems.append(f"{name}\n      expected {cols}\n      got      {got}")
    else:
        print(f"  ok  {name}")

if problems:
    print("\nCONTRACT MISMATCH:")
    for p in problems:
        print(f"  - {p}")
    sys.exit(1)
print("\nall present contracts match Appendix A")
