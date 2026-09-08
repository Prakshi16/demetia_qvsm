"""Shared helpers for the Track A speech scripts (spec real_dataset_setup.md Phase 1.2)."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"

DEMENTIA_DIR = DATA / "dementia-audio"
NODEMENTIA_DIR = DATA / "nodementia-audio"
CLEAN_DIR = DATA / "audio_clean"

TARGET_SR = 16_000
WINDOW_S = 30.0
HOP_S = 10.0

# The 18 features extract_speech_features() returns, in order (Appendix A).
SPEECH_FEATURES = [
    "pause_rate",
    "speech_rate",
    "pitch_mean",
    "jitter",
    "shimmer",
    *[f"mfcc_{i}" for i in range(1, 14)],
]

# Folder-name -> canonical speaker_id overrides (apostrophe / punctuation oddities).
_SLUG_OVERRIDES = {
    "ronan o_rahilly": "ronanorahilly",
}


def slugify_speaker(name: str) -> str:
    """'Abe Burrows' -> 'abeburrows'; strips all spaces and punctuation."""
    low = name.strip().lower()
    if low in _SLUG_OVERRIDES:
        return _SLUG_OVERRIDES[low]
    return re.sub(r"[^a-z0-9]+", "", low)


def rel(path: Path) -> str:
    """Repo-relative POSIX path string."""
    return str(Path(path).resolve().relative_to(REPO)).replace("\\", "/")


def iter_speaker_folders():
    """Yield (folder_path, label) for every speaker folder in both roots."""
    for root, label in ((DEMENTIA_DIR, 1), (NODEMENTIA_DIR, 0)):
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            yield folder, label
