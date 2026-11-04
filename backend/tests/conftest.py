"""Shared fixtures for the backend unit tests.

Run from the repo root with the project venv:

    python -m pytest backend/tests -q

No database or network is touched: API tests override the auth and DB
dependencies, and everything else is pure functions. Tests that need the real
OASIS scans (gitignored data/) skip themselves when the data isn't present.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

# Settings() needs these at import time. Use the real .env when it exists;
# otherwise dummy values are fine because no test connects anywhere.
if not (BACKEND / ".env").exists():
    os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost:5432/test")
    os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
    os.environ.setdefault("SUPABASE_KEY", "test-key")
    os.environ.setdefault("JWT_SECRET", "test-secret")
os.environ.setdefault("MODEL_DIR", str(REPO / "results"))

DATA = REPO / "data"
OAS1_SUBJECT = DATA / "oasis1_MRI_scans" / "OAS1_0001_MR1"
OAS2_SESSION = DATA / "OAS2_RAW_PART1" / "OAS2_0001_MR1"
needs_oasis = pytest.mark.skipif(
    not (OAS1_SUBJECT.exists() and OAS2_SESSION.exists()),
    reason="real OASIS scans (gitignored data/) not present",
)


def make_head(shape=(96, 112, 96), atrophy=0.3, skull=True, seed=0) -> np.ndarray:
    """A crude T1-like head phantom.

    Layers from outside in: scalp (bright), skull (dark), CSF (dark), grey
    matter, white matter (brightest), ventricles (CSF). ``atrophy`` (0..1)
    widens the CSF layer and ventricles, which is what nWBV measures.
    """
    rng = np.random.default_rng(seed)
    z, y, x = np.indices(shape, dtype=np.float32)
    cz, cy, cx = (np.array(shape) - 1) / 2
    rz, ry, rx = np.array(shape) * 0.42
    r = np.sqrt(((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2)
    vol = np.zeros(shape, np.float32)
    brain_edge = 0.86 if skull else 1.0
    if skull:
        vol[(r <= 1.0) & (r > 0.95)] = 0.8   # scalp
        vol[(r <= 0.95) & (r > 0.88)] = 0.08  # skull
    csf_edge = brain_edge - 0.02 - 0.08 * atrophy
    vol[(r <= brain_edge) & (r > csf_edge)] = 0.15  # surface CSF
    vol[(r <= csf_edge) & (r > csf_edge - 0.12)] = 0.45  # grey matter
    vol[r <= csf_edge - 0.12] = 0.7  # white matter
    vent = np.sqrt(((z - cz) / (rz * 0.18)) ** 2 + ((y - cy) / (ry * (0.18 + 0.25 * atrophy))) ** 2 + ((x - cx) / (rx * 0.12)) ** 2)
    vol[vent <= 1.0] = 0.12  # ventricles
    vol += rng.normal(0, 0.02, shape).astype(np.float32)
    return np.clip(vol, 0, 1) * 1000.0


@pytest.fixture
def head_volume() -> np.ndarray:
    return make_head()


@pytest.fixture
def nifti_bytes(head_volume) -> bytes:
    import nibabel as nib

    return nib.Nifti1Image(head_volume, np.diag([2.0, 2.0, 2.0, 1.0])).to_bytes()
