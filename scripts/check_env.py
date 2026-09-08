"""Phase 1.1a gate — verify the ML env and that the Phase 1 pickles still load.

Run:  .venv/Scripts/python.exe scripts/check_env.py

Exits non-zero on any failure. Nothing else in the real-dataset build should run
until this passes (spec real_dataset_setup.md Phase 1.1a).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

# (module, attribute holding the version, expected value or None to just report)
PINS = [
    ("numpy", "__version__", "2.2.6"),
    ("scipy", "__version__", "1.15.3"),
    ("sklearn", "__version__", "1.7.2"),
    ("qiskit", "__version__", "2.4.1"),
    ("qiskit_machine_learning", "__version__", "0.9.0"),
]

# Track A deps that live in the MAIN env. pyannote.audio + deepfilternet do NOT —
# they force numpy<2 and corrupt this env, so speech_02_clean.py runs from a
# separate .venv-clean/ (see scripts/requirements-clean.txt).
OPTIONAL = ["librosa", "soundfile", "torch", "torchaudio", "transformers"]

problems: list[str] = []


def check_pins() -> None:
    for mod_name, attr, expected in PINS:
        try:
            mod = __import__(mod_name)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"import {mod_name} failed: {exc!r}")
            continue
        got = getattr(mod, attr, "?")
        flag = "" if (expected is None or got == expected) else f"  <-- expected {expected}"
        if flag:
            problems.append(f"{mod_name} {got} != {expected}")
        print(f"  {mod_name:<26} {got}{flag}")


def check_optional() -> None:
    for mod_name in OPTIONAL:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "(no __version__)")
            print(f"  {mod_name:<26} {ver}")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{mod_name} not importable: {exc.__class__.__name__}")
            print(f"  {mod_name:<26} NOT INSTALLED ({exc.__class__.__name__})")


def check_cuda() -> None:
    try:
        import torch

        ok = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if ok else "-"
        print(f"  torch.cuda.is_available() = {ok}   device: {name}")
        if not ok:
            problems.append("CUDA not available to torch (RTX 4050 expected)")
    except Exception as exc:  # noqa: BLE001
        print(f"  torch not usable: {exc!r}")
        problems.append("torch import/CUDA check failed")


def check_pickles() -> None:
    import numpy as np

    for name in ("qsvm_model.pkl", "svm_model.pkl"):
        path = RESULTS / name
        if not path.exists():
            problems.append(f"{name} missing at {path}")
            continue
        try:
            with path.open("rb") as fh:
                model = pickle.load(fh)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{name} failed to unpickle: {exc!r}")
            continue

        # The Phase 1 pickles are full sklearn Pipelines that take the raw
        # 27-feature row (per-modality PCA + ZZFeatureMap fusion live inside).
        row = np.zeros((1, 27), dtype=float)
        try:
            if hasattr(model, "decision_function"):
                val = float(model.decision_function(row)[0])
                print(f"  {name:<16} loaded OK  decision_function(zeros)={val:+.4f}")
            elif hasattr(model, "predict"):
                val = model.predict(row)[0]
                print(f"  {name:<16} loaded OK  predict(zeros)={val}")
            else:
                problems.append(f"{name} has neither decision_function nor predict")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{name} loaded but inference raised: {exc!r}")


def main() -> int:
    print("== frozen pins ==")
    check_pins()
    print("\n== optional / Track A deps ==")
    check_optional()
    print("\n== CUDA ==")
    check_cuda()
    print("\n== Phase 1 pickles ==")
    check_pickles()

    print("\n" + ("-" * 60))
    if problems:
        print("FAIL:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("OK — environment is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
