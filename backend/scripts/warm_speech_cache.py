"""Run the speech extractor once so numba compiles (and caches) librosa's kernels.

Called from the Dockerfile at build time with NUMBA_CACHE_DIR pointing inside the
image and NUMBA_CPU_NAME=generic (so the cached machine code is valid on whatever
CPU the container later runs on). Without this, a cold container spends minutes
compiling on its first speech upload: measured 3 min 45 s on Render's free 0.1-CPU
instance, long enough for the platform's proxy to drop the request.

Needs no secrets or database: it imports only the feature extractor.
"""
from __future__ import annotations

import io
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.speech_features import extract_speech_features  # noqa: E402


def tone_wav(seconds: float = 2.0, rate: int = 16000) -> io.BytesIO:
    """Voiced bursts with pauses, so every code path (pitch, pauses, MFCC) runs."""
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    signal = 0.3 * np.sin(2 * np.pi * (130 + 20 * np.sin(2 * np.pi * 0.7 * t)) * t)
    signal[(t % 0.6) > 0.45] = 0.0
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((signal * 32767).astype(np.int16).tobytes())
    buffer.seek(0)
    return buffer


def main() -> int:
    started = time.perf_counter()
    features = extract_speech_features(tone_wav())
    first = time.perf_counter() - started
    started = time.perf_counter()
    extract_speech_features(tone_wav())
    second = time.perf_counter() - started
    print(f"speech extractor warm: {len(features)} features, first run {first:.1f}s, second {second:.1f}s")
    return 0 if len(features) == 18 else 1


if __name__ == "__main__":
    raise SystemExit(main())
