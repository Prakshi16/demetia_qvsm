"""
Speech feature extraction.

This placeholder implementation generates a deterministic 18-feature
speech vector from an uploaded audio file.

The output order MUST remain:

[
    pause_rate,
    speech_rate,
    pitch_mean,
    jitter,
    shimmer,
    mfcc_1,
    mfcc_2,
    ...
    mfcc_13
]

When real audio processing (librosa/parselmouth) is added later,
only the body of extract_speech_features() should change.
"""

from __future__ import annotations

import hashlib
import random


FEATURE_ORDER = [
    "pause_rate",
    "speech_rate",
    "pitch_mean",
    "jitter",
    "shimmer",
    *[f"mfcc_{i}" for i in range(1, 14)],
]


def extract_speech_features(file) -> list[float]:
    """
    Generate an 18-element speech feature vector.

    Args:
        file:
            A file-like object or bytes containing the uploaded audio.

    Returns:
        list[float]:
            18 speech features in FEATURE_ORDER.
    """

    file_bytes = file.read() if hasattr(file, "read") else bytes(file)

    seed = int(hashlib.md5(file_bytes).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)

    # Acoustic features
    pause_rate = round(rng.uniform(0.05, 0.35), 4)
    speech_rate = round(rng.uniform(2.5, 5.5), 4)
    pitch_mean = round(rng.uniform(80.0, 250.0), 4)
    jitter = round(rng.uniform(0.001, 0.03), 5)
    shimmer = round(rng.uniform(0.01, 0.08), 5)

    # MFCC features
    mfcc_ranges = [
        (-60, 60),
        (-40, 40),
        (-30, 30),
        (-25, 25),
        (-20, 20),
        (-18, 18),
        (-15, 15),
        (-12, 12),
        (-10, 10),
        (-8, 8),
        (-7, 7),
        (-6, 6),
        (-5, 5),
    ]

    mfccs = [
        round(rng.uniform(low, high), 4)
        for low, high in mfcc_ranges
    ]

    features = [
        pause_rate,
        speech_rate,
        pitch_mean,
        jitter,
        shimmer,
        *mfccs,
    ]

    assert len(features) == 18, f"Expected 18 features, got {len(features)}"

    return [float(value) for value in features]