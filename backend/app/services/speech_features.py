"""
Real speech feature extraction using librosa.

Output order:

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
"""

from __future__ import annotations

import io
import numpy as np
import librosa


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
    Extract 18 real speech features from an uploaded audio file.

    Args:
        file:
            File-like object or bytes containing audio.

    Returns:
        list[float]:
            18 speech features in FEATURE_ORDER.
    """

    # ---------------------------------------------------------
    # 1. Read uploaded audio
    # ---------------------------------------------------------

    file_bytes = file.read() if hasattr(file, "read") else bytes(file)

    if not file_bytes:
        raise ValueError("Audio file is empty.")

    # librosa can load audio from a BytesIO object
    audio_buffer = io.BytesIO(file_bytes)

    # Load audio
    y, sr = librosa.load(
        audio_buffer,
        sr=16000,
        mono=True,
    )

    if len(y) == 0:
        raise ValueError("Could not extract audio samples.")

    # ---------------------------------------------------------
    # 2. Pause rate
    # ---------------------------------------------------------
    #
    # Detect silent portions using RMS energy.
    #

    rms = librosa.feature.rms(y=y)[0]

    threshold = np.percentile(rms, 20)

    silent_frames = rms < threshold

    pause_rate = float(np.mean(silent_frames))

    # ---------------------------------------------------------
    # 3. Speech rate
    # ---------------------------------------------------------
    #
    # Approximate speech rate using onset events.
    #

    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr
    )

    duration = librosa.get_duration(y=y, sr=sr)

    if duration > 0:
        speech_rate = len(onset_frames) / duration
    else:
        speech_rate = 0.0

    # ---------------------------------------------------------
    # 4. Pitch mean
    # ---------------------------------------------------------

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
    )

    valid_pitch = f0[np.isfinite(f0)]

    if len(valid_pitch) > 0:
        pitch_mean = float(np.mean(valid_pitch))
    else:
        pitch_mean = 0.0

    # ---------------------------------------------------------
    # 5. Jitter
    # ---------------------------------------------------------
    #
    # Jitter = variation between consecutive pitch periods.
    #

    if len(valid_pitch) > 1:

        periods = 1.0 / valid_pitch

        jitter = float(
            np.mean(
                np.abs(np.diff(periods))
            )
            / np.mean(periods)
        )

    else:
        jitter = 0.0

    # ---------------------------------------------------------
    # 6. Shimmer
    # ---------------------------------------------------------
    #
    # Approximate amplitude variation between frames.
    #

    frame_rms = librosa.feature.rms(y=y)[0]

    if len(frame_rms) > 1 and np.mean(frame_rms) > 0:

        shimmer = float(
            np.mean(np.abs(np.diff(frame_rms)))
            / np.mean(frame_rms)
        )

    else:
        shimmer = 0.0

    # ---------------------------------------------------------
    # 7. MFCCs
    # ---------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13,
    )

    # Take mean of each MFCC coefficient across time
    mfcc_means = np.mean(mfcc, axis=1)

    # ---------------------------------------------------------
    # 8. Build final 18-feature vector
    # ---------------------------------------------------------

    features = [
        pause_rate,
        speech_rate,
        pitch_mean,
        jitter,
        shimmer,
        *mfcc_means.tolist(),
    ]

    # ---------------------------------------------------------
    # 9. Safety checks
    # ---------------------------------------------------------

    if len(features) != 18:
        raise ValueError(
            f"Expected 18 speech features, got {len(features)}"
        )

    # Replace NaN / infinity if audio produces them
    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return [float(round(value, 4)) for value in features]