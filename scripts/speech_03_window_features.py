"""Phase 1.2 steps 3-4 — window each clip and extract the 18 speech features per window.

  .venv/Scripts/python.exe scripts/speech_03_window_features.py raw
  .venv/Scripts/python.exe scripts/speech_03_window_features.py clean

raw   -> reads clip_path from data/speech_clips.csv        -> data/speech_windows_raw.csv
clean -> reads the matching file under data/audio_clean/   -> data/speech_windows_clean.csv

Windows: 30 s, 10 s hop; a clip < 30 s becomes one window = the whole clip.
Each window is handed to backend extract_speech_features() VERBATIM as a BytesIO
(the function takes a file-like/bytes object, NOT a path), so the 18-vector matches
the production /speech path exactly.

librosa.pyin costs ~5 s/window, so clips are processed across a process pool
(~16 cores here). Output columns (Appendix A):
  speaker_id, clip_path, window_idx, pause_rate, speech_rate, pitch_mean,
  jitter, shimmer, mfcc_1..mfcc_13
"""

from __future__ import annotations

import io
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from _speech_common import CLEAN_DIR, DATA, HOP_S, REPO, SPEECH_FEATURES, TARGET_SR, WINDOW_S, rel

sys.path.insert(0, str(REPO / "backend"))
from app.services.speech_features import extract_speech_features  # noqa: E402

N_WORKERS = min(12, (os.cpu_count() or 4))


def window_bounds(n_samples: int, sr: int) -> list[tuple[int, int]]:
    win = int(WINDOW_S * sr)
    hop = int(HOP_S * sr)
    if n_samples <= win:
        return [(0, n_samples)]
    return [(s, s + win) for s in range(0, n_samples - win + 1, hop)]


def process_clip(task: tuple[str, str, str]) -> list[dict]:
    """(speaker_id, clip_path_key, src_abs) -> one row dict per window."""
    speaker_id, clip_key, src = task
    try:
        y, _ = librosa.load(src, sr=TARGET_SR, mono=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  LOAD FAIL {src}: {exc!r}", flush=True)
        return []
    out: list[dict] = []
    for w_idx, (a, b) in enumerate(window_bounds(len(y), TARGET_SR)):
        buf = io.BytesIO()
        sf.write(buf, y[a:b], TARGET_SR, format="WAV", subtype="PCM_16")
        buf.seek(0)
        feats = extract_speech_features(buf)
        out.append(
            {
                "speaker_id": speaker_id,
                "clip_path": clip_key,
                "window_idx": w_idx,
                **dict(zip(SPEECH_FEATURES, feats)),
            }
        )
    return out


def clean_path_for(clip_rel: str, speaker_id: str) -> Path:
    return CLEAN_DIR / speaker_id / (Path(clip_rel).stem + ".wav")


def build_tasks(clips: pd.DataFrame, variant: str) -> tuple[list[tuple], int]:
    tasks, missing = [], 0
    for _, clip in clips.iterrows():
        key = clip["clip_path"]
        src = (
            REPO / key
            if variant == "raw"
            else clean_path_for(key, clip["speaker_id"])
        )
        if not Path(src).exists():
            missing += 1
            print(f"  MISSING {variant}: {src}")
            continue
        tasks.append((clip["speaker_id"], key, str(src)))
    return tasks, missing


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {"raw", "clean"}:
        print(__doc__)
        return 2
    variant = sys.argv[1]
    out_csv = DATA / f"speech_windows_{variant}.csv"

    clips = pd.read_csv(DATA / "speech_clips.csv")
    tasks, n_missing = build_tasks(clips, variant)
    print(f"{len(tasks)} clips to process on {N_WORKERS} workers "
          f"({n_missing} missing)...", flush=True)

    rows: list[dict] = []
    t0 = time.time()
    with Pool(N_WORKERS) as pool:
        for done, clip_rows in enumerate(pool.imap_unordered(process_clip, tasks), 1):
            rows.extend(clip_rows)
            if done % 20 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} clips  ({len(rows)} windows, "
                      f"{time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows, columns=["speaker_id", "clip_path", "window_idx", *SPEECH_FEATURES])
    df = df.sort_values(["speaker_id", "clip_path", "window_idx"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)

    feat = df[SPEECH_FEATURES].to_numpy()
    per_clip = df.groupby("clip_path").size()
    print(f"\nwrote {rel(out_csv)}")
    print(f"  windows:             {len(df)}")
    print(f"  clips covered:       {df['clip_path'].nunique()} / {len(clips)} (missing {n_missing})")
    print(f"  speakers:            {df['speaker_id'].nunique()}")
    print(f"  NaN/inf in features: {int((~np.isfinite(feat)).sum())}")
    print(f"  windows per clip:    min {per_clip.min()}, max {per_clip.max()}, "
          f"mean {per_clip.mean():.1f}")
    print(f"  elapsed:             {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
