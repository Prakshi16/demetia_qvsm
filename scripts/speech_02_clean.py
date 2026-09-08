"""Phase 1.2 step 2 — clean the speech corpus (denoise, optionally diarize).

Output:
  data/audio_clean/<speaker_id>/<clip_stem>.wav   16 kHz mono, denoised
  data/audio_clean/_clean_log.csv                 clip_path, status, dom_speaker_frac, n_speakers

status in {diarized_denoised, diarized_only, denoise_only, single_speaker,
           fallback_raw}. Every input clip always yields a 16 kHz mono output so
speech_03 `clean` can run over the whole corpus regardless of what degraded.

The keep/drop decision on cleaning is made later on grouped CV (spec Phase 1.9);
this step just produces the cleaned variant alongside the raw one. Spec Appendix D
already lists "is cleaning worth the backend-parity cost" as an open question.

DENOISE is on by default (DeepFilterNet3, works fine).
DIARIZATION is OFF by default: pyannote/speaker-diarization-3.1 pulls speechbrain,
which *segfaults on import* in this env on Windows (uncatchable — kills the
process). To attempt it anyway, set SPEECH_DIARIZE=1. Interviewer removal is
otherwise a documented follow-up (revisit on Linux / with a fixed speechbrain).

RUN FROM .venv-clean/ — NOT the main .venv/. pyannote.audio + deepfilternet pin
numpy<2.0 and corrupt the main env. This script only writes wav files, so the
env split is invisible downstream.

Run:  .venv-clean/Scripts/python.exe scripts/speech_02_clean.py
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from _speech_common import CLEAN_DIR, DATA, REPO, TARGET_SR, rel

warnings.filterwarnings("ignore")

DFN_SR = 48_000
LOG = CLEAN_DIR / "_clean_log.csv"


def _patch_hf_hub_use_auth_token() -> None:
    """pyannote.audio 3.4 passes the long-removed `use_auth_token=` kwarg straight
    through to huggingface_hub download fns. On hf_hub >= 0.26 that raises
    TypeError. Translate it to `token=` (and drop it when None/True)."""
    import huggingface_hub as hh

    def wrap(fn):
        def inner(*a, **kw):
            if "use_auth_token" in kw:
                val = kw.pop("use_auth_token")
                if isinstance(val, str):
                    kw.setdefault("token", val)
            return fn(*a, **kw)

        return inner

    for name in ("hf_hub_download", "snapshot_download"):
        if hasattr(hh, name) and not getattr(getattr(hh, name), "_uat_patched", False):
            patched = wrap(getattr(hh, name))
            patched._uat_patched = True
            setattr(hh, name, patched)


def load_diarizer():
    if os.environ.get("SPEECH_DIARIZE") != "1":
        print("  diarization OFF (set SPEECH_DIARIZE=1 to attempt) — denoise only.")
        return None
    try:
        import torch

        _patch_hf_hub_use_auth_token()
        from pyannote.audio import Pipeline

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or True
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
        if pipe is None:
            print("  pyannote returned None — gated model not accepted for this token.")
            return None
        if torch.cuda.is_available():
            pipe.to(torch.device("cuda"))
        return pipe
    except Exception as exc:  # noqa: BLE001
        print(f"  diarizer unavailable ({exc.__class__.__name__}: {exc}) — will not diarize.")
        return None


def load_denoiser():
    try:
        from df.enhance import enhance, init_df

        model, state, _ = init_df()
        return (enhance, model, state)
    except Exception as exc:  # noqa: BLE001
        print(f"  denoiser unavailable ({exc.__class__.__name__}: {exc}) — will not denoise.")
        return None


def dominant_segments(diarization, total_s: float):
    """Return (list[(start,end)] for the longest-talking speaker, that speaker's frac, n_speakers)."""
    spk_time: dict[str, float] = {}
    for seg, _, spk in diarization.itertracks(yield_label=True):
        spk_time[spk] = spk_time.get(spk, 0.0) + (seg.end - seg.start)
    if not spk_time:
        return None, 0.0, 0
    dom = max(spk_time, key=spk_time.get)
    frac = spk_time[dom] / max(total_s, 1e-6)
    segs = [
        (seg.start, seg.end)
        for seg, _, spk in diarization.itertracks(yield_label=True)
        if spk == dom
    ]
    return sorted(segs), frac, len(spk_time)


def main() -> int:
    clips = pd.read_csv(DATA / "speech_clips.csv")
    diarizer = load_diarizer()
    denoiser = load_denoiser()

    log_rows: list[dict] = []
    for i, clip in clips.iterrows():
        src = REPO / clip["clip_path"]
        speaker_id = clip["speaker_id"]
        out_dir = CLEAN_DIR / speaker_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (src.stem + ".wav")

        y, _ = librosa.load(str(src), sr=TARGET_SR, mono=True)  # soundfile backend: handles float wav
        total_s = len(y) / TARGET_SR
        status = "fallback_raw"
        dom_frac, n_spk = 1.0, 1

        if diarizer is not None:
            try:
                import torch

                diar = diarizer(
                    {"waveform": torch.from_numpy(y).unsqueeze(0), "sample_rate": TARGET_SR}
                )
                segs, dom_frac, n_spk = dominant_segments(diar, total_s)
                if segs and n_spk > 1:
                    keep = np.concatenate(
                        [y[int(a * TARGET_SR) : int(b * TARGET_SR)] for a, b in segs]
                    )
                    if len(keep) > TARGET_SR:  # at least 1 s survived
                        y = keep
                        status = "diarized_only"
                elif n_spk == 1:
                    status = "single_speaker"
            except Exception as exc:  # noqa: BLE001
                print(f"  diar fail {src.name}: {exc!r}")

        if denoiser is not None:
            try:
                import torch

                enhance, model, state = denoiser
                y48 = librosa.resample(y, orig_sr=TARGET_SR, target_sr=DFN_SR)
                t48 = torch.from_numpy(np.ascontiguousarray(y48)).float().unsqueeze(0)
                enh = enhance(model, state, t48)
                y48 = (enh.squeeze().cpu().numpy() if hasattr(enh, "cpu")
                       else np.asarray(enh).squeeze())
                y = librosa.resample(y48, orig_sr=DFN_SR, target_sr=TARGET_SR)
                status = "diarized_denoised" if status == "diarized_only" else (
                    "denoise_only" if status in {"fallback_raw", "single_speaker"} else status
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  denoise fail {src.name}: {exc!r}")

        y = np.clip(y, -1.0, 1.0).astype(np.float32)
        sf.write(str(out_path), y, TARGET_SR, subtype="PCM_16")
        log_rows.append(
            {
                "clip_path": clip["clip_path"],
                "status": status,
                "dom_speaker_frac": round(float(dom_frac), 3),
                "n_speakers": int(n_spk),
            }
        )
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(clips)} clips")

    log = pd.DataFrame(log_rows)
    log.to_csv(LOG, index=False)
    print(f"\nwrote {rel(LOG)}  ({len(log)} clips)")
    print(log["status"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
