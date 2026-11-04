# Synthetic demo inputs

All files here are **synthetic**. The MRI files are intensity/geometry phantoms,
not anatomy; the audio is tone bursts, not speech. Nothing here is a real
patient and the model labels carry no clinical meaning. They exist only to drive
the screening flow end to end and to demo a "Demented" vs "Nondemented" result.

## What the model actually sees

**2026-09-24: the model was retrained on real data** (OASIS clinical/MRI cross-matched with
windowed speech; see `real_dataset_setup.md` / `PROJECT_CONTEXT.md` §28-29). The feature layout
changed from 27 to 24 — `CDR` was dropped from the clinical block (OASIS's label is CDR-derived,
so keeping it would leak the label) and the MRI block shrank from 4 to 2 (`hippocampal_volume` /
`cortical_thickness` were synthetic-only estimates with no OASIS tabular counterpart).

Each screening builds one 24-feature row:

| Block | Features | Source |
|---|---|---|
| Clinical (4) | MMSE, ASF, EDUC, SES | the "Clinical measures" form (ASF is derived from eTIV; CDR is collected but not a model input) |
| MRI (2) | nWBV, eTIV | MRI upload → `mri_features.py` extractor |
| Speech (18) | pause rate, speech rate, pitch, jitter, shimmer, MFCC 1–13 | speech upload → `speech_features.py` extractor |

**2026-09-25: the classical SVM is now the only model the app serves** — the QSVM box has been
removed from the screen. On the real evaluation set (`results/metrics_full.json`), the fused
classical SVM (81.1%) beats the fused QSVM (78.1%) on every metric but specificity, so the app was
switched from QSVM-canonical to SVM-canonical (see `backend/app/services/prediction.py`'s
2026-09-25 note). Single-modality accuracy for context: clinical ~72%/69% (SVM/QSVM), MRI ~78%/78%,
speech ~64%/56% — the **fused** SVM beats every single modality, and no one block dominates the
decision the way MRI did on the old synthetic set. The QSVM is still trained and evaluated in
`multimodal_qsvm.ipynb`/`metrics_full.ipynb` for the research comparison; it just isn't wired into
the live screening flow any more.

## 2026-11-04 update: extractor v2, any MRI format, QSVM toggle

- **The MRI phantoms here are now flagged, not trusted.** Extractor v2 is calibrated on real OASIS
  scans, and these geometric phantoms sit far outside real MRI statistics. They still upload and
  drive the flow end to end, but expect the warning *"The estimate fell outside the range seen in
  training"* and clamped values. For a meaningful demo, upload a real T1 scan in any format the
  app accepts: NIfTI, MGZ, an Analyze `.hdr`+`.img` pair (select both files), a DICOM file or
  series (or a `.zip` of one), or a PNG/JPG of a single slice.
- **Both models now run on every screening.** The Visit Detail and screening screens have a
  Classical SVM · Quantum SVM · Compare switch. The classical SVM is still the canonical result.

## Recipes

Pick one MRI + one audio, any format. These files are still synthetic phantoms (intensity/geometry
shapes and tone bursts, not real anatomy or speech — see the note at the top), so their exact
`nWBV`/`eTIV` values sit well outside the OASIS training range and don't have a principled
"should be higher/lower for dementia" ordering the way a real scan would. **2026-09-25:** the MRI
extractor was calibrated against 39 real OASIS-1 scans (`PROJECT_CONTEXT.md` §31) — it now handles
*real* anatomy well (see `results/mri_extractor_calibration.pkl`'s LOOCV numbers), but these
geometric phantoms are exactly the kind of far-out-of-distribution input the calibration wasn't fit
to, so their behavior is unverified either way. **Prefer real scans for a demo** — see
`data/oasis1_MRI_scans/` for real, already-uploadable-after-conversion OASIS-1 subjects (convert
`.img`→`.nii.gz` with `nibabel`; `PROJECT_CONTEXT.md` §31.3 has three verified real
Nondemented/Borderline/Demented cases with their true `nWBV` values).

**Margins below are not re-verified against the retrained model** (2026-09-24: both the Docker
image build and the local `librosa`/`numba` speech path were blocked in the environment this
migration was done in — see `PROJECT_CONTEXT.md` §29). Re-run these three combinations through a
live `docker compose up` before quoting specific margins again; until then, treat the file picks
below as a starting point for the demo, not a guaranteed label/margin.

### → Nondemented (starting point, re-verify margin)
- MRI: `mri/nondemented.*`
- Audio: `audio/fluent.*`
- Clinical: MMSE `29`, Education `16`, SES `3`

### → Demented (starting point, re-verify margin)
- MRI: `mri/demented.*`
- Audio: `audio/hesitant.*`
- Clinical: MMSE `22` or lower, Education `14`, SES `3`

### Borderline (shows model uncertainty / clinician-override demo)
- MRI: `mri/borderline.*` + any audio
- Intended to sit near the decision boundary — re-verify after a live run rather than assuming
  small margins.

## Formats

- MRI: `.nii.gz` `.nii` `.mgz` `.mgh` `.dcm` — identical feature vector per phantom.
- Audio: `.wav` `.mp3` `.m4a` `.webm` — `.m4a`/`.webm` exercise the server-side
  transcode path; feature vectors match within rounding.
