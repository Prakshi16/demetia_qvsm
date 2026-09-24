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

The fused QSVM ("Quantum SVM") is the canonical output; the classical SVM box is
**display-only**, shown for comparison. Neither is degenerate on the retrained model — on the
real evaluation set (`results/metrics_full.json`), single-modality accuracy is clinical ~72%/69%
(SVM/QSVM), MRI ~78%/78%, speech ~64%/56%, and the **fused** model (~81%/78%) beats every single
modality — there's no one block that dominates the decision the way MRI did on the old synthetic
set.

## Recipes

Pick one MRI + one audio, any format. These files are still synthetic phantoms (intensity/geometry
shapes and tone bursts, not real anatomy or speech — see the note at the top), so their exact
`nWBV`/`eTIV` values sit well outside the OASIS training range and don't have a principled
"should be higher/lower for dementia" ordering the way a real scan would.

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
