# Synthetic demo inputs

All files here are **synthetic**. The MRI files are intensity/geometry phantoms,
not anatomy; the audio is tone bursts, not speech. Nothing here is a real
patient and the model labels carry no clinical meaning. They exist only to drive
the screening flow end to end and to demo a "Demented" vs "Nondemented" result.

## What the model actually sees

Each screening builds one 27-feature row:

| Block | Features | Source |
|---|---|---|
| Clinical (5) | CDR, MMSE, ASF, EDUC, SES | the "Clinical measures" form (ASF is derived from eTIV) |
| MRI (4) | nWBV, eTIV, hippocampal volume, cortical thickness | MRI upload → `mri_features.py` extractor |
| Speech (18) | pause rate, speech rate, pitch, jitter, shimmer, MFCC 1–13 | speech upload → `speech_features.py` extractor |

The fused QSVM ("Quantum SVM") is the canonical output. The classical SVM box is
**display-only** — on this synthetic data it is degenerate and always prints
`Nondemented, margin 0.02`. That is expected, not a bug.

Empirically the **MRI block dominates** the QSVM decision. The clinical form
nudges the margin but rarely flips the label.

## Recipes

Pick one MRI + one audio, any format. Clinical values below make the result
robust; the MRI file alone gets you ~90% of the way.

### → Nondemented
- MRI: `mri/nondemented.*`
- Audio: `audio/fluent.*`
- Clinical: CDR `0`, MMSE `29`, Education `16`, SES `3`
- Result: QSVM **Nondemented**, margin ≈ 0.19

### → Demented
- MRI: `mri/demented.*`
- Audio: `audio/hesitant.*`
- Clinical: CDR `0.5`+, MMSE `24` or lower, Education `14`, SES `3`
- Result: QSVM **Demented**, margin ≈ 0.28–0.36 (holds across all clinical presets)

### Borderline (shows model uncertainty / clinician-override demo)
- MRI: `mri/borderline.*` + any audio
- Result: sits near the decision boundary, small margins, flips with clinical
  input — good for demoing "the clinician disagrees with the model".

## Formats

- MRI: `.nii.gz` `.nii` `.mgz` `.mgh` `.dcm` — identical feature vector per phantom.
- Audio: `.wav` `.mp3` `.m4a` `.webm` — `.m4a`/`.webm` exercise the server-side
  transcode path; feature vectors match within rounding.
