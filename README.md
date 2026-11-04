# Cortex Health Portal — Multimodal QSVM Dementia Detection

A hospital screening app that fuses **clinical scores + MRI-derived volumetrics + speech features**
to flag dementia risk, wrapped in a full multi-hospital clinical workflow (patient intake,
screening/follow-up visits, clinician review, diagnosis history). Built as a capstone comparing a
6-qubit Quantum SVM against a classical SVM baseline. **The classical SVM is the canonical
prediction** (see "Headline result" below for why). The QSVM runs alongside it as a labelled
*research view*: every screening screen has a Classical SVM · Quantum SVM · Compare switch that
animates the same patient between the two models' decision axes and shows both models'
cross-validated metrics.

## Headline result (real data, 2026-09-24) — and why the app serves the SVM, not the QSVM

The model is trained on a **real** dataset — OASIS-1/OASIS-2 clinical + MRI data, cross-matched by
`(label, sex, age)` with windowed features from a real speech corpus (131 speakers). It replaced an
earlier fully-synthetic dataset that turned out to be trivially separable (see
`PROJECT_CONTEXT.md` §6.1 / §28-29). 5-fold `GroupKFold` cross-validation (speaker/subject-grouped,
no noise injection needed):

| Modality | Classical SVM | QSVM (ZZFeatureMap) |
|---|---|---|
| Clinical only (4 feat) | 71.7% | 68.7% |
| MRI only (2 feat) | 77.7% | 78.1% |
| Speech only (18 feat) | 63.5% | 56.2% |
| **Fused (24 feat, 6 qubits)** | **81.1% ± 3.0%** | **78.1% ± 7.8%** |

Fused beats every single modality on both classifiers. The classical SVM also beats the QSVM on
every metric but specificity — accuracy, precision, F1, macro-F1, and especially ROC-AUC (90.8% vs
84.5%, the widest gap of any metric). **2026-09-25: the app was switched to serve the classical SVM
as the clinician-facing prediction** (previously the QSVM was canonical and the SVM was shown only
as a side-by-side comparison — see the note at the top of `backend/app/services/prediction.py`). The
quantum-shows-no-advantage finding is disclosed, not hidden — it's exactly why the app changed.
Full metrics (precision/recall/specificity/F1/ROC-AUC, sex-stratified rows, a sensitivity table, and
a second "clinical+MRI only, all ~385 OASIS subjects" evaluation) are in `results/metrics_full.json`,
produced by `metrics_full.ipynb`. Dataset provenance, the matching algorithm, and known limitations
are in `data/DATASHEET.md`.

## Architecture

```
Clinical form ──┐
MRI upload ─────┼─► per-modality StandardScaler + PCA(2) ─► concat (6-D) ─┬─► SVC   canonical: clinician queue + agreement flag
Speech upload ──┘                                                        └─► MinMaxScaler ─► ZZFeatureMap(6) ─► QSVC   research view (toggle)
```

The QSVM can never block the canonical result: if it fails or its pickle is missing, its columns
stay empty and the toggle disables itself.

### MRI: any reasonable scan

The MRI upload accepts **NIfTI** (`.nii`, `.nii.gz`), **MGH/MGZ**, **Analyze** `.hdr` + `.img`
pairs (as two files or zipped), **DICOM** (one file, a whole series as several files, or a
`.zip`; slices are ordered by position, extensions optional), and **2-D images** of a slice
(`.png`, `.jpg`, `.bmp`, `.tif`). Up to 50 MB.

Extractor v2 (`backend/app/services/mri_features.py`) doesn't need a perfect skull strip. It
reorients the scan, resamples it to 2 mm, and describes how much CSF-like signal sits at each
depth below the head surface. Calibrated regressors then map that description to nWBV and eTIV.
They were fit on 997 labelled OASIS scans of every input type (raw with skull, atlas-space,
skull-stripped) plus 3,090 2-D slices, validated with subject-grouped 5-fold CV:

| Input | nWBV error (MAE) | eTIV error (MAE) |
|---|---|---|
| Raw 3-D scan, skull on (OASIS-2 / OASIS-1) | 0.018 / 0.016 | 60 / 97 mL |
| Skull-stripped 3-D | 0.013 | not recoverable (atlas-normalised) |
| 2-D slice image | ≈0.026 | not measurable, training median used and flagged |
| Guessing the median | 0.039 | 130 mL |

Swapping the true MRI values for extracted ones keeps the SVM's label on 91% of raw scans. Every
upload records how it was read (input kind, format, expected error, warnings), and the clinician
sees that next to the scan. The extractor is calibrated for T1-weighted structural MRI.

### Design system

The frontend is plain CSS on a three-tier token system (`frontend/src/styles/`): primitives,
then semantic light/dark roles, then component tokens, with layers for base, components,
layout and screens. Light and dark themes (following the OS, or toggled in the nav) pass a
WCAG AA contrast check (`npm run check:contrast`). The living reference is the in-app
**/design-system** page.

- **Backend:** FastAPI + SQLAlchemy + Supabase (Postgres + Storage + Auth), `backend/`.
- **Frontend:** React + Vite, `frontend/`.
- **Model:** scikit-learn + Qiskit / `qiskit-machine-learning`, pickled pipelines in `results/`.
- **Dataset build:** `scripts/` (OASIS harmonisation, speech windowing, matching) — see
  `real_dataset_setup.md` (gitignored canonical spec) and `PROJECT_CONTEXT.md` §28-29.

## Setup

1. **Backend** — `cd backend && cp .env.example .env` and fill in your Supabase project's
   connection string, API key, and storage buckets (see the comments in `.env.example`). Then:
   ```
   docker compose up --build
   ```
   or, without Docker, `pip install -r backend/requirements.txt` and
   `uvicorn app.main:app --reload` from `backend/`. `GET /api/v1/health` should report
   `{"status":"ok","db":"ok","model":"ok","qsvm":"ok","mri_calibration":"ok"}`. Apply every
   file in `backend/migrations/` in order (they're idempotent) before first run.
2. **Frontend** — `cd frontend && npm install && npm run dev`.
3. **Retrain the model** (optional — pickles are already in `results/` and baked into the Docker
   image) — see `real_dataset_setup.md` Phase 1.8-1.9, or run the notebooks in order: `clinical_qsvm.ipynb` →
   `mri_qsvm.ipynb` → `speech_qsvm.ipynb` → `multimodal_qsvm.ipynb` → `metrics_full.ipynb`, then
   `python backend/scripts/train_classical_svm.py`.

## Testing a change

- `python -m pytest backend/tests tests` — backend unit tests (validation, MRI loaders for every
  format, the extractor on real OASIS scans when `data/` is present, the viewer conversion, the
  dual-model prediction seam) plus the dataset-matcher tests. No DB or network needed.
- `cd frontend && npm test` — validator and prediction-math unit tests; `npm run lint`;
  `npm run check:contrast` — WCAG contrast for every token pair, both themes.
- `python scripts/verify_real_dataset.py` — the Phase 1.7 automated dataset-quality gate (schema,
  leak probe, representativeness, age QC).
- `python backend/scripts/smoke_prediction.py` — model-serving seam, no DB required.
- `python backend/scripts/smoke_e2e.py` — full product flow against a running API + DB (needs
  `backend/.env` and `docker compose up`), including strict numeric validation, both models, and
  MRI uploads as NIfTI, PNG slice, Analyze pair and zipped DICOM series. Run it inside the
  container: `docker compose exec -e PYTHONPATH=/app -w /app backend python scripts/smoke_e2e.py`.
- After a retrain: `python backend/scripts/build_model_info.py` (refreshes the toggle's metrics
  and axis scales) and `python scripts/calibrate_mri_extractor.py` (refits the MRI calibration).

## Key documents

- `DEPLOY.md`: step-by-step Supabase + Render + Vercel deployment, with troubleshooting.
- `PROJECT_CONTEXT.md` — the chronological build record (append-only sections).
- `data/DATASHEET.md` — dataset provenance, collection process, known limitations, sign-off log.
- `real_dataset_setup.md` *(gitignored)* — the real-data build's canonical spec.
- `sample_data/README.md` — demo MRI/audio phantoms for driving the screening flow without real
  patient data.
