# Cortex Health Portal — Multimodal QSVM Dementia Detection

A hospital screening app that fuses **clinical scores + MRI-derived volumetrics + speech features**
into a 6-qubit Quantum SVM (with a classical SVM shown alongside for comparison) to flag dementia
risk, wrapped in a full multi-hospital clinical workflow (patient intake, screening/follow-up visits,
clinician review, diagnosis history).

## Headline result (real data, 2026-09-24)

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

Fused beats every single modality on both classifiers, and the quantum kernel shows no consistent
advantage over the classical RBF kernel — an honest, disclosed finding, not a bug. Full metrics
(precision/recall/specificity/F1/ROC-AUC, sex-stratified rows, a sensitivity table, and a second
"clinical+MRI only, all ~385 OASIS subjects" evaluation) are in `results/metrics_full.json`, produced
by `metrics_full.ipynb`. Dataset provenance, the matching algorithm, and known limitations are in
`data/DATASHEET.md`.

## Architecture

```
Clinical form ──┐
MRI upload ─────┼─► per-modality StandardScaler + PCA(2) ─► concat (6-D) ─► MinMaxScaler
Speech upload ──┘                                                              │
                                                                                ▼
                                                          ZZFeatureMap(6) ─► QSVC  (quantum)
                                                          same 6-D          ─► SVC  (classical, display-only)
```

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
   `{"status":"ok","db":"ok","model":"ok"}`.
2. **Frontend** — `cd frontend && npm install && npm run dev`.
3. **Retrain the model** (optional — pickles are already in `results/` and baked into the Docker
   image) — see `real_dataset_setup.md` Phase 1.8-1.9, or run the notebooks in order: `clinical_qsvm.ipynb` →
   `mri_qsvm.ipynb` → `speech_qsvm.ipynb` → `multimodal_qsvm.ipynb` → `metrics_full.ipynb`, then
   `python backend/scripts/train_classical_svm.py`.

## Smoke-testing a change

- `pytest tests/test_build_real_dataset.py` — dataset-matcher unit tests.
- `python scripts/verify_real_dataset.py` — the Phase 1.7 automated dataset-quality gate (schema,
  leak probe, representativeness, age QC).
- `python backend/scripts/smoke_prediction.py` — model-serving seam, no DB required.
- `python backend/scripts/smoke_e2e.py` — full product flow against a running API + DB (needs
  `backend/.env` and `docker compose up`).

## Key documents

- `PROJECT_CONTEXT.md` — the chronological build record (append-only sections).
- `data/DATASHEET.md` — dataset provenance, collection process, known limitations, sign-off log.
- `real_dataset_setup.md` *(gitignored)* — the real-data build's canonical spec.
- `sample_data/README.md` — demo MRI/audio phantoms for driving the screening flow without real
  patient data.
