-- Cortex Health Portal — bring the QSVM back as a research view, and record how
-- each MRI was read.
-- Idempotent: safe to run more than once. Apply either with
--   psql "$DATABASE_URL" -f backend/migrations/006_qsvm_research_view_and_mri_meta.sql
-- or by pasting into the Supabase SQL editor.
--
-- Adds:
--   * visits.qsvm_prediction / visits.qsvm_confidence — the quantum-kernel SVM's
--     output for the same 24-feature row, computed alongside the classical SVM so
--     the visit screens can toggle between the two for academic comparison. The
--     classical SVM stays canonical: model_prediction / model_confidence are its
--     columns and agreement_flag still compares the doctor against it. The QSVM
--     feeds no computed field.
--   * visits.mri_extraction — JSON metadata from the MRI extractor: input kind
--     (3-D volume vs 2-D image), source format, method, expected error, and any
--     warnings (e.g. "eTIV can't be measured from one slice"). Shown to the
--     clinician next to the scan.
--
-- The legacy svm_prediction / svm_confidence columns (from 002, written only
-- before 2026-09-25) are left untouched.

ALTER TABLE visits ADD COLUMN IF NOT EXISTS qsvm_prediction model_prediction;
ALTER TABLE visits ADD COLUMN IF NOT EXISTS qsvm_confidence double precision;
ALTER TABLE visits ADD COLUMN IF NOT EXISTS mri_extraction  jsonb;
