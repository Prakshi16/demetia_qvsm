-- Cortex Health Portal — add classical-SVM comparison columns to visits.
-- Idempotent: safe to run more than once. Apply either with
--   psql "$DATABASE_URL" -f backend/migrations/002_add_svm_columns.sql
-- or by pasting into the Supabase SQL editor.
--
-- The app shows two model outputs on the visit detail page for research/comparison
-- (capstone demo, not a clinical tool): the QSVM result on the existing
-- model_prediction/model_confidence columns ("Quantum SVM"), and a classical SVM
-- result on the columns added here ("Classical SVM"). The classical SVM feeds no
-- computed field — agreement_flag stays tied to the QSVM prediction only.
-- Reuses the existing `model_prediction` enum type (Nondemented | Demented).

ALTER TABLE visits ADD COLUMN IF NOT EXISTS svm_prediction model_prediction;
ALTER TABLE visits ADD COLUMN IF NOT EXISTS svm_confidence double precision;
