-- Cortex Health Portal — initial schema (TO_build_phase_2.md §3)
-- Idempotent: safe to run more than once. Apply either with
--   psql "$DATABASE_URL" -f backend/migrations/001_init.sql
-- or by pasting into the Supabase SQL editor.
--
-- Row-Level Security is intentionally NOT enabled (Product Rule 12 / §2): hospital
-- isolation is enforced at the application query layer via get_scoped_query(hospital_id).

-- gen_random_uuid() lives in pgcrypto; enabled by default on Supabase but ensure it.
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- --- Enum types (guarded so re-running doesn't error) ---
DO $$ BEGIN
    CREATE TYPE role AS ENUM ('receptionist', 'clinician', 'hospital_admin');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE consent_given_by AS ENUM ('patient', 'guardian');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE visit_type AS ENUM ('screening', 'follow_up');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE mri_status AS ENUM ('not_applicable', 'idle', 'uploading', 'processing', 'done');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE speech_status AS ENUM ('not_applicable', 'idle', 'recording', 'processing', 'done');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE model_prediction AS ENUM ('Nondemented', 'Demented');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE visit_status AS ENUM ('awaiting_uploads', 'pending_review', 'reviewed', 'completed');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE doctor_diagnosis AS ENUM ('Nondemented', 'Demented', 'Needs further evaluation');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE agreement_flag AS ENUM ('match', 'mismatch');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- --- Tables ---
CREATE TABLE IF NOT EXISTS hospitals (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name       text NOT NULL,
    address    text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS users (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    hospital_id   uuid NOT NULL REFERENCES hospitals(id),
    name          text NOT NULL,
    email         text NOT NULL UNIQUE,
    password_hash text NOT NULL,
    role          role NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_users_hospital_id ON users(hospital_id);

CREATE TABLE IF NOT EXISTS patients (
    id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    hospital_id          uuid NOT NULL REFERENCES hospitals(id),
    name                 text NOT NULL,
    dob                  date,
    sex                  text,
    phone                text,
    address              text,
    consent_given_by     consent_given_by NOT NULL,
    consent_relationship text,
    created_at           timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_patients_hospital_id ON patients(hospital_id);

CREATE TABLE IF NOT EXISTS visits (
    id                         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id                 uuid NOT NULL REFERENCES patients(id),
    hospital_id                uuid NOT NULL REFERENCES hospitals(id),
    visit_type                 visit_type NOT NULL,
    visit_date                 timestamptz NOT NULL DEFAULT now(),
    created_by_user_id         uuid NOT NULL REFERENCES users(id),
    -- clinical (both visit types); edu/ses are screening-only, hence nullable
    mmse                       double precision,
    cdr                        double precision,
    edu                        double precision,
    ses                        double precision,
    -- modality state (screening only)
    mri_status                 mri_status NOT NULL DEFAULT 'not_applicable',
    speech_status              speech_status NOT NULL DEFAULT 'not_applicable',
    mri_feature_vector         jsonb,
    speech_feature_vector      jsonb,
    -- model output (screening only, filled once all 3 modalities are done)
    model_prediction           model_prediction,
    model_confidence           double precision,
    -- doctor review (screening only)
    requires_review            boolean NOT NULL DEFAULT false,
    status                     visit_status NOT NULL,
    doctor_diagnosis           doctor_diagnosis,   -- always the LATEST saved diagnosis
    doctor_notes               text,
    diagnosis_saved_at         timestamptz,
    diagnosis_saved_by_user_id uuid REFERENCES users(id),
    agreement_flag             agreement_flag,     -- computed on diagnosis save
    created_at                 timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_visits_patient_id ON visits(patient_id);
CREATE INDEX IF NOT EXISTS idx_visits_hospital_status ON visits(hospital_id, status);

-- Append-only diagnosis audit trail (Product Rule 5, same-day revision).
-- visits.doctor_* mirror the most recent row for a visit.
CREATE TABLE IF NOT EXISTS diagnosis_history (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    visit_id         uuid NOT NULL REFERENCES visits(id),
    doctor_diagnosis doctor_diagnosis NOT NULL,
    doctor_notes     text,
    saved_by_user_id uuid NOT NULL REFERENCES users(id),
    saved_at         timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_diagnosis_history_visit_id ON diagnosis_history(visit_id);

CREATE TABLE IF NOT EXISTS audit_log (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    hospital_id uuid NOT NULL REFERENCES hospitals(id),
    user_id     uuid REFERENCES users(id),
    action      text NOT NULL,
    target_type text,
    target_id   text,
    timestamp   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_audit_log_hospital_id ON audit_log(hospital_id);
