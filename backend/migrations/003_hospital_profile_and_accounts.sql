-- Cortex Health Portal — hospital identity + admin-provisioned staff accounts.
-- Idempotent: safe to run more than once. Apply either with
--   psql "$DATABASE_URL" -f backend/migrations/003_hospital_profile_and_accounts.sql
-- or by pasting into the Supabase SQL editor.
--
-- Adds:
--   * hospitals.pincode  — the branch discriminator. Two branches of the same
--     hospital chain (e.g. several "Manipal Hospital" in one city) will not share
--     a 6-digit pincode, so (name, pincode) is what tells branches apart.
--   * hospitals.city     — display only, not part of the uniqueness key.
--   * hospitals.logo_url — public URL of the uploaded hospital logo.
--   * users.must_change_password — set when an admin provisions or resets an
--     account; the app forces a password change on the next sign-in and clears it.

ALTER TABLE hospitals ADD COLUMN IF NOT EXISTS pincode  text;
ALTER TABLE hospitals ADD COLUMN IF NOT EXISTS city     text;
ALTER TABLE hospitals ADD COLUMN IF NOT EXISTS logo_url text;

ALTER TABLE users ADD COLUMN IF NOT EXISTS must_change_password boolean NOT NULL DEFAULT false;

-- One hospital name per pincode. coalesce() collapses existing NULL-pincode rows
-- to '' so they still can't collide with each other. If this fails because the
-- database already holds same-name/same-pincode duplicates, dedupe them first —
-- the application layer (auth.register_hospital) enforces the same rule regardless.
CREATE UNIQUE INDEX IF NOT EXISTS uq_hospitals_name_pincode
    ON hospitals (lower(name), coalesce(pincode, ''));
