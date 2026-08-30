-- Cortex Health Portal — one email domain per hospital.
-- Idempotent: safe to run more than once. Apply either with
--   psql "$DATABASE_URL" -f backend/migrations/004_hospital_email_domain.sql
-- or by pasting into the Supabase SQL editor.
--
-- Adds:
--   * hospitals.email_domain — the single domain every account at the hospital
--     uses. Set once at register-hospital; the app never edits it. Login
--     addresses (admin's and staff's) are derived from the person's name plus
--     this domain (backend/app/services/emails.py), never typed.
--
-- Also removes the two placeholder "xyz" dev hospitals left over from migration
-- 003 (they were given synthetic pincodes 100001 / 100002 so the unique index
-- could build). They predate the email-domain rule and hold no real data.

ALTER TABLE hospitals ADD COLUMN IF NOT EXISTS email_domain text;

-- DELETEs are naturally idempotent — nothing matches on a second run.
DELETE FROM audit_log
 WHERE hospital_id IN (SELECT id FROM hospitals WHERE lower(name) = 'xyz');
DELETE FROM diagnosis_history
 WHERE visit_id IN (
     SELECT v.id FROM visits v
       JOIN patients p ON p.id = v.patient_id
      WHERE p.hospital_id IN (SELECT id FROM hospitals WHERE lower(name) = 'xyz'));
DELETE FROM visits
 WHERE hospital_id IN (SELECT id FROM hospitals WHERE lower(name) = 'xyz');
DELETE FROM patients
 WHERE hospital_id IN (SELECT id FROM hospitals WHERE lower(name) = 'xyz');
DELETE FROM users
 WHERE hospital_id IN (SELECT id FROM hospitals WHERE lower(name) = 'xyz');
DELETE FROM hospitals WHERE lower(name) = 'xyz';
