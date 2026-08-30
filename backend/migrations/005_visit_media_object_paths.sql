-- Cortex Health Portal — remember where each visit's MRI / speech upload lives.
-- Idempotent: safe to run more than once. Apply either with
--   psql "$DATABASE_URL" -f backend/migrations/005_visit_media_object_paths.sql
-- or by pasting into the Supabase SQL editor.
--
-- Adds:
--   * visits.mri_object_path    — the object key in the private SUPABASE_BUCKET
--   * visits.speech_object_path — for the uploaded scan / recording, e.g.
--     "mri/<visit_id>/<filename>". Written by the upload endpoints
--     (backend/app/routers/mri_upload.py, speech.py) so the clinician's file
--     endpoint (GET /visits/{id}/file/{kind}) can mint a signed URL without
--     guessing the filename. Nullable: rows uploaded before this migration fall
--     back to a storage.list() lookup on the mri/<visit_id> folder.

ALTER TABLE visits ADD COLUMN IF NOT EXISTS mri_object_path    text;
ALTER TABLE visits ADD COLUMN IF NOT EXISTS speech_object_path text;
