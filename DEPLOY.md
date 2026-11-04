# Deploying Cortex Health Portal

Three free-tier services, set up in this order: **Supabase** (database and file storage),
**Render** (the API, from `render.yaml`), **Vercel** (the frontend, from `frontend/vercel.json`).
Plan on about 30 minutes, most of it waiting for Render's first build.

Nothing secret lives in this repo. Every secret is pasted into a dashboard, never committed.

---

## 1. Supabase: database and storage

1. **Make sure the project is awake.** Free Supabase projects pause after a week without traffic.
   At [supabase.com/dashboard](https://supabase.com/dashboard), if the project says *Paused*,
   click **Restore project** and wait for it to come up. (This happened once already; a paused
   project shows up as `"db":"error"` with DNS failures, not as a password problem.)
2. **Schema.** Run every file in `backend/migrations/` in order (`001` … `006`) in
   **SQL Editor → New query → paste → Run**. They are idempotent, so re-running is harmless.
   *The dev project you already use has all six applied (006 on 2026-11-04). If you deploy against
   that same project, skip this step.*
3. **Storage buckets** (**Storage → New bucket**):
   - `patient-uploads`: **Private**, 50 MB file limit. Already exists in the dev project.
   - `public-assets`: **Public**, 2 MB limit, images only. **Not created yet.** Without it, hospital
     logo upload returns a clear error; everything else works.
4. **Collect three values** (keep this tab open):
   - **DATABASE_URL**: **Connect** (top bar) → **Connection string** → **Session pooler** → copy
     the URI and put your database password in it. It looks like
     `postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres`.
     **Use the Session pooler, not "Direct connection".** The direct host is IPv6-only and Render's
     egress is IPv4, so the direct URI fails with a network error that looks like a credentials
     problem.
   - **SUPABASE_URL**: **Project Settings → Data API** → Project URL, e.g.
     `https://<ref>.supabase.co`. Bare origin, no `/rest/v1`.
   - **SUPABASE_KEY**: **Project Settings → API Keys → Legacy API keys → `service_role`**. It must
     be the long JWT (`eyJ…`): the pinned `supabase==2.11.0` client rejects the newer
     `sb_secret_…` keys, and only `service_role` bypasses Storage's row-level security.

## 2. Render: the API

1. [dashboard.render.com](https://dashboard.render.com) → sign in with GitHub → **New → Blueprint**
   → pick **Prakshi16/demetia_qvsm** → branch **main**. Render reads `render.yaml` and proposes
   one service, **cortex-health-portal-api** (Docker, free plan, Singapore).
2. It asks for the five secrets marked `sync: false`:

   | Key | Value |
   |---|---|
   | `DATABASE_URL` | the Session pooler URI from step 1.4 |
   | `SUPABASE_URL` | the project URL from step 1.4 |
   | `SUPABASE_KEY` | the `service_role` JWT from step 1.4 |
   | `JWT_SECRET` | a new random string: run `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
   | `CORS_ORIGINS` | put `https://example.com` for now; you'll replace it in step 4 |

   `SUPABASE_BUCKET=patient-uploads` and `MODEL_DIR=/model` are already set by the blueprint.
3. **Apply**. The first build takes about 10 minutes, because the image is ~2.2 GB with
   scientific Python and Qiskit. Watch **Logs** for `Application startup complete`.
4. Open `https://<service>.onrender.com/api/v1/health`. You want:
   ```json
   {"status":"ok","db":"ok","model":"ok","qsvm":"ok","mri_calibration":"ok"}
   ```
   Copy the service URL; Vercel needs it next.

**Free-tier behaviour to know before a demo:** the service sleeps after 15 minutes idle, and the
first request after that takes about a minute while it wakes. Open the `/health` URL a couple of
minutes before presenting. Memory is 512 MB; measured peak for a large raw MRI upload with both
models loaded is ~350 MB, and uploads over 32 M voxels are refused with a clear message.

## 3. Vercel: the frontend

1. [vercel.com/new](https://vercel.com/new) → sign in with GitHub → **Import** `demetia_qvsm`.
2. **Root Directory → `frontend`** (important). The framework preset becomes Vite; build and
   output settings come from `frontend/vercel.json`, so leave them alone.
3. **Environment Variables** → add
   `VITE_API_BASE_URL` = `https://<service>.onrender.com/api/v1` (your Render URL plus `/api/v1`).
4. **Deploy**. Note the production URL, e.g. `https://demetia-qvsm.vercel.app`.

## 4. Connect them (CORS)

Back in Render → the service → **Environment** → set
`CORS_ORIGINS` = your exact Vercel URL, e.g. `https://demetia-qvsm.vercel.app`: `https`, no
trailing slash. To allow more than one origin, separate them with commas. **Save changes**:
Render redeploys (this time from cache, about 2 minutes).

## 5. Check it end to end

1. Open the Vercel URL → **Register a hospital** → sign in as the admin → add a receptionist and a
   clinician (each sets their own password on first sign-in).
2. As the receptionist: register a patient → **New visit → Screening** → clinical measures → upload
   an MRI (any format: NIfTI, MGZ, an `.hdr`+`.img` pair, a DICOM series or `.zip`, or a PNG of a
   slice) and a recording → the model output appears, with the **Classical SVM · Quantum SVM ·
   Compare** switch.
3. As the clinician: the patient is in **Pending review** → open the visit → view the scan, play
   the recording, switch models, save a diagnosis.
4. Optional, scripted: the full 74-check suite can target the deployed API from any machine with
   the backend dependencies and the production `DATABASE_URL` (it cleans up after itself):
   `SMOKE_BASE_URL=https://<service>.onrender.com/api/v1 python backend/scripts/smoke_e2e.py`
   (run from `backend/` with `PYTHONPATH=.`).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `/health` → `"db":"error"` | Paused Supabase project (restore it), or the Direct-connection URI instead of the Session pooler, or a wrong password in the URI |
| Browser console: *blocked by CORS policy* | `CORS_ORIGINS` on Render doesn't exactly match the Vercel URL (scheme, no trailing slash), or Render hasn't redeployed since you changed it |
| Every request 404s from the frontend | `VITE_API_BASE_URL` missing `/api/v1`, or set after the build. Change it, then **Redeploy** on Vercel (env vars are baked in at build time) |
| Everyone is signed out after a deploy | `JWT_SECRET` changed. Expected; sign in again |
| Upload fails with a storage error (502) | `SUPABASE_KEY` isn't the legacy `service_role` JWT, or the bucket isn't named `patient-uploads` |
| Logo upload says the bucket is missing | Create the public `public-assets` bucket (step 1.3) |
| `"model":"error"` | Built with the wrong Docker context. `render.yaml` must keep `dockerContext: .` (repo root) |
| Render log shows the instance restarted mid-upload | Out of memory. Check the upload wasn't an unusually large multi-volume file; the 32 M-voxel cap should prevent this |
