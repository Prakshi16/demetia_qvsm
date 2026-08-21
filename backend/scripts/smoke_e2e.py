"""End-to-end smoke test — the §10 sequence, against a running API.

Walks the whole product in the order the spec lays it out: register a hospital,
staff it with a clinician and a receptionist, register a patient with consent,
screen them, watch the model fire, have the doctor disagree, then follow up.
Along the way it exercises the §4 visit-type rules and the Product Rules that
are easy to regress (2A resumability, 5 same-day revision, 7 consent, 11 the
upload allowlist, 12 hospital scoping).

Everything it creates is named 'ZZ Smoke %' and is deleted on the way out, so it
is safe to run against the shared dev database. Uses synthetic NIfTI and WAV
data generated in-process — no fixture files, no real patient data.

Run it from inside the backend container, which already has every dependency:

    docker compose up -d
    docker cp backend/scripts/smoke_e2e.py demetia_qvsm-backend-1:/app/
    docker exec -e PYTHONPATH=/app -w /app demetia_qvsm-backend-1 python smoke_e2e.py

Exits non-zero if any check fails.
"""
from __future__ import annotations

import gzip
import io
import os
import sys
import uuid
import wave
from datetime import date, timedelta

import httpx
import nibabel as nib
import numpy as np
from sqlalchemy import text

from app.db import SessionLocal

BASE = os.environ.get("SMOKE_BASE_URL", "http://localhost:8000/api/v1")
TAG = uuid.uuid4().hex[:8]
PREFIX = "ZZ Smoke"

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# Synthetic inputs. Dummy data only, per the project's standing rule.
# ---------------------------------------------------------------------------

def make_nifti() -> bytes:
    """A brain-ish volume: a bright core inside a dimmer intracranial shell.

    The extractor is an intensity/geometry proxy, so this gives it the contrast
    it looks for. The resulting eTIV is far outside the OASIS training range —
    that is the synthetic phantom, not a bug.
    """
    rng = np.random.default_rng(7)
    grid = np.mgrid[:48, :48, :48].astype(np.float32)
    centre = np.array([24.0, 24.0, 24.0]).reshape(3, 1, 1, 1)
    radius = np.sqrt(((grid - centre) ** 2).sum(axis=0))

    volume = np.zeros((48, 48, 48), dtype=np.float32)
    volume[radius < 20] = 0.45
    volume[radius < 16] = 0.85
    volume = np.clip(volume + rng.normal(0, 0.02, volume.shape).astype(np.float32), 0, 1)

    image = nib.Nifti1Image(volume, affine=np.diag([1.5, 1.5, 1.5, 1.0]))
    buffer = io.BytesIO()
    file_map = image.make_file_map()
    file_map["image"].fileobj = gzip.GzipFile(fileobj=buffer, mode="wb")
    image.to_file_map(file_map)
    file_map["image"].fileobj.close()
    return buffer.getvalue()


def make_wav() -> bytes:
    """Voiced bursts separated by silence, so pause rate and speech rate have
    something real to measure rather than one flat tone."""
    sample_rate = 16000
    rng = np.random.default_rng(3)
    segments = []
    for index in range(6):
        t = np.linspace(0, 0.45, int(sample_rate * 0.45), endpoint=False)
        f0 = 120 + 25 * np.sin(2 * np.pi * 0.8 * t) + index * 6
        voiced = 0.35 * np.sin(2 * np.pi * f0 * t) + 0.12 * np.sin(2 * np.pi * 2 * f0 * t)
        segments.append(voiced * np.hanning(len(t)) + rng.normal(0, 0.004, len(t)))
        segments.append(np.zeros(int(sample_rate * 0.18)))

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(
            (np.clip(np.concatenate(segments), -1, 1) * 24000).astype(np.int16).tobytes()
        )
    return buffer.getvalue()


def backdate_visit(visit_id: str, days: int) -> None:
    """Move a visit into the past.

    §10 asks to confirm that a patient screened over a year ago is offered a
    screening again rather than a follow-up. There is no API for that — the
    interval is measured against visit_date — so the only honest way to test the
    rule is to age the row directly.
    """
    with SessionLocal() as db:
        db.execute(
            text("UPDATE visits SET visit_date = :d WHERE id = :i"),
            {"d": date.today() - timedelta(days=days), "i": visit_id},
        )
        db.commit()


def cleanup() -> None:
    """Delete every row this run created, children first."""
    with SessionLocal() as db:
        db.execute(text("""
            DELETE FROM diagnosis_history WHERE visit_id IN (
                SELECT v.id FROM visits v JOIN patients p ON p.id = v.patient_id
                 WHERE p.hospital_id IN (SELECT id FROM hospitals WHERE name LIKE :p))
        """), {"p": f"{PREFIX}%"})
        db.execute(text("""
            DELETE FROM audit_log WHERE hospital_id IN
                (SELECT id FROM hospitals WHERE name LIKE :p)
        """), {"p": f"{PREFIX}%"})
        db.execute(text("""
            DELETE FROM visits WHERE patient_id IN (
                SELECT id FROM patients WHERE hospital_id IN
                    (SELECT id FROM hospitals WHERE name LIKE :p))
        """), {"p": f"{PREFIX}%"})
        for table in ("patients", "users"):
            db.execute(
                text(f"DELETE FROM {table} WHERE hospital_id IN "
                     "(SELECT id FROM hospitals WHERE name LIKE :p)"),
                {"p": f"{PREFIX}%"},
            )
        db.execute(text("DELETE FROM hospitals WHERE name LIKE :p"), {"p": f"{PREFIX}%"})
        db.commit()

        remaining = db.execute(
            text("SELECT count(*) FROM hospitals WHERE name LIKE :p"), {"p": f"{PREFIX}%"}
        ).scalar()
        print(f"\ncleanup: {remaining} smoke hospital(s) remaining")


# ---------------------------------------------------------------------------

def run(client: httpx.Client) -> None:
    nifti, wav = make_nifti(), make_wav()

    # -- health ------------------------------------------------------------
    health = client.get(f"{BASE}/health").json()
    check("health: db and model both ok",
          health.get("db") == "ok" and health.get("model") == "ok", str(health))

    # -- register hospital -> admin, then a clinician and a receptionist ----
    response = client.post(f"{BASE}/auth/register-hospital", json={
        "hospital_name": f"{PREFIX} E2E {TAG}", "address": "temp",
        "admin_name": "Admin", "admin_email": f"admin.{TAG}@smoke.test",
        "password": "smoketest123"})
    response.raise_for_status()
    hospital_id = response.json()["user"]["hospital_id"]

    staff = {}
    for role in ("clinician", "receptionist"):
        response = client.post(f"{BASE}/auth/register-staff", json={
            "hospital_id": hospital_id, "name": role.title(),
            "email": f"{role}.{TAG}@smoke.test", "password": "smoketest123",
            "role": role})
        response.raise_for_status()
        staff[role] = {"Authorization": f"Bearer {response.json()['token']}"}
    check("hospital + clinician + receptionist registered", len(staff) == 2)

    doctor, desk = staff["clinician"], staff["receptionist"]

    # -- Rule 7: consent ---------------------------------------------------
    response = client.post(f"{BASE}/patients", headers=desk, json={
        "name": f"{PREFIX} NoRelationship {TAG}", "consent_given_by": "guardian"})
    check("guardian consent without a relationship -> 422",
          response.status_code == 422, str(response.status_code))

    response = client.post(f"{BASE}/patients", headers=desk, json={
        "name": f"{PREFIX} Patient {TAG}", "dob": "1946-04-02", "sex": "Female",
        "phone": "5550111", "address": "temp", "consent_given_by": "patient"})
    response.raise_for_status()
    patient_id = response.json()["id"]
    check("patient registered with consent", True)

    # -- §4: a patient with no history gets a screening --------------------
    decision = client.get(f"{BASE}/patients/{patient_id}/next-visit-type",
                          headers=desk).json()
    check("no history -> screening", decision["visit_type"] == "screening",
          decision["reason"])

    response = client.post(f"{BASE}/visits", headers=desk, json={
        "patient_id": patient_id, "visit_type": "follow_up", "mmse": 26, "cdr": 0.5})
    check("follow-up before any screening -> 400", response.status_code == 400,
          str(response.status_code))

    # -- screening visit ---------------------------------------------------
    response = client.post(f"{BASE}/visits", headers=desk, json={
        "patient_id": patient_id, "visit_type": "screening",
        "mmse": 24, "cdr": 0.5, "edu": 12, "ses": 3})
    response.raise_for_status()
    visit = response.json()
    visit_id = visit["id"]
    check("screening visit opens as awaiting_uploads",
          visit["status"] == "awaiting_uploads", visit["status"])

    # Rule 11: the allowlist rejects before anything is read.
    response = client.post(f"{BASE}/visits/{visit_id}/mri-upload", headers=desk,
                           files={"file": ("notes.txt", b"not a scan", "text/plain")})
    check("Rule 11: .txt to the MRI endpoint -> 400", response.status_code == 400,
          str(response.status_code))

    response = client.post(f"{BASE}/visits/{visit_id}/mri-upload",
                           files={"file": ("s.nii.gz", nifti, "application/gzip")})
    check("no token -> 401", response.status_code == 401, str(response.status_code))

    response = client.post(f"{BASE}/visits/{uuid.uuid4()}/mri-upload", headers=desk,
                           files={"file": ("s.nii.gz", nifti, "application/gzip")})
    check("Rule 12: unknown visit -> 404, never 403", response.status_code == 404,
          str(response.status_code))

    # -- MRI first: the model must NOT fire on one modality ----------------
    response = client.post(f"{BASE}/visits/{visit_id}/mri-upload", headers=desk,
                           files={"file": ("s.nii.gz", nifti, "application/gzip")})
    response.raise_for_status()
    after_mri = response.json()
    check("MRI persists a 4-feature vector",
          len(after_mri.get("mri_feature_vector") or []) == 4,
          str(after_mri.get("mri_feature_vector")))
    check("one modality is not enough: still awaiting_uploads, no prediction",
          after_mri["status"] == "awaiting_uploads"
          and after_mri["model_prediction"] is None,
          after_mri["status"])

    # Rule 2A: a half-finished screening is a real, resumable state.
    resumed = client.get(f"{BASE}/visits/{visit_id}", headers=desk).json()
    check("Rule 2A: half-finished screening is resumable",
          resumed["status"] == "awaiting_uploads" and resumed["mri_status"] == "done"
          and resumed["speech_status"] != "done", resumed["speech_status"])

    # -- speech closes the seam and the model runs -------------------------
    response = client.post(f"{BASE}/visits/{visit_id}/speech-upload", headers=desk,
                           files={"file": ("s.wav", wav, "audio/wav")})
    response.raise_for_status()
    after_speech = response.json()
    model_says = after_speech["model_prediction"]
    check("speech persists an 18-feature vector",
          len(after_speech.get("speech_feature_vector") or []) == 18)
    check("both modalities in -> prediction appears, status pending_review",
          after_speech["status"] == "pending_review" and model_says is not None,
          f"{model_says} margin={after_speech['model_confidence']}")
    check("classical SVM comparison present too",
          after_speech["svm_prediction"] is not None,
          f"{after_speech['svm_prediction']} margin={after_speech['svm_confidence']}")

    persisted = client.get(f"{BASE}/visits/{visit_id}", headers=desk).json()
    check("prediction is persisted, not just echoed",
          persisted["model_prediction"] == model_says)

    queue = client.get(f"{BASE}/patients/pending-review", headers=doctor).json()
    check("patient reaches the clinician's pending-review queue",
          any(p["id"] == patient_id for p in queue), f"{len(queue)} in queue")

    decision = client.get(f"{BASE}/patients/{patient_id}/next-visit-type",
                          headers=desk).json()
    check("still screening while the review is outstanding",
          decision["visit_type"] == "screening", decision["reason"])

    # -- the mismatch path, tested explicitly per §10 ----------------------
    disagrees = "Demented" if model_says == "Nondemented" else "Nondemented"
    response = client.post(f"{BASE}/visits/{visit_id}/diagnosis", headers=doctor, json={
        "doctor_diagnosis": disagrees, "doctor_notes": "smoke test"})
    response.raise_for_status()
    reviewed = response.json()
    check("diagnosis saved -> reviewed", reviewed["status"] == "reviewed",
          reviewed["status"])
    check("§10 mismatch path: agreement_flag = mismatch",
          reviewed["agreement_flag"] == "mismatch",
          f"model={model_says} doctor={disagrees}")

    # Rule 5: same UTC day, the diagnosis is still editable.
    response = client.post(f"{BASE}/visits/{visit_id}/diagnosis", headers=doctor, json={
        "doctor_diagnosis": model_says, "doctor_notes": "revised same day"})
    check("Rule 5: same-day revision accepted", response.status_code == 200,
          str(response.status_code))
    if response.status_code == 200:
        check("revision clears the mismatch",
              response.json()["agreement_flag"] != "mismatch",
              str(response.json()["agreement_flag"]))

    # -- follow-up ---------------------------------------------------------
    decision = client.get(f"{BASE}/patients/{patient_id}/next-visit-type",
                          headers=desk).json()
    check("after a confirmed diagnosis -> follow_up",
          decision["visit_type"] == "follow_up", decision["reason"])

    response = client.post(f"{BASE}/visits", headers=desk, json={
        "patient_id": patient_id, "visit_type": "follow_up", "mmse": 22, "cdr": 1})
    response.raise_for_status()
    follow_up = response.json()
    check("follow-up saves as completed, no review step",
          follow_up["status"] == "completed" and follow_up["requires_review"] is False,
          follow_up["status"])
    check("follow-up runs no model and has no modalities",
          follow_up["model_prediction"] is None
          and follow_up["mri_status"] == "not_applicable"
          and follow_up["speech_status"] == "not_applicable")

    profile = client.get(f"{BASE}/patients/{patient_id}", headers=desk).json()
    check("trend chart gains a second point", len(profile["trend"]) == 2,
          str([(p["mmse"], p["cdr"]) for p in profile["trend"]]))

    # -- the re-screen interval, by ageing the screening visit -------------
    backdate_visit(visit_id, days=400)
    decision = client.get(f"{BASE}/patients/{patient_id}/next-visit-type",
                          headers=desk).json()
    check("screened >12 months ago -> screening again, not follow_up",
          decision["visit_type"] == "screening", decision["reason"])

    # -- the override, available regardless of what §4 decided -------------
    backdate_visit(visit_id, days=1)
    decision = client.get(f"{BASE}/patients/{patient_id}/next-visit-type"
                          "?force_screening=true", headers=desk).json()
    check("force_screening override always returns screening",
          decision["visit_type"] == "screening", decision["reason"])

    # -- Rule 12: a second hospital cannot see the first one's patient -----
    response = client.post(f"{BASE}/auth/register-hospital", json={
        "hospital_name": f"{PREFIX} Other {TAG}", "address": "temp",
        "admin_name": "Other", "admin_email": f"other.{TAG}@smoke.test",
        "password": "smoketest123"})
    response.raise_for_status()
    outsider = {"Authorization": f"Bearer {response.json()['token']}"}

    response = client.get(f"{BASE}/patients/{patient_id}", headers=outsider)
    check("Rule 12: another hospital's patient -> 404, never 403",
          response.status_code == 404, str(response.status_code))
    response = client.get(f"{BASE}/visits/{visit_id}", headers=outsider)
    check("Rule 12: another hospital's visit -> 404, never 403",
          response.status_code == 404, str(response.status_code))


def main() -> int:
    client = httpx.Client(timeout=180.0)
    try:
        run(client)
    except Exception as error:  # noqa: BLE001 - a crash is a failed run, and cleanup still has to happen
        check("run completed without an unhandled error", False, repr(error)[:200])
    finally:
        cleanup()

    failed = [name for name, ok, _ in results if not ok]
    print("=" * 66)
    print(f"{len(results) - len(failed)}/{len(results)} passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
