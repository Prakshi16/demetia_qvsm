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
import json
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

def make_nifti(inner_radius: float = 16.0) -> bytes:
    """A brain-ish volume: a bright core inside a dimmer intracranial shell.

    The extractor is an intensity/geometry proxy, so this gives it the contrast
    it looks for. The resulting eTIV is far outside the OASIS training range —
    that is the synthetic phantom, not a bug.

    inner_radius shrinks/grows the bright core (default 16, same as always) —
    used by the re-upload check below to produce a visibly different feature
    vector from the default call, so "the vector actually changed" is a real
    assertion, not just "the request succeeded."
    """
    rng = np.random.default_rng(7)
    grid = np.mgrid[:48, :48, :48].astype(np.float32)
    centre = np.array([24.0, 24.0, 24.0]).reshape(3, 1, 1, 1)
    radius = np.sqrt(((grid - centre) ** 2).sum(axis=0))

    volume = np.zeros((48, 48, 48), dtype=np.float32)
    volume[radius < 20] = 0.45
    volume[radius < inner_radius] = 0.85
    volume = np.clip(volume + rng.normal(0, 0.02, volume.shape).astype(np.float32), 0, 1)

    image = nib.Nifti1Image(volume, affine=np.diag([1.5, 1.5, 1.5, 1.0]))
    buffer = io.BytesIO()
    file_map = image.make_file_map()
    file_map["image"].fileobj = gzip.GzipFile(fileobj=buffer, mode="wb")
    image.to_file_map(file_map)
    file_map["image"].fileobj.close()
    return buffer.getvalue()


def _head_phantom(shape=(64, 76, 64)) -> np.ndarray:
    """Layered T1-like head (scalp, skull, CSF, grey, white, ventricles)."""
    rng = np.random.default_rng(11)
    z, y, x = np.indices(shape, dtype=np.float32)
    c = (np.array(shape) - 1) / 2
    rad = np.array(shape) * 0.42
    r = np.sqrt(((z - c[0]) / rad[0]) ** 2 + ((y - c[1]) / rad[1]) ** 2 + ((x - c[2]) / rad[2]) ** 2)
    vol = np.zeros(shape, np.float32)
    vol[r <= 1.0] = 0.8
    vol[r <= 0.95] = 0.08
    vol[r <= 0.86] = 0.15
    vol[r <= 0.8] = 0.45
    vol[r <= 0.68] = 0.7
    vol += rng.normal(0, 0.03, shape).astype(np.float32)
    return (np.clip(vol, 0, 1) * 1000).astype(np.int16)


def make_png_slice() -> bytes:
    from PIL import Image

    sl = _head_phantom()[:, :, 32].astype(np.float32)
    buffer = io.BytesIO()
    Image.fromarray((sl / sl.max() * 255).astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def make_analyze_pair() -> tuple[bytes, bytes]:
    image = nib.AnalyzeImage(_head_phantom(), np.diag([2.0, 2.0, 2.0, 1.0]))
    header, data = io.BytesIO(), io.BytesIO()
    image.to_file_map({"header": nib.FileHolder(fileobj=header), "image": nib.FileHolder(fileobj=data)})
    return header.getvalue(), data.getvalue()


def make_dicom_series_zip() -> bytes:
    import zipfile

    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

    vol = _head_phantom((48, 56, 40)).astype(np.uint16)
    series = generate_uid()
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        for z in range(vol.shape[2]):
            meta = FileMetaDataset()
            meta.MediaStorageSOPClassUID = MRImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds = Dataset()
            ds.file_meta = meta
            ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, meta.MediaStorageSOPInstanceUID
            ds.SeriesInstanceUID, ds.Modality, ds.InstanceNumber = series, "MR", z + 1
            pixels = vol[:, :, z].T
            ds.Rows, ds.Columns = pixels.shape
            ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
            ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
            ds.PixelSpacing, ds.SliceThickness = [2.5, 2.5], 2.5
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            ds.ImagePositionPatient = [0.0, 0.0, z * 2.5]
            ds.PixelData = pixels.tobytes()
            buffer = io.BytesIO()
            ds.save_as(buffer, enforce_file_format=True)
            archive.writestr(f"DICOM/IM{z:04d}", buffer.getvalue())
    return out.getvalue()


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

    # -- register hospital -> admin, then admin provisions the staff ---------
    # The domain is chosen here; the admin's own login address is derived from
    # the name + domain (never sent), as every staff address then is.
    domain = f"{TAG}.smoke.test"
    response = client.post(f"{BASE}/auth/register-hospital", json={
        "hospital_name": f"{PREFIX} E2E {TAG}", "address": "temp",
        "pincode": "560017", "city": "Bengaluru",
        "admin_name": "Anita Rao", "email_domain": domain,
        "password": "smoketest123"})
    response.raise_for_status()
    reg = response.json()
    admin = {"Authorization": f"Bearer {reg['token']}"}
    check("admin login address is name-derived on the chosen domain",
          reg["user"]["email"] == f"anita.rao@{domain}", reg["user"]["email"])

    hospital_now = client.get(f"{BASE}/hospital", headers=admin).json()
    check("hospital carries its email domain",
          hospital_now.get("email_domain") == domain, str(hospital_now.get("email_domain")))

    # fault #7: same name + same pincode is a duplicate; same name + a different
    # pincode is a branch and is allowed.
    dupe = client.post(f"{BASE}/auth/register-hospital", json={
        "hospital_name": f"{PREFIX} E2E {TAG}", "address": "temp",
        "pincode": "560017", "admin_name": "Bala Nair",
        "email_domain": domain, "password": "smoketest123"})
    check("fault #7: same hospital name + same pincode -> 409",
          dupe.status_code == 409, str(dupe.status_code))
    branch = client.post(f"{BASE}/auth/register-hospital", json={
        "hospital_name": f"{PREFIX} E2E {TAG}", "address": "temp",
        "pincode": "560066", "admin_name": "Chandra Bose",
        "email_domain": domain, "password": "smoketest123"})
    check("fault #7: same name + different pincode -> branch allowed",
          branch.status_code == 200, str(branch.status_code))

    # fault #6: a digits-only name is rejected.
    bad_name = client.post(f"{BASE}/auth/register-hospital", json={
        "hospital_name": "123456", "pincode": "560099", "admin_name": "X Y",
        "email_domain": domain, "password": "smoketest123"})
    check("fault #6: digits-only hospital name -> 422",
          bad_name.status_code == 422, str(bad_name.status_code))

    # a malformed email domain is rejected.
    bad_domain = client.post(f"{BASE}/auth/register-hospital", json={
        "hospital_name": f"{PREFIX} BadDomain {TAG}", "pincode": "560098",
        "admin_name": "Dee Dee", "email_domain": "not a domain",
        "password": "smoketest123"})
    check("malformed email domain -> 422", bad_domain.status_code == 422,
          str(bad_domain.status_code))

    staff = {}
    for role in ("clinician", "receptionist"):
        created = client.post(f"{BASE}/staff", headers=admin, json={
            "name": role.title(), "role": role,
            "temporary_password": "temp-pass-123"})
        created.raise_for_status()
        email = created.json()["email"]
        check(f"{role} email derived from the name on the hospital domain",
              email == f"{role}@{domain}", email)
        check(f"provisioned {role} must set a password on first sign-in",
              created.json()["must_change_password"] is True)

        # first sign-in with the temp password, then the forced change
        login_resp = client.post(f"{BASE}/auth/login", json={
            "email": email, "password": "temp-pass-123"})
        login_resp.raise_for_status()
        check(f"{role} login token flags must_change_password",
              login_resp.json()["user"]["must_change_password"] is True)
        temp_headers = {"Authorization": f"Bearer {login_resp.json()['token']}"}

        changed = client.post(f"{BASE}/auth/change-password", headers=temp_headers, json={
            "current_password": "temp-pass-123", "new_password": "smoketest123"})
        changed.raise_for_status()
        check(f"{role} change-password clears the flag",
              changed.json()["user"]["must_change_password"] is False)
        staff[role] = {"Authorization": f"Bearer {changed.json()['token']}"}
    check("hospital + clinician + receptionist provisioned", len(staff) == 2)

    # two staff with the same name get progressively shortened addresses.
    dup1 = client.post(f"{BASE}/staff", headers=admin, json={
        "name": "Meera Iyer", "role": "clinician",
        "temporary_password": "temp-pass-123"}).json()
    dup2 = client.post(f"{BASE}/staff", headers=admin, json={
        "name": "Meera Iyer", "role": "receptionist",
        "temporary_password": "temp-pass-123"}).json()
    check("same-name staff get distinct, shortened derived emails",
          dup1["email"] == f"meera.iyer@{domain}"
          and dup2["email"] == f"meera.i@{domain}",
          f"{dup1['email']} / {dup2['email']}")

    doctor, desk = staff["clinician"], staff["receptionist"]

    # -- receptionist-only: patient intake and starting visits belong to the
    #    front desk; a clinician or admin calling them gets 403 --------------
    for label, hdr in (("clinician", doctor), ("admin", admin)):
        r = client.post(f"{BASE}/patients", headers=hdr, json={
            "name": f"{PREFIX} Nope {TAG}", "consent_given_by": "patient"})
        check(f"{label} cannot register a patient -> 403",
              r.status_code == 403, str(r.status_code))
        r = client.post(f"{BASE}/visits", headers=hdr, json={
            "patient_id": str(uuid.uuid4()), "visit_type": "screening",
            "mmse": 24, "cdr": 0.5, "edu": 12, "ses": 3})
        check(f"{label} cannot open a visit -> 403",
              r.status_code == 403, str(r.status_code))

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

    r = client.get(f"{BASE}/patients/{patient_id}/next-visit-type", headers=doctor)
    check("clinician cannot query next-visit-type -> 403",
          r.status_code == 403, str(r.status_code))

    # -- §4: a patient with no history gets a screening --------------------
    decision = client.get(f"{BASE}/patients/{patient_id}/next-visit-type",
                          headers=desk).json()
    check("no history -> screening", decision["visit_type"] == "screening",
          decision["reason"])

    response = client.post(f"{BASE}/visits", headers=desk, json={
        "patient_id": patient_id, "visit_type": "follow_up", "mmse": 26, "cdr": 0.5})
    check("follow-up before any screening -> 400", response.status_code == 400,
          str(response.status_code))

    # -- numeric fields only take numbers, in range -------------------------
    good = {"patient_id": patient_id, "visit_type": "screening", "mmse": 24, "cdr": 0.5, "edu": 12, "ses": 3}
    bad_cases = {
        "MMSE as a string": {"mmse": "24"},
        "MMSE out of range (31)": {"mmse": 31},
        "MMSE with a decimal": {"mmse": 24.5},
        "CDR off the scale (0.7)": {"cdr": 0.7},
        "education negative": {"edu": -1},
        "SES out of range (6)": {"ses": 6},
        "SES as a boolean": {"ses": True},
        "education missing": {"edu": None},
    }
    rejected = []
    for label, change in bad_cases.items():
        r = client.post(f"{BASE}/visits", headers=desk, json={**good, **change})
        if r.status_code != 422:
            rejected.append(f"{label} -> {r.status_code}")
    check("every malformed clinical number -> 422 (8 cases)", not rejected, "; ".join(rejected))
    raw_nan = '{"patient_id": "%s", "visit_type": "screening", "mmse": NaN, "cdr": 0.5, "edu": 12, "ses": 3}' % patient_id
    r = client.post(f"{BASE}/visits", headers={**desk, "content-type": "application/json"}, content=raw_nan)
    check("a bare NaN literal -> 422, not a 500", r.status_code == 422, str(r.status_code))
    r = client.post(f"{BASE}/patients", headers=desk, json={
        "name": f"{PREFIX} Phone {TAG}", "consent_given_by": "patient", "phone": "call me maybe"})
    check("patient with a non-numeric phone -> 422", r.status_code == 422, str(r.status_code))

    # -- screening visit ---------------------------------------------------
    response = client.post(f"{BASE}/visits", headers=desk, json=good)
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
    check("MRI persists a 2-feature vector",
          len(after_mri.get("mri_feature_vector") or []) == 2,
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
    check("legacy svm_prediction column stays unset (SVM result lives in model_prediction now)",
          after_speech["svm_prediction"] is None,
          f"{after_speech['svm_prediction']}")
    check("QSVM research view is computed alongside the canonical SVM",
          after_speech.get("qsvm_prediction") in ("Demented", "Nondemented")
          and after_speech.get("qsvm_confidence") is not None,
          f"{after_speech.get('qsvm_prediction')} margin={after_speech.get('qsvm_confidence')}")
    meta = after_speech.get("mri_extraction") or {}
    check("MRI extraction metadata is stored (kind, format, method)",
          meta.get("input_kind") == "volume" and meta.get("source_format") == "nifti"
          and meta.get("method") in ("calibrated", "uncalibrated-heuristic"),
          json.dumps({k: meta.get(k) for k in ("input_kind", "source_format", "method")}))

    info = client.get(f"{BASE}/models/info", headers=doctor)
    check("model comparison data is served for the toggle",
          info.status_code == 200
          and {m["key"] for m in info.json()["models"]} == {"svm", "qsvm"},
          f"{info.status_code}")

    persisted = client.get(f"{BASE}/visits/{visit_id}", headers=desk).json()
    check("prediction is persisted, not just echoed",
          persisted["model_prediction"] == model_says)

    # -- re-upload: the receptionist picked the wrong file the first time -----
    # Still pending_review (no diagnosis saved yet) -- re-upload must be accepted,
    # overwrite the feature vector, and recompute the prediction from it.
    before_mri_vector = persisted["mri_feature_vector"]
    response = client.post(f"{BASE}/visits/{visit_id}/mri-upload", headers=desk,
                           files={"file": ("s2.nii.gz", make_nifti(inner_radius=8.0), "application/gzip")})
    check("re-upload before a diagnosis is saved -> 200, not rejected",
          response.status_code == 200, str(response.status_code))
    after_reupload = response.json()
    check("re-upload overwrites the feature vector",
          after_reupload["mri_feature_vector"] != before_mri_vector,
          f"{before_mri_vector} -> {after_reupload['mri_feature_vector']}")
    check("re-upload recomputes the prediction (still present, status back to pending_review)",
          after_reupload["model_prediction"] is not None
          and after_reupload["status"] == "pending_review",
          f"{after_reupload['model_prediction']} / {after_reupload['status']}")
    # -- any MRI format: a 2-D slice, an Analyze pair as two files, a zipped
    # DICOM series. Each is a re-upload, so each also re-runs both models.
    slice_png = make_png_slice()
    response = client.post(f"{BASE}/visits/{visit_id}/mri-upload", headers=desk,
                           files={"file": ("axial.png", slice_png, "image/png")})
    body = response.json() if response.status_code == 200 else {}
    check("MRI as a PNG slice -> 200, flagged as a slice with eTIV not measured",
          response.status_code == 200
          and (body.get("mri_extraction") or {}).get("input_kind") == "slice"
          and (body.get("mri_extraction") or {}).get("etiv_measured") is False,
          f"{response.status_code} {response.text[:160]}")

    header, image = make_analyze_pair()
    response = client.post(f"{BASE}/visits/{visit_id}/mri-upload", headers=desk,
                           files=[("files", ("scan.hdr", header, "application/octet-stream")),
                                  ("files", ("scan.img", image, "application/octet-stream"))])
    body = response.json() if response.status_code == 200 else {}
    stored = client.get(f"{BASE}/visits/{visit_id}/file/mri", headers=doctor)
    stored_name = stored.json().get("filename", "") if stored.status_code == 200 else ""
    check("MRI as an Analyze .hdr + .img pair (two files) -> 200, stored as one bundle",
          response.status_code == 200
          and (body.get("mri_extraction") or {}).get("input_kind") == "volume"
          and stored_name.endswith("_bundle.zip"),
          f"{response.status_code} stored as {stored_name!r}")
    r = client.get(f"{BASE}/visits/{visit_id}/scan", headers=doctor)
    check("the bundled Analyze pair still opens in the viewer",
          r.status_code == 200 and r.content[:2] == b"\x1f\x8b", f"{r.status_code}")

    response = client.post(f"{BASE}/visits/{visit_id}/mri-upload", headers=desk,
                           files={"file": ("series.zip", make_dicom_series_zip(), "application/zip")})
    body = response.json() if response.status_code == 200 else {}
    check("MRI as a zipped DICOM series -> 200",
          response.status_code == 200
          and (body.get("mri_extraction") or {}).get("source_format") == "dicom-series",
          f"{response.status_code} {response.text[:160]}")

    response = client.post(f"{BASE}/visits/{visit_id}/mri-upload", headers=desk,
                           files={"file": ("broken.nii.gz", b"\x1f\x8bnot really", "application/gzip")})
    check("a corrupt scan is a clean 422, not a 500", response.status_code == 422,
          f"{response.status_code} {response.text[:120]}")

    # Back to a NIfTI volume for the rest of the flow.
    response = client.post(f"{BASE}/visits/{visit_id}/mri-upload", headers=desk,
                           files={"file": ("s3.nii.gz", make_nifti(), "application/gzip")})
    response.raise_for_status()
    after_reupload = response.json()
    # The re-uploads may have flipped the label -- resync model_says so the
    # rest of the flow (the mismatch path below) uses the current prediction.
    model_says = after_reupload["model_prediction"]

    # -- the clinician can pull the raw scan / recording back -------------
    for kind in ("mri", "speech"):
        r = client.get(f"{BASE}/visits/{visit_id}/file/{kind}", headers=doctor)
        check(f"clinician gets a signed URL for the {kind} upload",
              r.status_code == 200 and bool(r.json().get("url")),
              f"{r.status_code} {r.text[:120]}")

    r = client.get(f"{BASE}/visits/{visit_id}/scan", headers=doctor)
    check("clinician gets the scan as gzip NIfTI for the viewer",
          r.status_code == 200
          and r.headers.get("content-type") == "application/gzip"
          and r.content[:2] == b"\x1f\x8b",
          f"{r.status_code} {r.headers.get('content-type')} {len(r.content)}b")

    r = client.get(f"{BASE}/visits/{visit_id}/audio", headers=doctor)
    check("clinician gets the recording as playable MP3",
          r.status_code == 200
          and r.headers.get("content-type") == "audio/mpeg"
          and r.content[:3] in (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"),
          f"{r.status_code} {r.headers.get('content-type')} {len(r.content)}b")

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

        # Re-posting the identical diagnosis is a no-op — refused, and no
        # history row written for it.
        again = client.post(f"{BASE}/visits/{visit_id}/diagnosis", headers=doctor,
                            json={"doctor_diagnosis": model_says,
                                  "doctor_notes": "revised same day"})
        check("identical diagnosis re-save -> 400 'No changes'",
              again.status_code == 400 and "No changes" in again.text,
              f"{again.status_code} {again.text[:120]}")
        hist = client.get(f"{BASE}/visits/{visit_id}", headers=doctor).json()
        check("the rejected no-op wrote no history row",
              len(hist.get("diagnosis_history") or []) == 2,
              str(len(hist.get("diagnosis_history") or [])))

    # -- re-upload after the diagnosis is saved must now be refused ----------
    response = client.post(f"{BASE}/visits/{visit_id}/mri-upload", headers=desk,
                           files={"file": ("s3.nii.gz", make_nifti(inner_radius=12.0), "application/gzip")})
    check("re-upload after a diagnosis is saved -> 400, not silently accepted",
          response.status_code == 400 and "already saved" in response.text,
          f"{response.status_code} {response.text[:120]}")

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

    r = client.get(f"{BASE}/visits/{follow_up['id']}/file/mri", headers=desk)
    check("a follow-up has no MRI file -> 404", r.status_code == 404,
          str(r.status_code))

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
        "pincode": "110001", "admin_name": "Otto Klein",
        "email_domain": f"other-{TAG}.smoke.test", "password": "smoketest123"})
    response.raise_for_status()
    outsider = {"Authorization": f"Bearer {response.json()['token']}"}

    response = client.get(f"{BASE}/patients/{patient_id}", headers=outsider)
    check("Rule 12: another hospital's patient -> 404, never 403",
          response.status_code == 404, str(response.status_code))
    response = client.get(f"{BASE}/visits/{visit_id}", headers=outsider)
    check("Rule 12: another hospital's visit -> 404, never 403",
          response.status_code == 404, str(response.status_code))
    response = client.get(f"{BASE}/visits/{visit_id}/file/mri", headers=outsider)
    check("Rule 12: another hospital's visit file -> 404, never 403",
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
