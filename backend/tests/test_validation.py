"""Input validation, at the schema level and over HTTP.

Every numeric clinical field must be a real number of the right kind and in
range; strings, booleans, NaN and Infinity are refused. The HTTP tests override
auth and the DB, so a 422 here proves the request never reached a handler.
"""
from __future__ import annotations

import json
import uuid
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.schemas import DiagnosisCreate, HospitalUpdate, PatientCreate, VisitCreate

PID = str(uuid.uuid4())


def screening(**fields):
    return {"patient_id": PID, "visit_type": "screening", "mmse": 28, "cdr": 0, "edu": 16, "ses": 2, **fields}


# --------------------------------------------------------------------------- #
# VisitCreate
# --------------------------------------------------------------------------- #
def test_valid_screening_and_follow_up():
    VisitCreate(**screening())
    VisitCreate(**screening(cdr=0.5, mmse=0, edu=0, ses=5))
    VisitCreate(patient_id=PID, visit_type="follow_up", mmse=30, cdr=3)
    assert VisitCreate(**screening(mmse=28.0)).mmse == 28.0  # 28.0 is a whole number


@pytest.mark.parametrize(
    "field,value",
    [
        ("mmse", "28"), ("mmse", True), ("mmse", -1), ("mmse", 31), ("mmse", 27.5),
        ("cdr", 0.7), ("cdr", 4), ("cdr", "0.5"), ("cdr", -0.5),
        ("edu", 31), ("edu", -2), ("edu", 12.5), ("edu", "twelve"),
        ("ses", 0), ("ses", 6), ("ses", 2.5), ("ses", False),
    ],
)
def test_bad_numbers_rejected(field, value):
    with pytest.raises(ValidationError):
        VisitCreate(**screening(**{field: value}))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nan_and_infinity_rejected(value):
    with pytest.raises(ValidationError):
        VisitCreate(**screening(mmse=value))


@pytest.mark.parametrize("missing", ["mmse", "cdr", "edu", "ses"])
def test_screening_requires_every_measure(missing):
    body = screening()
    body.pop(missing)
    with pytest.raises(ValidationError, match=missing):
        VisitCreate(**body)


def test_follow_up_refuses_screening_only_fields():
    with pytest.raises(ValidationError, match="screening visits only"):
        VisitCreate(patient_id=PID, visit_type="follow_up", mmse=25, cdr=1, edu=12)
    with pytest.raises(ValidationError, match="mmse"):
        VisitCreate(patient_id=PID, visit_type="follow_up", cdr=1)


# --------------------------------------------------------------------------- #
# PatientCreate and friends
# --------------------------------------------------------------------------- #
def patient(**fields):
    return {"name": "Asha Rao", "consent_given_by": "patient", **fields}


def test_valid_patient():
    PatientCreate(**patient(phone="+91 98765 43210", dob="1950-02-01", sex="Female", address="12 MG Road"))


@pytest.mark.parametrize(
    "fields",
    [
        {"phone": "12345"},
        {"phone": "98765abc12"},
        {"phone": "1" * 16},
        {"dob": (date.today() + timedelta(days=1)).isoformat()},
        {"dob": "1890-01-01"},
        {"dob": "not-a-date"},
        {"sex": "M"},
        {"name": "12345"},
        {"name": "A"},
        {"address": "x" * 301},
        {"address": "bad\x00text"},
        {"consent_given_by": "guardian"},
        {"consent_given_by": "guardian", "consent_relationship": "123"},
        {"consent_given_by": "someone"},
    ],
)
def test_bad_patients_rejected(fields):
    with pytest.raises(ValidationError):
        PatientCreate(**patient(**fields))


def test_blank_optional_fields_become_none():
    created = PatientCreate(**patient(phone="  ", sex="", address="   "))
    assert created.phone is None and created.sex is None and created.address is None


def test_hospital_city_and_pincode():
    HospitalUpdate(city="New Delhi", pincode="110001")
    for bad in ({"city": "Delhi 6"}, {"city": "@@"}, {"pincode": "1100"}, {"pincode": "11000a"}):
        with pytest.raises(ValidationError):
            HospitalUpdate(**bad)


def test_diagnosis_notes_length():
    DiagnosisCreate(doctor_diagnosis="Demented", doctor_notes="x" * 4000)
    with pytest.raises(ValidationError):
        DiagnosisCreate(doctor_diagnosis="Demented", doctor_notes="x" * 4001)
    with pytest.raises(ValidationError):
        DiagnosisCreate(doctor_diagnosis="Maybe")


# --------------------------------------------------------------------------- #
# over HTTP
# --------------------------------------------------------------------------- #
@pytest.fixture
def client():
    from app.db import get_db
    from app.deps import CurrentUser, get_current_user, require_receptionist
    from app.main import app

    user = CurrentUser(user_id=uuid.uuid4(), hospital_id=uuid.uuid4(), role="receptionist", name="Test Desk")

    class NoDb:
        def __getattr__(self, name):
            raise AssertionError("handler reached the database; validation should have stopped it")

    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[require_receptionist] = lambda: user
    app.dependency_overrides[get_db] = lambda: NoDb()
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize(
    "body",
    [
        screening(mmse="28"),
        screening(mmse=31),
        screening(cdr=0.7),
        screening(edu=12.5),
        screening(ses=9),
        {k: v for k, v in screening().items() if k != "edu"},
    ],
)
def test_create_visit_rejects_bad_numbers_with_422(client, body):
    response = client.post("/api/v1/visits", json=body)
    assert response.status_code == 422, response.text


def test_create_visit_rejects_raw_nan_literal(client):
    raw = json.dumps(screening()).replace('"mmse": 28', '"mmse": NaN')
    response = client.post("/api/v1/visits", content=raw, headers={"content-type": "application/json"})
    assert response.status_code == 422


def test_create_patient_rejects_bad_phone_with_422(client):
    response = client.post("/api/v1/patients", json=patient(phone="call me"))
    assert response.status_code == 422


def test_model_info_endpoint(client):
    response = client.get("/api/v1/models/info")
    assert response.status_code == 200
    data = response.json()
    keys = {model["key"] for model in data["models"]}
    assert keys == {"svm", "qsvm"}
    svm = next(model for model in data["models"] if model["key"] == "svm")
    assert svm["canonical"] is True and svm["decision_scale"] > 0
    assert 0 < svm["metrics"]["accuracy"]["mean"] <= 1
