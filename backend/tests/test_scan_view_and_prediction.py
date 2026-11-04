"""The in-browser viewer conversion, and the dual-model prediction seam."""
from __future__ import annotations

import gzip
import uuid

import numpy as np
import pytest

from app.models import Visit
from app.services import prediction
from app.services.scan_view import to_viewer_nifti
from test_mri_io import dicom_series, image_bytes, nifti_pair_bytes, zip_bytes


# --------------------------------------------------------------------------- #
# scan_view
# --------------------------------------------------------------------------- #
def _read_nifti(raw):
    import nibabel as nib

    return nib.Nifti1Image.from_bytes(gzip.decompress(raw))


def test_viewer_accepts_everything_the_extractor_accepts(head_volume, nifti_bytes):
    header, data = nifti_pair_bytes(head_volume)
    small = head_volume[::4, ::4, ::4]
    cases = {
        "scan.nii": nifti_bytes,
        "scan.nii.gz": gzip.compress(nifti_bytes),
        "pair_bundle.zip": zip_bytes([("a.hdr", header), ("a.img", data)]),
        "series_bundle.zip": zip_bytes(dicom_series(small)),
        "slice.png": image_bytes(head_volume[:, :, 48]),
    }
    for name, raw in cases.items():
        image = _read_nifti(to_viewer_nifti(raw, name))
        assert image.ndim == 3, name
        assert np.isfinite(np.asarray(image.dataobj)).all(), name


def test_viewer_slice_is_upright(head_volume):
    picture = np.zeros((64, 48))
    picture[:10, :] = 255  # bright band at the TOP of the picture
    image = _read_nifti(to_viewer_nifti(image_bytes(picture + 1), "top.png"))
    data = np.asarray(image.dataobj)[:, :, 0]  # (x, y) with y pointing up
    assert data.shape == (48, 64)
    assert data[:, -5].mean() > data[:, 5].mean()  # top of picture = high y


def test_viewer_rejects_junk():
    with pytest.raises(ValueError):
        to_viewer_nifti(b"nope", "x.dcm")


# --------------------------------------------------------------------------- #
# prediction seam
# --------------------------------------------------------------------------- #
# A real training row (data/multimodal_dementia_dataset.csv, Label 1), as used by
# backend/scripts/smoke_prediction.py.
MRI = [0.696, 1987.0]
SPEECH = [0.29, 2.6, 160.0, 0.02, 0.1] + [-300.0, 100.0, 10.0, 20.0, 0.0, 5.0, -5.0, 2.0, -2.0, 1.0, 0.0, 1.0, -1.0]


def make_visit(**overrides):
    fields = dict(
        id=uuid.uuid4(), visit_type="screening", mri_status="done", speech_status="done",
        mri_feature_vector=MRI, speech_feature_vector=SPEECH, mmse=27.0, edu=12.0, ses=2.0,
        status="awaiting_uploads", model_prediction=None,
    )
    fields.update(overrides)
    return Visit(**fields)


def test_feature_row_layout():
    row = prediction.build_feature_row(make_visit())
    assert len(row) == 24
    assert row[1] == pytest.approx(prediction.ASF_ETIV_CONST / MRI[1])
    assert row[4:6] == MRI


@pytest.mark.parametrize("bad", [{"mri_feature_vector": [1.0]}, {"speech_feature_vector": [0.0] * 17}, {"mri_feature_vector": [0.7, 0.0]}])
def test_feature_row_rejects_malformed_vectors(bad):
    with pytest.raises(ValueError):
        prediction.build_feature_row(make_visit(**bad))


def test_both_models_run_and_svm_stays_canonical():
    visit = make_visit()
    assert prediction.check_and_run_prediction(None, visit) is True
    assert visit.status == "pending_review"
    assert visit.model_prediction in ("Demented", "Nondemented")
    assert visit.qsvm_prediction in ("Demented", "Nondemented")
    assert visit.model_confidence >= 0 and visit.qsvm_confidence >= 0
    assert visit.svm_prediction is None  # legacy column no longer written


def test_qsvm_failure_never_blocks_the_canonical_prediction(monkeypatch):
    class Broken:
        def decision_function(self, rows):
            raise RuntimeError("simulator crashed")

    monkeypatch.setattr(prediction, "_load_qsvm", lambda: Broken())
    visit = make_visit()
    assert prediction.check_and_run_prediction(None, visit) is True
    assert visit.model_prediction is not None
    assert visit.qsvm_prediction is None and visit.qsvm_confidence is None


def test_missing_qsvm_pickle_leaves_research_view_empty(monkeypatch):
    monkeypatch.setattr(prediction, "_load_qsvm", lambda: None)
    visit = make_visit()
    prediction.check_and_run_prediction(None, visit)
    assert visit.model_prediction is not None and visit.qsvm_prediction is None


def test_not_ready_or_already_predicted_is_a_no_op():
    assert prediction.check_and_run_prediction(None, make_visit(speech_status="idle")) is False
    assert prediction.check_and_run_prediction(None, make_visit(visit_type="follow_up")) is False
    done = make_visit(model_prediction="Demented")
    assert prediction.check_and_run_prediction(None, done) is False


def test_clear_stale_prediction_clears_both_models():
    visit = make_visit()
    prediction.check_and_run_prediction(None, visit)
    prediction.clear_stale_prediction(visit)
    assert visit.model_prediction is None and visit.qsvm_prediction is None
    assert prediction.check_and_run_prediction(None, visit) is True
