"""MRI extractor v2: robust features, the calibrated artifact, and behaviour on
real OASIS scans of every input type (skipped when data/ isn't present)."""
from __future__ import annotations

import gzip
import io
import math

import numpy as np
import pytest

from app.services import mri_features
from app.services.mri_features import (
    ETIV_RANGE,
    NWBV_RANGE,
    MriFeatureExtractionError,
    compute_features,
    extract_from_files,
    feature_names,
)
from app.services.mri_io import load_scan
from conftest import OAS1_SUBJECT, OAS2_SESSION, make_head, needs_oasis
from test_mri_io import dicom_series, image_bytes, nifti_pair_bytes, zip_bytes


def nifti(vol, spacing=2.0):
    import nibabel as nib

    return nib.Nifti1Image(vol, np.diag([spacing, spacing, spacing, 1.0])).to_bytes()


def test_feature_contract_is_stable(head_volume):
    scan = load_scan([("h.nii", nifti(head_volume))])
    kind, feats, _ = compute_features(scan)
    assert kind == "volume"
    assert list(feats)  # non-empty
    assert set(feature_names("volume")) <= set(feats)
    assert all(math.isfinite(feats[name]) for name in feature_names("volume"))


def test_extraction_is_in_range_and_explains_itself(nifti_bytes):
    result = extract_from_files([("h.nii", nifti_bytes)])
    nwbv, etiv = result.features
    assert NWBV_RANGE[0] <= nwbv <= NWBV_RANGE[1]
    assert ETIV_RANGE[0] <= etiv <= ETIV_RANGE[1]
    meta = result.meta
    assert meta["input_kind"] == "volume" and meta["etiv_measured"] is True
    assert meta["method"] == "calibrated"
    assert "nwbv" in meta["expected_error"]


def test_unrealistic_input_is_flagged(head_volume):
    """The phantom's flat tissue bands are far outside real MRI statistics; the
    extractor must still answer inside the plausible range and say so."""
    result = extract_from_files([("h.nii", nifti(head_volume))])
    assert any("outside the range seen in training" in w for w in result.meta["warnings"])


def test_dicom_series_decodes_like_nifti(head_volume):
    """Same voxels via a DICOM series -> same loaded volume -> same features."""
    import test_mri_io as tio

    small = head_volume[::2, ::2, ::2]
    rounded = np.round(small)
    files = [(f"IM{z}", tio.dicom_slice(rounded[:, :, z].T, z, spacing=(4.0, 4.0, 4.0))) for z in range(small.shape[2])]
    via_dicom = load_scan(files)
    via_nifti = load_scan([("s.nii", nifti(rounded, spacing=4.0))])
    assert np.allclose(via_dicom.data, via_nifti.data)
    assert compute_features(via_dicom)[1] == pytest.approx(compute_features(via_nifti)[1])


def test_two_d_image_flags_unmeasurable_etiv(head_volume):
    result = extract_from_files([("slice.png", image_bytes(head_volume[:, :, 48]))])
    assert result.meta["input_kind"] == "slice"
    assert result.meta["etiv_measured"] is False
    assert result.features[1] == pytest.approx(mri_features._load_calibration()["slice"]["etiv_fallback"], abs=1)
    assert any("eTIV" in warning for warning in result.meta["warnings"])


def test_blank_and_empty_inputs_fail_cleanly():
    with pytest.raises(MriFeatureExtractionError, match="blank"):
        extract_from_files([("z.nii", nifti(np.zeros((40, 40, 40), np.float32)))])
    with pytest.raises(MriFeatureExtractionError):
        extract_from_files([("z.nii", b"junk")])


def test_too_few_slices_fail_cleanly():
    with pytest.raises(MriFeatureExtractionError, match="too few slices"):
        extract_from_files([("thin.nii", nifti(make_head(shape=(96, 112, 3)), spacing=2.0))])


def test_missing_calibration_degrades_not_fails(monkeypatch, nifti_bytes):
    monkeypatch.setattr(mri_features, "_load_calibration", lambda: None)
    result = extract_from_files([("h.nii", nifti_bytes)])
    assert result.meta["method"] == "uncalibrated-heuristic"
    assert any("Calibration file missing" in w for w in result.meta["warnings"])


def test_calibration_artifact_matches_feature_contract():
    payload = mri_features._load_calibration()
    assert payload is not None and payload["version"] == mri_features.CALIBRATION_VERSION
    for kind in ("volume", "slice"):
        assert payload[kind]["feature_names"] == feature_names(kind)
        assert payload[kind]["stats"]["nwbv"]["mae"] < 0.035


# --------------------------------------------------------------------------- #
# real scans
# --------------------------------------------------------------------------- #
def _pair(path):
    return [(path.name, path.read_bytes()), (path.with_suffix(".img").name, path.with_suffix(".img").read_bytes())]


def _truth(session):
    import pandas as pd

    from conftest import DATA

    if session.startswith("OAS1"):
        df = pd.read_excel(DATA / "oasis1_clinical data.xlsx").set_index("ID")
    else:
        df = pd.read_excel(DATA / "oasis2_mri_clinical.xlsx").set_index("MRI ID")
    return float(df.loc[session, "nWBV"]), float(df.loc[session, "eTIV"])


@needs_oasis
@pytest.mark.parametrize(
    "label,relpath",
    [
        ("oas1 raw (skull on)", "RAW/OAS1_0001_MR1_mpr-1_anon.hdr"),
        ("oas1 atlas", "PROCESSED/MPRAGE/T88_111/OAS1_0001_MR1_mpr_n4_anon_111_t88_gfc.hdr"),
        ("oas1 skull-stripped", "PROCESSED/MPRAGE/T88_111/OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc.hdr"),
    ],
)
def test_real_oasis1_scans_land_near_truth(label, relpath):
    # In-sample for the calibration (this subject was in the fit), so this is a
    # regression check that the pipeline runs end to end on real files, not an
    # accuracy claim. The out-of-sample numbers live in the calibration stats.
    true_nwbv, _ = _truth("OAS1_0001_MR1")
    result = extract_from_files(_pair(OAS1_SUBJECT / relpath))
    assert abs(result.features[0] - true_nwbv) < 0.06, label


@needs_oasis
def test_real_oasis2_raw_and_its_dicom_conversion_agree():
    hdr = OAS2_SESSION / "RAW" / "mpr-1.nifti.hdr"
    direct = extract_from_files(_pair(hdr))
    true_nwbv, true_etiv = _truth("OAS2_0001_MR1")
    assert abs(direct.features[0] - true_nwbv) < 0.06
    assert abs(direct.features[1] - true_etiv) < 300

    scan = load_scan(_pair(hdr))
    import test_mri_io as tio

    vol = scan.data
    spacing = scan.spacing
    files = [(f"IM{z}", tio.dicom_slice(vol[:, :, z].T, z, spacing=spacing)) for z in range(vol.shape[2])]
    via_dicom = extract_from_files(files)
    assert via_dicom.features[0] == pytest.approx(direct.features[0], abs=0.02)


@needs_oasis
def test_real_scan_same_numbers_across_containers():
    """A scan's values mustn't depend on how it was packaged."""
    import nibabel as nib

    pair = _pair(OAS2_SESSION / "RAW" / "mpr-1.nifti.hdr")
    reference = extract_from_files(pair).features
    scan = load_scan(pair)
    as_nifti = nib.Nifti1Image(scan.data, scan.affine).to_bytes()
    variants = {
        "nii": [("s.nii", as_nifti)],
        "nii.gz": [("s.nii.gz", gzip.compress(as_nifti))],
        "zip": [("s.zip", zip_bytes(pair))],
    }
    for name, files in variants.items():
        values = extract_from_files(files).features
        assert values[0] == pytest.approx(reference[0], abs=0.005), name
        assert values[1] == pytest.approx(reference[1], rel=0.02), name


@needs_oasis
def test_real_scans_are_robust_to_noise_and_intensity_scaling():
    import copy

    scan = load_scan(_pair(OAS2_SESSION / "RAW" / "mpr-1.nifti.hdr"))
    base = mri_features.extract_from_scan(scan).features
    noisy = copy.copy(scan)
    noisy.data = scan.data + np.random.default_rng(0).normal(0, 2, scan.data.shape).astype(np.float32)
    rescaled = copy.copy(scan)
    rescaled.data = scan.data * 1.7 + 30
    for variant in (noisy, rescaled):
        values = mri_features.extract_from_scan(variant).features
        assert values[0] == pytest.approx(base[0], abs=0.005)
        assert values[1] == pytest.approx(base[1], rel=0.02)


@needs_oasis
def test_real_atrophy_ordering_is_preserved():
    """The most- and least-atrophied OASIS-2 sessions on disk keep their order."""
    import pandas as pd

    from conftest import DATA

    truth = pd.read_excel(DATA / "oasis2_mri_clinical.xlsx").set_index("MRI ID")["nWBV"]
    on_disk = [d.name for d in (DATA / "OAS2_RAW_PART1").glob("OAS2_*_MR*") if d.name in truth.index]
    low, high = truth[on_disk].idxmin(), truth[on_disk].idxmax()
    extracted = {
        session: extract_from_files(_pair(DATA / "OAS2_RAW_PART1" / session / "RAW" / "mpr-1.nifti.hdr")).features[0]
        for session in (low, high)
    }
    assert extracted[low] < extracted[high]


@needs_oasis
def test_real_gif_slice():
    gif = OAS1_SUBJECT / "PROCESSED" / "MPRAGE" / "T88_111" / "OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc_tra_90.gif"
    result = extract_from_files([(gif.name, gif.read_bytes())])
    true_nwbv, _ = _truth("OAS1_0001_MR1")
    assert result.meta["input_kind"] == "slice"
    assert abs(result.features[0] - true_nwbv) < 0.08
