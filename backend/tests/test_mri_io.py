"""mri_io.load_scan: every accepted format decodes to the same volume, and
every malformed upload fails with a user-facing ScanLoadError."""
from __future__ import annotations

import gzip
import io
import zipfile

import numpy as np
import pytest

from app.services.mri_io import ScanLoadError, has_accepted_extension, load_scan

AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def nifti_pair_bytes(vol, kind="analyze"):
    import nibabel as nib

    cls = nib.AnalyzeImage if kind == "analyze" else nib.Nifti1Pair
    image = cls(vol.astype(np.int16), AFFINE)
    header, data = io.BytesIO(), io.BytesIO()
    image.to_file_map({"header": nib.FileHolder(fileobj=header), "image": nib.FileHolder(fileobj=data)})
    return header.getvalue(), data.getvalue()


def zip_bytes(members):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, raw in members:
            archive.writestr(name, raw)
    return buffer.getvalue()


def dicom_slice(pixels, z_index, spacing=(2.0, 2.0, 2.0), series_uid="1.2.3.4"):
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = series_uid
    ds.Modality = "MR"
    ds.InstanceNumber = z_index + 1
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelSpacing = [spacing[0], spacing[1]]
    ds.SliceThickness = spacing[2]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0.0, 0.0, z_index * spacing[2]]
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    buffer = io.BytesIO()
    ds.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def dicom_series(vol, shuffle=True):
    slices = [dicom_slice(vol[:, :, z].T, z) for z in range(vol.shape[2])]
    order = np.random.default_rng(1).permutation(len(slices)) if shuffle else range(len(slices))
    return [(f"IM{i:04d}", slices[i]) for i in order]  # no extensions, out of order


def image_bytes(array, fmt="PNG", mode="L"):
    from PIL import Image

    img = Image.fromarray((array / array.max() * 255).astype(np.uint8)).convert(mode)
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# volumes
# --------------------------------------------------------------------------- #
def test_nifti_and_gzip(head_volume, nifti_bytes):
    for name, raw in (("a.nii", nifti_bytes), ("a.nii.gz", gzip.compress(nifti_bytes))):
        scan = load_scan([(name, raw)])
        assert scan.kind == "volume" and scan.source_format == "nifti"
        assert scan.data.shape == head_volume.shape
        assert scan.spacing == (2.0, 2.0, 2.0)
        assert np.allclose(scan.data, head_volume, atol=1e-3)


def test_mgz(head_volume):
    import nibabel as nib

    raw = gzip.compress(nib.MGHImage(head_volume, AFFINE).to_bytes())
    scan = load_scan([("brain.mgz", raw)])
    assert scan.source_format == "mgh" and scan.data.shape == head_volume.shape


@pytest.mark.parametrize("kind", ["analyze", "nifti-pair"])
def test_hdr_img_pair_as_two_files_and_zipped(head_volume, kind):
    header, data = nifti_pair_bytes(head_volume, kind)
    for files in (
        [("scan.hdr", header), ("scan.img", data)],
        [("scan.img", data), ("scan.hdr", header)],  # order doesn't matter
        [("pair.zip", zip_bytes([("folder/scan.hdr", header), ("folder/scan.img", data)]))],
    ):
        scan = load_scan(files)
        assert scan.kind == "volume"
        assert scan.data.shape == head_volume.shape
        assert scan.spacing == (2.0, 2.0, 2.0)


def test_lonely_img_is_rejected_with_a_helpful_message(head_volume):
    _, data = nifti_pair_bytes(head_volume)
    with pytest.raises(ScanLoadError, match="matching .hdr"):
        load_scan([("scan.img", data)])


def test_dicom_series_out_of_order_without_extensions(head_volume):
    small = head_volume[::2, ::2, ::2]
    members = dicom_series(small)
    for files in (members, [("series.zip", zip_bytes(members))]):
        scan = load_scan(files)
        assert scan.kind == "volume" and scan.source_format == "dicom-series"
        assert scan.data.shape == small.shape
        assert scan.spacing == (2.0, 2.0, 2.0)
        # Slices must be restacked by position, not by (shuffled) filename.
        assert np.allclose(scan.data, np.round(small), atol=1.0)


def test_dicom_series_picks_the_largest_series(head_volume):
    small = head_volume[::4, ::4, ::4]
    main = dicom_series(small, shuffle=False)
    localiser = [("loc1", dicom_slice(small[:, :, 0].T, 0, series_uid="9.9.9"))]
    scan = load_scan(main + localiser)
    assert scan.data.shape == small.shape
    assert any("series" in note for note in scan.notes)


def test_single_dicom_slice_is_a_slice(head_volume):
    raw = dicom_slice(head_volume[:, :, 48].T, 0)
    scan = load_scan([("one.dcm", raw)])
    assert scan.kind == "slice"
    assert scan.data.shape == head_volume[:, :, 48].T.shape


# --------------------------------------------------------------------------- #
# images
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fmt,mode,name", [("PNG", "L", "s.png"), ("JPEG", "RGB", "s.jpg"), ("BMP", "L", "s.bmp"), ("TIFF", "L", "s.tif")])
def test_images_become_greyscale_slices(head_volume, fmt, mode, name):
    raw = image_bytes(head_volume[:, :, 48], fmt, mode)
    scan = load_scan([(name, raw)])
    assert scan.kind == "slice" and scan.spacing is None
    assert scan.data.ndim == 2 and scan.data.shape == head_volume[:, :, 48].shape


def test_tiny_image_rejected():
    with pytest.raises(ScanLoadError, match="too small"):
        load_scan([("x.png", image_bytes(np.ones((10, 10)) * 100))])


# --------------------------------------------------------------------------- #
# junk
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "files,match",
    [
        ([], "No MRI file"),
        ([("x.nii", b"not a nifti at all")], "could not be read"),
        ([("x.nii.gz", gzip.compress(b"garbage" * 100))], "could not be read"),
        ([("x.zip", b"PK\x03\x04broken")], "damaged"),
        ([("x.zip", zip_bytes([]))], "empty"),
        ([("x.png", b"\x89PNG\r\n\x1a\nnope")], "could not be read"),
        ([("notes.txt", b"hello")], "Unsupported"),
    ],
)
def test_bad_uploads_raise_user_facing_errors(files, match):
    with pytest.raises(ScanLoadError, match=match):
        load_scan(files)


def test_zip_bomb_guard(monkeypatch):
    import app.services.mri_io as mri_io

    monkeypatch.setattr(mri_io, "MAX_UNCOMPRESSED_BYTES", 1000)
    raw = zip_bytes([("a.dcm", b"\0" * 5000)])
    with pytest.raises(ScanLoadError, match="too large"):
        load_scan([("big.zip", raw)])


def test_extension_check():
    for ok in ("a.nii", "A.NII.GZ", "b.mgz", "c.hdr", "d.dcm", "e.zip", "f.png", "g.JPEG", "dir/h.tif"):
        assert has_accepted_extension(ok), ok
    for bad in ("a.txt", "b.exe", "c", "d.nii.bak"):
        assert not has_accepted_extension(bad), bad
