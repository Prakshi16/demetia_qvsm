from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path
from typing import BinaryIO, Protocol


MRI_FEATURE_ORDER: tuple[str, str] = (
    "nWBV",
    "eTIV",
)


class UploadFileLike(Protocol):
    filename: str | None
    content_type: str | None
    file: BinaryIO


class MriFeatureExtractionNotReady(RuntimeError):
    """Kept for backwards-compatible imports from the upload route."""


class MriFeatureExtractionError(ValueError):
    """Raised when an MRI file cannot be converted into the 2-feature vector."""


def validate_mri_feature_vector(features: list[float]) -> list[float]:
    if len(features) != len(MRI_FEATURE_ORDER):
        raise ValueError(
            f"MRI feature vector must contain exactly {len(MRI_FEATURE_ORDER)} values "
            f"in this order: {', '.join(MRI_FEATURE_ORDER)}."
        )

    validated = [float(value) for value in features]

    if any(not math.isfinite(value) for value in validated):
        raise ValueError("MRI feature vector contains a non-finite value.")

    return validated


def extract_mri_features(file: UploadFileLike) -> list[float]:
    """Extract a 2-value MRI feature vector from an uploaded scan.

    Feature order:
    1. nWBV
    2. eTIV

    `hippocampal_volume`/`cortical_thickness` were retired with the 27->24 real-data migration
    (real_dataset_setup.md Appendix C) -- they were synthetic-only estimates with no OASIS tabular
    counterpart, and are not part of the real training data's clinical/MRI feature set. Phase 2 of
    the build plan will eventually replace nWBV/eTIV themselves with a learned 3D-CNN embedding.

    Important:
    This is a lightweight demo extractor written from scratch so Bishal's upload module
    works end-to-end. It estimates proxy values from MRI voxel data using intensity
    thresholding and geometric heuristics. These proxies are not a substitute for a
    validated neuroimaging pipeline such as FreeSurfer, FSL, ANTs, or the real Phase 1
    preprocessing code.

    Replace this implementation before using the app for research conclusions or
    clinical decision support.
    """

    volume, voxel_volume_mm3 = _load_uploaded_volume(file)
    normalized_volume = _normalize_volume(volume)
    intracranial_mask = _estimate_intracranial_mask(normalized_volume)
    brain_mask = _estimate_brain_mask(normalized_volume, intracranial_mask)

    if intracranial_mask.sum() == 0 or brain_mask.sum() == 0:
        raise MriFeatureExtractionError("Unable to segment MRI foreground from uploaded file.")

    etiv_ml = _volume_ml(intracranial_mask, voxel_volume_mm3)
    brain_volume_ml = _volume_ml(brain_mask, voxel_volume_mm3)
    nwbv = _clamp(brain_volume_ml / etiv_ml, 0.35, 0.95)

    return validate_mri_feature_vector(
        [
            round(nwbv, 4),
            round(etiv_ml, 2),
        ]
    )


def _load_uploaded_volume(file: UploadFileLike) -> tuple["object", float]:
    filename = file.filename or ""
    suffix = _temporary_suffix(filename)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp_file:
        file.file.seek(0)
        shutil.copyfileobj(file.file, temp_file)
        temp_file.flush()

        temp_path = Path(temp_file.name)

        if filename.lower().endswith((".nii", ".nii.gz", ".mgh", ".mgz")):
            return _load_nifti_like_volume(temp_path)

        if filename.lower().endswith((".dcm", ".dicom")):
            return _load_dicom_volume(temp_path)

    raise MriFeatureExtractionError("Unsupported MRI file format.")


def _load_nifti_like_volume(path: Path) -> tuple["object", float]:
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:
        raise MriFeatureExtractionError(
            "NIfTI/MGH uploads require the 'nibabel' package in backend requirements."
        ) from exc

    image = nib.load(str(path))
    volume = np.asarray(image.get_fdata(dtype="float32"))

    if volume.ndim > 3:
        volume = volume[..., 0]

    zooms = image.header.get_zooms()[:3]
    voxel_volume_mm3 = float(abs(zooms[0] * zooms[1] * zooms[2]))

    return volume, voxel_volume_mm3


def _load_dicom_volume(path: Path) -> tuple["object", float]:
    try:
        import numpy as np
        import pydicom
    except ImportError as exc:
        raise MriFeatureExtractionError(
            "DICOM uploads require the 'pydicom' package in backend requirements."
        ) from exc

    dataset = pydicom.dcmread(str(path))
    volume = np.asarray(dataset.pixel_array, dtype="float32")

    if volume.ndim == 2:
        volume = volume[:, :, None]

    pixel_spacing = [float(value) for value in getattr(dataset, "PixelSpacing", [1.0, 1.0])]
    slice_thickness = float(getattr(dataset, "SliceThickness", 1.0))
    voxel_volume_mm3 = pixel_spacing[0] * pixel_spacing[1] * slice_thickness

    return volume, voxel_volume_mm3


def _normalize_volume(volume: "object") -> "object":
    import numpy as np

    array = np.asarray(volume, dtype="float32")
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    array = np.squeeze(array)

    if array.ndim == 2:
        array = array[:, :, None]

    if array.ndim != 3:
        raise MriFeatureExtractionError("MRI upload must contain a 2D or 3D image volume.")

    low, high = np.percentile(array, [1, 99])

    if high <= low:
        raise MriFeatureExtractionError("MRI image intensity range is too small to process.")

    clipped = np.clip(array, low, high)
    return (clipped - low) / (high - low)


def _estimate_intracranial_mask(normalized_volume: "object") -> "object":
    import numpy as np

    threshold = max(_otsu_threshold(normalized_volume), 0.05)
    mask = normalized_volume > threshold

    return _clean_binary_mask(mask)


def _estimate_brain_mask(normalized_volume: "object", intracranial_mask: "object") -> "object":
    import numpy as np

    foreground_values = normalized_volume[intracranial_mask]

    if foreground_values.size == 0:
        return intracranial_mask

    tissue_threshold = np.percentile(foreground_values, 20)
    brain_mask = intracranial_mask & (normalized_volume >= tissue_threshold)

    return _clean_binary_mask(brain_mask)


def _otsu_threshold(values: "object") -> float:
    import numpy as np

    flattened = np.asarray(values, dtype="float32").ravel()
    flattened = flattened[np.isfinite(flattened)]

    if flattened.size == 0:
        return 0.0

    hist, bin_edges = np.histogram(flattened, bins=128, range=(0.0, 1.0))
    total = flattened.size
    sum_total = float((hist * bin_edges[:-1]).sum())
    weight_background = 0.0
    sum_background = 0.0
    max_variance = -1.0
    threshold = 0.0

    for index, count in enumerate(hist):
        weight_background += count

        if weight_background == 0:
            continue

        weight_foreground = total - weight_background

        if weight_foreground == 0:
            break

        sum_background += count * bin_edges[index]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        between_class_variance = (
            weight_background
            * weight_foreground
            * (mean_background - mean_foreground)
            * (mean_background - mean_foreground)
        )

        if between_class_variance > max_variance:
            max_variance = between_class_variance
            threshold = float(bin_edges[index])

    return threshold


def _clean_binary_mask(mask: "object") -> "object":
    try:
        from scipy import ndimage
    except ImportError:
        return mask

    cleaned = ndimage.binary_opening(mask, iterations=1)
    cleaned = ndimage.binary_closing(cleaned, iterations=2)
    cleaned = ndimage.binary_fill_holes(cleaned)
    labels, label_count = ndimage.label(cleaned)

    if label_count <= 1:
        return cleaned

    component_sizes = ndimage.sum(cleaned, labels, range(1, label_count + 1))
    largest_label = int(component_sizes.argmax()) + 1

    return labels == largest_label


def _volume_ml(mask: "object", voxel_volume_mm3: float) -> float:
    import numpy as np

    return float(np.count_nonzero(mask) * voxel_volume_mm3 / 1000.0)


def _temporary_suffix(filename: str) -> str:
    lowered = filename.lower()

    if lowered.endswith(".nii.gz"):
        return ".nii.gz"

    suffix = Path(filename).suffix
    return suffix if suffix else ".mri"


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


__all__ = [
    "MRI_FEATURE_ORDER",
    "MriFeatureExtractionError",
    "MriFeatureExtractionNotReady",
    "extract_mri_features",
    "validate_mri_feature_vector",
]
