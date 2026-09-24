from __future__ import annotations

import math
import pickle
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO, Protocol

from app.config import settings


MRI_FEATURE_ORDER: tuple[str, str] = (
    "nWBV",
    "eTIV",
)

# Feature vector shared between the live extractor and scripts/calibrate_mri_extractor.py --
# order matters, both sides must build it identically (see compute_calibration_features).
CALIBRATION_FEATURE_NAMES: tuple[str, ...] = (
    "icv_ml",
    "brain_ml",
    "nwbv_heuristic",
    "vol_ratio_p30",
    "vol_ratio_p40",
    "vol_ratio_p50",
    "vol_ratio_p60",
    "vol_ratio_p70",
    "foreground_mean",
    "foreground_std",
)

CALIBRATION_PICKLE = "mri_extractor_calibration.pkl"


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
    works end-to-end -- intensity thresholding and geometric heuristics, not a real
    neuroimaging pipeline (FreeSurfer/FSL/ANTs). **2026-09-25:** its raw output used to be
    served directly and was found to badly compress real atrophy differences (three real
    OASIS subjects with true nWBV 0.83/0.76/0.67 all extracted to ~0.85-0.88, flipping
    predictions). It's now calibrated: the same heuristic masks feed a small Ridge regression
    fit against 39 real OASIS-1 scans' true nWBV/eTIV (`scripts/calibrate_mri_extractor.py`,
    LOOCV-validated, stats in the loaded pickle). Still a proxy, not real segmentation --
    the calibration is bounded by a 39-subject fit, and the model falls back to the raw
    heuristic if `results/mri_extractor_calibration.pkl` isn't present. See
    `PROJECT_CONTEXT.md` for the LOOCV numbers and the diagnosis that led here.

    Replace this implementation before using the app for research conclusions or
    clinical decision support.
    """

    volume, voxel_volume_mm3 = _load_uploaded_volume(file)
    normalized_volume = _normalize_volume(volume)
    features = compute_calibration_features(normalized_volume, voxel_volume_mm3)

    calibration = _load_calibration()
    if calibration is not None:
        x = [calibration_feature_vector(features)]
        nwbv = _clamp(float(calibration["nwbv_model"].predict(x)[0]), 0.35, 0.95)
        etiv_ml = max(float(calibration["etiv_model"].predict(x)[0]), 1.0)
    else:
        # No calibration pickle on disk -- fall back to the raw heuristic rather than fail.
        nwbv = _clamp(features["nwbv_heuristic"], 0.35, 0.95)
        etiv_ml = features["icv_ml"]

    return validate_mri_feature_vector(
        [
            round(nwbv, 4),
            round(etiv_ml, 2),
        ]
    )


def compute_calibration_features(normalized_volume: "object", voxel_volume_mm3: float) -> dict[str, float]:
    """The feature vector shared by the live extractor and the calibration script.

    Pure function (normalized volume in, features out) so calibration-fit time and serving time
    always compute it identically -- the same "verbatim shared function" pattern
    scripts/speech_03_window_features.py uses for extract_speech_features. Order of
    CALIBRATION_FEATURE_NAMES is the contract; calibration_feature_vector() enforces it.
    """
    import numpy as np

    intracranial_mask = _estimate_intracranial_mask(normalized_volume)
    brain_mask = _estimate_brain_mask(normalized_volume, intracranial_mask)

    if intracranial_mask.sum() == 0:
        raise MriFeatureExtractionError("Unable to segment MRI foreground from uploaded file.")

    icv_ml = _volume_ml(intracranial_mask, voxel_volume_mm3)
    brain_ml = _volume_ml(brain_mask, voxel_volume_mm3) if brain_mask.sum() else 0.0
    nwbv_heuristic = brain_ml / icv_ml if icv_ml > 0 else 0.0

    foreground = normalized_volume[intracranial_mask]
    features: dict[str, float] = {
        "icv_ml": icv_ml,
        "brain_ml": brain_ml,
        "nwbv_heuristic": nwbv_heuristic,
        "foreground_mean": float(foreground.mean()),
        "foreground_std": float(foreground.std()),
    }
    for p in (30, 40, 50, 60, 70):
        cutoff = np.percentile(foreground, p)
        mask_p = intracranial_mask & (normalized_volume >= cutoff)
        vol_p_ml = _volume_ml(mask_p, voxel_volume_mm3)
        features[f"vol_ratio_p{p}"] = vol_p_ml / icv_ml if icv_ml > 0 else 0.0

    return features


def calibration_feature_vector(features: dict[str, float]) -> list[float]:
    """features dict -> ordered list matching CALIBRATION_FEATURE_NAMES."""
    return [features[name] for name in CALIBRATION_FEATURE_NAMES]


def extract_calibration_features_from_path(path: Path) -> dict[str, float]:
    """Load a scan directly from a local filesystem path (no upload wrapper, no tempfile hop)
    and compute the shared calibration feature vector. Used by
    scripts/calibrate_mri_extractor.py, which reads real OASIS files directly rather than
    through the FastAPI upload path.
    """
    lowered = str(path).lower()
    if lowered.endswith((".nii", ".nii.gz", ".mgh", ".mgz", ".img", ".hdr")):
        volume, voxel_volume_mm3 = _load_nifti_like_volume(path)
    elif lowered.endswith((".dcm", ".dicom")):
        volume, voxel_volume_mm3 = _load_dicom_volume(path)
    else:
        raise MriFeatureExtractionError(f"Unsupported MRI file format: {path}")

    normalized_volume = _normalize_volume(volume)
    return compute_calibration_features(normalized_volume, voxel_volume_mm3)


def _calibration_dir() -> Path:
    """Where the calibration pickle lives -- mirrors prediction.py's _model_dir()."""
    configured = Path(settings.MODEL_DIR)
    if (configured / CALIBRATION_PICKLE).exists():
        return configured
    return Path(__file__).resolve().parents[3] / "results"


@lru_cache(maxsize=1)
def _load_calibration() -> dict[str, object] | None:
    """Load the fitted nWBV/eTIV regressors once. Cached. Returns None (not an error) if the
    pickle isn't present, so a missing calibration artifact degrades to the old raw heuristic
    rather than taking MRI upload down.
    """
    path = _calibration_dir() / CALIBRATION_PICKLE
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


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
    "CALIBRATION_FEATURE_NAMES",
    "MriFeatureExtractionError",
    "MriFeatureExtractionNotReady",
    "extract_mri_features",
    "validate_mri_feature_vector",
    "compute_calibration_features",
    "calibration_feature_vector",
    "extract_calibration_features_from_path",
]
