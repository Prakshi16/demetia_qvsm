"""MRI upload -> the two model inputs, ``[nWBV, eTIV]``.

How it works (extractor v2, 2026-11-04)
---------------------------------------
v1 (2026-09-25) calibrated a two-threshold heuristic against 39 processed,
skull-stripped OASIS-1 scans. That fixed the demo cases but only for that one
kind of input: a raw scan with the skull still on, a scan in a different
orientation, a DICOM series or a single picture of a slice all fell outside the
39 images it had seen.

v2 doesn't try to segment the brain precisely. Instead it builds a description
of the scan that doesn't depend on a perfect skull strip, and learns the
mapping to OASIS's true values from many more labelled scans:

1. **Load anything** (``mri_io.load_scan``): NIfTI, MGH, Analyze pairs, DICOM
   files or series, zips of those, or a 2-D image.
2. **Normalise**: reorient (done by the loader), resample a volume to 2 mm
   isotropic so every morphological step means the same thing for every scan,
   and rescale intensities to their robust 0.5–99.5 percentile range.
3. **Head mask** by Otsu thresholding, largest connected component, holes
   filled.
4. **Depth profile**: the Euclidean distance of every head voxel from the head's
   surface. Skull and scalp only ever occupy the outer ~1–1.5 cm, so the
   fraction of dark (CSF-like) voxels at each depth tracks atrophy, because
   ventricles and sulci widen as the brain shrinks. This works whether or not
   the skull is present. The CSF/tissue cut is
   Otsu's threshold on the deep interior, which is brain regardless of input.
5. **Calibrated regressors** (``scripts/calibrate_mri_extractor.py``) map
   that profile to nWBV and eTIV. They are fit on OASIS-1 raw, OASIS-1 atlas-space
   (with and without skull) and OASIS-2 raw scans with known true values,
   validated by subject-grouped CV, so no subject is in both train and test. The
   pickle records the per-input-type error.

A 2-D image has no physical scale, so depths are measured relative to the
head's size, and **eTIV cannot be measured from one slice**. It is filled
with the training median and flagged in the returned metadata. The clinician
sees that flag on the visit.

Still a proxy for a real neuroimaging pipeline (FreeSurfer/FSL), calibrated for
**T1-weighted** structural MRI. Other contrasts load and run, but the numbers
are not meaningful and the metadata says so when it can tell.
"""
from __future__ import annotations

import logging
import math
import pickle
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.config import settings
from app.services.mri_io import LoadedScan, ScanLoadError, load_scan

logger = logging.getLogger(__name__)

MRI_FEATURE_ORDER: tuple[str, str] = ("nWBV", "eTIV")

CALIBRATION_PICKLE = "mri_extractor_calibration.pkl"
CALIBRATION_VERSION = 2

TARGET_SPACING_MM = 2.0
# Depths (mm below the head surface) at which the volume profile is sampled.
VOLUME_DEPTHS_MM = (0, 4, 8, 12, 16, 20, 25, 30, 40, 50)
# Relative depths (fraction of the deepest point) — used for both kinds, and
# the only scale a 2-D picture has.
RELATIVE_DEPTHS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
# Dark-voxel cut-offs as a fraction of the interior tissue median.
DARK_FRACTIONS = (0.35, 0.5, 0.65, 0.8)
HIST_BINS = 12
TIGHT_PERCENTILES = (10, 20, 30, 40, 50, 60, 70)

# Plausible output range — nWBV in OASIS spans ~0.64–0.84; eTIV ~1100–2000 mL.
NWBV_RANGE = (0.55, 0.92)
ETIV_RANGE = (900.0, 2300.0)
# Training-set median eTIV, used when eTIV can't be measured (a 2-D picture).
ETIV_FALLBACK_ML = 1475.0


class MriFeatureExtractionNotReady(RuntimeError):
    """Kept for backwards-compatible imports from the upload route."""


class MriFeatureExtractionError(ValueError):
    """Raised when an MRI upload cannot be turned into the 2-feature vector."""


@dataclass
class MriExtraction:
    """Result of one extraction: the model inputs plus how they were obtained."""

    features: list[float]
    meta: dict = field(default_factory=dict)


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


# --------------------------------------------------------------------------- #
# public entry points
# --------------------------------------------------------------------------- #
def extract_from_files(files: list[tuple[str, bytes]]) -> MriExtraction:
    """Upload bytes -> ``MriExtraction``. Raises ``MriFeatureExtractionError``."""
    try:
        scan = load_scan(files)
    except ScanLoadError as exc:
        raise MriFeatureExtractionError(str(exc)) from exc
    return extract_from_scan(scan)


def extract_from_scan(scan: LoadedScan) -> MriExtraction:
    kind, feats, quality = compute_features(scan)
    calibration = _load_calibration()
    warnings = list(scan.notes) + quality

    meta: dict = {
        "extractor": f"v{CALIBRATION_VERSION}",
        "input_kind": kind,
        "source_format": scan.source_format,
        "shape": [int(s) for s in scan.data.shape],
        "spacing_mm": [round(float(s), 3) for s in scan.spacing] if scan.spacing else None,
        "etiv_measured": kind == "volume",
    }

    block = (calibration or {}).get(kind)
    if block is not None:
        x = [[feats[name] for name in block["feature_names"]]]
        nwbv = float(block["nwbv_model"].predict(x)[0])
        if kind == "volume":
            etiv = float(block["etiv_model"].predict(x)[0])
        else:
            etiv = float(block.get("etiv_fallback", ETIV_FALLBACK_ML))
        meta["method"] = "calibrated"
        meta["expected_error"] = {
            target: {"mae": round(stats["mae"], 4), "r2": round(stats["r2"], 3)}
            for target, stats in block.get("stats", {}).items()
        }
        gate = getattr(block["nwbv_model"], "stripped_probability", None)
        if gate is not None:
            try:
                meta["skull_stripped_probability"] = round(float(gate(x)[0]), 3)
            except AttributeError:
                pass
    else:
        # No (or an outdated) calibration artifact: degrade to an uncalibrated
        # estimate rather than taking MRI upload down.
        nwbv = 0.95 - 0.55 * feats["rel_dark50_q30"]
        etiv = feats.get("head_ml_at_12mm", ETIV_FALLBACK_ML) if kind == "volume" else ETIV_FALLBACK_ML
        meta["method"] = "uncalibrated-heuristic"
        warnings.append("Calibration file missing: values are an uncalibrated estimate.")

    if kind != "volume":
        warnings.append(
            "Single 2-D image: eTIV can't be measured from one slice, so the training "
            f"median ({ETIV_FALLBACK_ML:.0f} mL) is used. nWBV is estimated from this slice only."
        )

    clamped_nwbv = _clamp(nwbv, *NWBV_RANGE)
    clamped_etiv = _clamp(etiv, *ETIV_RANGE)
    if abs(clamped_nwbv - nwbv) > 1e-9 or abs(clamped_etiv - etiv) > 1e-9:
        warnings.append(
            "The estimate fell outside the range seen in training and was clamped; "
            "check this is a T1-weighted brain MRI."
        )

    meta["warnings"] = warnings
    features = validate_mri_feature_vector([round(clamped_nwbv, 4), round(clamped_etiv, 2)])
    return MriExtraction(features=features, meta=meta)


# --------------------------------------------------------------------------- #
# feature computation (shared verbatim with scripts/calibrate_mri_extractor.py)
# --------------------------------------------------------------------------- #
def compute_features(scan: LoadedScan) -> tuple[str, dict[str, float], list[str]]:
    """``LoadedScan`` -> (kind, feature dict, quality warnings).

    The calibration script calls this exact function on its training scans, so
    serving and fitting can never compute different features.
    """
    if scan.is_volume:
        volume = _resample(scan.data, scan.spacing or (1.0, 1.0, 1.0))
        return ("volume",) + _profile(volume, voxel_mm=TARGET_SPACING_MM)
    image = scan.data
    if image.ndim != 2:
        raise MriFeatureExtractionError("Expected a 2-D image.")
    return ("slice",) + _profile(image, voxel_mm=None)


def feature_names(kind: str) -> list[str]:
    """Column order of the feature dict for ``kind`` — a stable contract."""
    names = ["head_size", "max_depth", "interior_cut", "interior_median"]
    if kind == "volume":
        names += ["tight_ml", "tight_mean", "tight_std"] + [f"tight_ratio_p{p}" for p in TIGHT_PERCENTILES]
        names += [f"head_ml_at_{d}mm" for d in VOLUME_DEPTHS_MM]
        names += [f"dark{int(f * 100)}_at_{d}mm" for d in VOLUME_DEPTHS_MM for f in DARK_FRACTIONS]
        names += [f"otsu_dark_at_{d}mm" for d in VOLUME_DEPTHS_MM]
    names += [f"rel_size_q{int(q * 100)}" for q in RELATIVE_DEPTHS]
    names += [f"rel_dark{int(f * 100)}_q{int(q * 100)}" for q in RELATIVE_DEPTHS for f in DARK_FRACTIONS]
    names += [f"rel_otsu_dark_q{int(q * 100)}" for q in RELATIVE_DEPTHS]
    names += [f"hist_{i}" for i in range(HIST_BINS)]
    return names


def _profile(array: np.ndarray, voxel_mm: float | None) -> tuple[dict[str, float], list[str]]:
    from scipy import ndimage

    warnings: list[str] = []
    norm = _normalise(array)
    head = _head_mask(norm)
    if head.sum() < 200:
        raise MriFeatureExtractionError(
            "Could not find a head/brain in this image — is it an MRI scan?"
        )

    sampling = voxel_mm if voxel_mm is not None else 1.0
    depth = ndimage.distance_transform_edt(head) * sampling
    max_depth = float(depth.max())

    # The deep interior is brain whatever the input; split it into CSF vs tissue.
    interior = depth >= 0.35 * max_depth
    interior_values = norm[interior]
    interior_cut = _otsu(interior_values)
    tissue_values = interior_values[interior_values > interior_cut]
    interior_median = float(np.median(tissue_values)) if tissue_values.size else float(np.median(interior_values))
    interior_median = max(interior_median, 1e-3)

    unit = 1000.0 if array.ndim == 3 else 100.0  # mL for volumes, cm^2-ish for slices
    voxel_size = (sampling ** array.ndim) / unit if voxel_mm is not None else 1.0
    head_count = float(head.sum())

    feats: dict[str, float] = {
        "head_size": head_count * voxel_size if voxel_mm is not None else head_count / head.size,
        "max_depth": max_depth if voxel_mm is not None else max_depth / math.sqrt(head_count / math.pi),
        "interior_cut": interior_cut,
        "interior_median": interior_median,
    }

    def region_stats(region: np.ndarray, prefix: str) -> None:
        values = norm[region]
        if values.size == 0:
            for f in DARK_FRACTIONS:
                feats[f"{prefix['dark']}{int(f * 100)}{prefix['suffix']}"] = 0.0
            feats[prefix["otsu"]] = 0.0
            return
        for f in DARK_FRACTIONS:
            feats[f"{prefix['dark']}{int(f * 100)}{prefix['suffix']}"] = float(
                np.mean(values < f * interior_median)
            )
        feats[prefix["otsu"]] = float(np.mean(values < interior_cut))

    if voxel_mm is not None:
        # v1's measures, kept because they suit already skull-stripped input: a
        # tight Otsu mask (≈ the brain when there is no skull) and the share of it
        # above each intensity percentile.
        tight = _largest_component(ndimage.binary_opening(norm > max(_otsu(norm[head]), 0.05)))
        tight = ndimage.binary_fill_holes(tight)
        tight_values = norm[tight] if tight.any() else norm[head]
        tight_count = max(float(tight.sum()), 1.0)
        feats["tight_ml"] = tight_count * voxel_size
        feats["tight_mean"] = float(tight_values.mean())
        feats["tight_std"] = float(tight_values.std())
        for p in TIGHT_PERCENTILES:
            cut = np.percentile(tight_values, p)
            feats[f"tight_ratio_p{p}"] = float((tight & (norm >= cut)).sum()) / tight_count
        for d in VOLUME_DEPTHS_MM:
            region = depth > d
            feats[f"head_ml_at_{d}mm"] = float(region.sum()) * voxel_size
            region_stats(region, {"dark": "dark", "suffix": f"_at_{d}mm", "otsu": f"otsu_dark_at_{d}mm"})

    for q in RELATIVE_DEPTHS:
        region = depth > q * max_depth
        feats[f"rel_size_q{int(q * 100)}"] = float(region.sum()) / head_count
        region_stats(region, {"dark": "rel_dark", "suffix": f"_q{int(q * 100)}", "otsu": f"rel_otsu_dark_q{int(q * 100)}"})

    # Intensity histogram of the brain-ish region, relative to tissue median.
    brainish = depth > 0.15 * max_depth
    rel = np.clip(norm[brainish] / interior_median, 0, 1.5)
    hist, _ = np.histogram(rel, bins=HIST_BINS, range=(0, 1.5))
    hist = hist / max(hist.sum(), 1)
    for i, h in enumerate(hist):
        feats[f"hist_{i}"] = float(h)

    # Contrast sanity: in T1, the deep interior (white matter) is brighter than
    # the overall head. If it is much darker, this is likely T2/FLAIR or not MRI.
    if interior_median < 0.15:
        warnings.append(
            "Image contrast doesn't look T1-weighted; the extractor is calibrated on "
            "T1 structural MRI, so treat the result with caution."
        )
    return feats, warnings


def _normalise(array: np.ndarray) -> np.ndarray:
    data = np.nan_to_num(np.asarray(array, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if data.size == 0:
        raise MriFeatureExtractionError("The scan contains no image data.")
    low, high = np.percentile(data, [0.5, 99.5])
    if high - low <= 1e-6:
        raise MriFeatureExtractionError("The scan is blank (no intensity variation).")
    return np.clip((data - low) / (high - low), 0.0, 1.0)


def _head_mask(norm: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    foreground = norm[norm > 0.02]
    threshold = max(_otsu(foreground) * 0.5, 0.04) if foreground.size else 0.05
    mask = norm > threshold
    mask = ndimage.binary_opening(mask, iterations=1)
    mask = _largest_component(mask)
    mask = ndimage.binary_closing(mask, iterations=2)
    if mask.ndim == 3:
        # Fill per axial slice too: an open neck/skull base defeats 3-D filling.
        mask = ndimage.binary_fill_holes(mask)
        for z in range(mask.shape[2]):
            mask[:, :, z] = ndimage.binary_fill_holes(mask[:, :, z])
    else:
        mask = ndimage.binary_fill_holes(mask)
    return mask


def _largest_component(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    labels, count = ndimage.label(mask)
    if count <= 1:
        return mask
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    return labels == (int(np.argmax(sizes)) + 1)


def _otsu(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32).ravel()
    if values.size == 0:
        return 0.5
    hist, edges = np.histogram(values, bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2
    weight_bg = np.cumsum(hist)
    weight_fg = weight_bg[-1] - weight_bg
    mean_cum = np.cumsum(hist * centers)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_bg = mean_cum / weight_bg
        mean_fg = (mean_cum[-1] - mean_cum) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between = np.nan_to_num(between)
    return float(edges[int(np.argmax(between)) + 1])


def _resample(volume: np.ndarray, spacing: tuple[float, ...]) -> np.ndarray:
    from scipy import ndimage

    zoom = [max(float(s), 0.1) / TARGET_SPACING_MM for s in spacing[:3]]
    out_shape = [int(round(n * z)) for n, z in zip(volume.shape, zoom)]
    if min(out_shape) < 8:
        raise MriFeatureExtractionError(
            "The volume has too few slices to process; upload the full 3-D scan."
        )
    return ndimage.zoom(np.asarray(volume, dtype=np.float32), zoom, order=1)


# --------------------------------------------------------------------------- #
# calibration model
# --------------------------------------------------------------------------- #
class GatedRegressor:
    """Route skull-stripped and skull-on inputs to separately fitted regressors.

    One regressor over everything is dominated by raw scans (most of the
    training data), which costs accuracy on skull-stripped input. A logistic
    gate estimates P(skull-stripped) from the same features, and the
    prediction is that probability's blend of the two regressors. Soft
    blending means a borderline input is never forced down the wrong branch.
    Defined here (not in the calibration script) so the pickled artifact can
    be loaded by the server.
    """

    def __init__(self, make_regressor, stripped_columns=None):
        self.make_regressor = make_regressor
        # The stripped branch has far fewer training scans, so it can be given a
        # smaller feature subset (column indices) to keep it from overfitting.
        self.stripped_columns = None if stripped_columns is None else list(stripped_columns)

    def _stripped_view(self, X):
        return X if self.stripped_columns is None else X[:, self.stripped_columns]

    def fit(self, X, y, stripped):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        stripped = np.asarray(stripped, dtype=bool)
        self.gate_ = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
        self.gate_.fit(X, stripped)
        self.stripped_ = self.make_regressor().fit(self._stripped_view(X[stripped]), y[stripped])
        self.skull_ = self.make_regressor().fit(X[~stripped], y[~stripped])
        self.make_regressor = None  # lambdas don't pickle; not needed after fit
        return self

    def stripped_probability(self, X):
        return self.gate_.predict_proba(np.asarray(X, dtype=float))[:, 1]

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        p = self.stripped_probability(X)
        return p * self.stripped_.predict(self._stripped_view(X)) + (1 - p) * self.skull_.predict(X)


class AveragedRegressor:
    """Mean of already-fitted regressors.

    Ridge (gated) and gradient boosting make different mistakes: the linear
    model extrapolates smoothly, the trees capture interactions. Their average
    beat either alone on every input type in subject-grouped CV.
    """

    def __init__(self, members):
        self.members = list(members)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.mean([member.predict(X) for member in self.members], axis=0)

    def stripped_probability(self, X):
        for member in self.members:
            if hasattr(member, "stripped_probability"):
                return member.stripped_probability(X)
        raise AttributeError("no gated member")


# --------------------------------------------------------------------------- #
# calibration artifact
# --------------------------------------------------------------------------- #
def _calibration_dir() -> Path:
    configured = Path(settings.MODEL_DIR)
    if (configured / CALIBRATION_PICKLE).exists():
        return configured
    return Path(__file__).resolve().parents[3] / "results"


@lru_cache(maxsize=1)
def _load_calibration() -> dict | None:
    path = _calibration_dir() / CALIBRATION_PICKLE
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:  # noqa: BLE001 - a broken artifact degrades, never crashes
        logger.exception("Could not load %s", path)
        return None
    if not isinstance(payload, dict) or payload.get("version") != CALIBRATION_VERSION:
        logger.warning("%s is not a v%s calibration; ignoring it", path, CALIBRATION_VERSION)
        return None
    return payload


def calibration_info() -> dict | None:
    """Summary of the loaded calibration (for /health and the model-info card)."""
    payload = _load_calibration()
    if payload is None:
        return None
    return {
        kind: payload[kind].get("stats", {})
        for kind in ("volume", "slice")
        if kind in payload
    }


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


__all__ = [
    "MRI_FEATURE_ORDER",
    "MriExtraction",
    "MriFeatureExtractionError",
    "MriFeatureExtractionNotReady",
    "calibration_info",
    "compute_features",
    "extract_from_files",
    "extract_from_scan",
    "feature_names",
    "validate_mri_feature_vector",
]
