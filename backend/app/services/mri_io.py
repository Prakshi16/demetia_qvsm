"""Load any reasonable MRI upload into one in-memory representation.

Shared by the feature extractor (``mri_features.py``) and the in-browser viewer
(``scan_view.py``) so both always agree on which formats work. Everything the
upload endpoint accepts goes through ``load_scan``.

Supported inputs
----------------
3-D volumes (physical voxel size known):

* NIfTI ``.nii`` / ``.nii.gz``
* FreeSurfer ``.mgh`` / ``.mgz``
* Analyze 7.5 ``.hdr`` + ``.img`` pair (the native OASIS format). A pair can't
  travel as one file, so it arrives either as two files in one multi-file
  upload or inside a ``.zip``.
* DICOM: a single ``.dcm`` (a 2-D slice or a multi-frame volume), or a whole
  series — several ``.dcm`` files in one upload or zipped. Slices are ordered by
  their position along the slice normal, not by filename. Files inside a zip
  need no extension; DICOM is recognised by its ``DICM`` preamble.
* ``.zip`` containing any of the above.

2-D images (no physical scale):

* ``.png``, ``.jpg``/``.jpeg``, ``.bmp``, ``.gif``, ``.tif``/``.tiff``,
  ``.webp`` — e.g. a single axial slice exported from a viewer or taken from a
  public slice dataset. Colour is converted to greyscale.

Anything else raises ``ScanLoadError`` with a message fit to show a user.
"""
from __future__ import annotations

import gzip
import io
import zipfile
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import numpy as np

VOLUME_EXTENSIONS = (".nii.gz", ".nii", ".mgz", ".mgh", ".hdr", ".img")
DICOM_EXTENSIONS = (".dcm", ".dicom", ".ima")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp")
ARCHIVE_EXTENSIONS = (".zip",)

ACCEPTED_EXTENSIONS = VOLUME_EXTENSIONS + DICOM_EXTENSIONS + IMAGE_EXTENSIONS + ARCHIVE_EXTENSIONS

# Hard caps that keep a hostile or broken upload from exhausting memory: a zip
# bomb, or a "volume" whose header claims billions of voxels.
MAX_ARCHIVE_MEMBERS = 4000
MAX_UNCOMPRESSED_BYTES = 600 * 1024 * 1024
# 32 M voxels (~128 MB as float32) covers any structural MRI (1 mm 256^3 is
# 16.8 M) while keeping peak memory inside a 512 MB host (Render free tier);
# a typical raw scan peaks at ~350 MB including the loaded models.
MAX_VOXELS = 32 * 1024 * 1024


class ScanLoadError(ValueError):
    """The upload could not be read as an MRI scan. ``str(exc)`` is user-facing."""


@dataclass
class LoadedScan:
    """A decoded scan.

    ``data`` is float32, 3-D ``(x, y, z)`` for a volume or 2-D ``(rows, cols)``
    for a picture. ``spacing`` is the voxel size in mm (``None`` for a picture,
    where there is no physical scale). ``affine`` is the voxel->world matrix when
    the format records one.
    """

    data: np.ndarray
    kind: str  # "volume" | "slice"
    source_format: str
    spacing: tuple[float, ...] | None = None
    affine: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def is_volume(self) -> bool:
        return self.kind == "volume"


def lower_name(filename: str) -> str:
    return PurePosixPath((filename or "").replace("\\", "/")).name.lower()


def has_accepted_extension(filename: str) -> bool:
    return lower_name(filename).endswith(ACCEPTED_EXTENSIONS)


def load_scan(files: list[tuple[str, bytes]]) -> LoadedScan:
    """Decode one upload (one or more files) into a ``LoadedScan``."""
    if not files:
        raise ScanLoadError("No MRI file was uploaded.")

    expanded = _expand_archives(files)
    if not expanded:
        raise ScanLoadError("The archive is empty.")

    names = [lower_name(name) for name, _ in expanded]

    # 1. An Analyze pair (needs both halves).
    headers = [i for i, n in enumerate(names) if n.endswith(".hdr")]
    if headers:
        return _load_analyze_pair(expanded, headers[0])
    if any(n.endswith(".img") for n in names):
        raise ScanLoadError(
            "An Analyze .img file needs its matching .hdr header. Upload both "
            "files together (select both), or zip them."
        )

    # 2. A single NIfTI / MGH volume (first one wins if several were sent).
    for (name, raw), lowered in zip(expanded, names):
        if lowered.endswith((".nii", ".nii.gz", ".mgh", ".mgz")):
            return _load_nibabel_bytes(raw, lowered)

    # 3. DICOM — by extension or by preamble.
    dicom_members = [
        (name, raw)
        for (name, raw), lowered in zip(expanded, names)
        if lowered.endswith(DICOM_EXTENSIONS) or _looks_like_dicom(raw)
    ]
    if dicom_members:
        return _load_dicom(dicom_members)

    # 4. A plain picture.
    for (name, raw), lowered in zip(expanded, names):
        if lowered.endswith(IMAGE_EXTENSIONS):
            return _load_image(raw)

    raise ScanLoadError(
        "Unsupported MRI file type. Use NIfTI (.nii/.nii.gz), MGH (.mgz), "
        "Analyze (.hdr + .img), DICOM (.dcm, a series, or a .zip of one), or an "
        "image of a slice (.png/.jpg)."
    )


# --------------------------------------------------------------------------- #
# archives
# --------------------------------------------------------------------------- #
def _expand_archives(files: list[tuple[str, bytes]]) -> list[tuple[str, bytes]]:
    out: list[tuple[str, bytes]] = []
    total = 0
    for name, raw in files:
        if not lower_name(name).endswith(".zip"):
            out.append((name, raw))
            continue
        try:
            archive = zipfile.ZipFile(io.BytesIO(raw))
        except zipfile.BadZipFile as exc:
            raise ScanLoadError("The .zip file is damaged or not a zip archive.") from exc
        members = [m for m in archive.infolist() if not m.is_dir()]
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise ScanLoadError(f"The archive has too many files (limit {MAX_ARCHIVE_MEMBERS}).")
        for member in members:
            base = PurePosixPath(member.filename).name
            # macOS resource forks and dotfiles are never scan data.
            if not base or base.startswith(".") or "__MACOSX" in member.filename:
                continue
            total += member.file_size
            if total > MAX_UNCOMPRESSED_BYTES:
                raise ScanLoadError("The archive is too large once uncompressed.")
            out.append((member.filename, archive.read(member)))
    return out


# --------------------------------------------------------------------------- #
# NIfTI / MGH / Analyze
# --------------------------------------------------------------------------- #
def _finish_nibabel(image, source_format: str) -> LoadedScan:
    import nibabel as nib

    shape = image.shape
    if len(shape) < 2:
        raise ScanLoadError("The scan has fewer than two dimensions.")
    if int(np.prod(shape[:3])) > MAX_VOXELS:
        raise ScanLoadError("The scan is too large to process (over 32 million voxels). Upload a single structural volume, or downsample it.")

    notes: list[str] = []
    # Reorient to the closest canonical (RAS) frame so the viewer shows it the
    # right way up and slice-based steps see a consistent axis order.
    try:
        if len(shape) >= 3:
            image = nib.as_closest_canonical(image)
    except Exception:  # noqa: BLE001 - an odd affine just keeps native order
        notes.append("Could not reorient the scan; using its stored axis order.")

    data = np.asarray(image.dataobj, dtype=np.float32)
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]  # trailing singleton axis (common in Analyze files)
    if data.ndim > 3:
        notes.append(f"{data.ndim}-D series: using the first volume.")
        while data.ndim > 3:
            data = data[..., 0]
    data = np.squeeze(data) if data.ndim == 3 and 1 in data.shape else data

    zooms = tuple(float(abs(z)) or 1.0 for z in image.header.get_zooms()[: data.ndim])
    if data.ndim == 2:
        return LoadedScan(
            data=data, kind="slice", source_format=source_format,
            spacing=zooms[:2], affine=None, notes=notes + ["Single-slice file."],
        )
    return LoadedScan(
        data=data, kind="volume", source_format=source_format,
        spacing=zooms[:3], affine=np.asarray(image.affine, dtype=float), notes=notes,
    )


def _load_nibabel_bytes(raw: bytes, lowered: str) -> LoadedScan:
    import nibabel as nib

    try:
        if lowered.endswith((".mgz", ".mgh")):
            payload = gzip.decompress(raw) if lowered.endswith(".mgz") or raw[:2] == b"\x1f\x8b" else raw
            image = nib.MGHImage.from_bytes(payload)
            return _finish_nibabel(image, "mgh")
        payload = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
        try:
            image = nib.Nifti1Image.from_bytes(payload)
        except Exception:  # noqa: BLE001 - try NIfTI-2 before giving up
            image = nib.Nifti2Image.from_bytes(payload)
        return _finish_nibabel(image, "nifti")
    except ScanLoadError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ScanLoadError("This NIfTI/MGH file could not be read; it may be damaged.") from exc


def _load_analyze_pair(files: list[tuple[str, bytes]], header_index: int) -> LoadedScan:
    import nibabel as nib

    header_name, header_raw = files[header_index]
    stem = lower_name(header_name)[: -len(".hdr")]
    image_raw = None
    for name, raw in files:
        if lower_name(name) == f"{stem}.img":
            image_raw = raw
            break
    if image_raw is None:
        imgs = [raw for name, raw in files if lower_name(name).endswith(".img")]
        if len(imgs) == 1:
            image_raw = imgs[0]
    if image_raw is None:
        raise ScanLoadError(
            f"Found the header {PurePosixPath(header_name).name} but not its matching "
            ".img file. Upload both together."
        )

    try:
        file_map = {
            "header": nib.FileHolder(fileobj=io.BytesIO(header_raw)),
            "image": nib.FileHolder(fileobj=io.BytesIO(image_raw)),
        }
        # OASIS ships "Analyze" pairs that are really NIfTI-pair (magic ni1);
        # try the richer reader first so a real orientation is honoured.
        try:
            image = nib.Nifti1Pair.from_file_map(file_map)
            source = "nifti-pair"
        except Exception:  # noqa: BLE001
            file_map = {
                "header": nib.FileHolder(fileobj=io.BytesIO(header_raw)),
                "image": nib.FileHolder(fileobj=io.BytesIO(image_raw)),
            }
            image = nib.AnalyzeImage.from_file_map(file_map)
            source = "analyze"
        # Force the data into memory while the BytesIO objects are alive; the
        # affine carries the voxel size, so a fresh NIfTI header is enough.
        data = np.asarray(image.dataobj, dtype=np.float32)
        image = nib.Nifti1Image(data, image.affine)
        return _finish_nibabel(image, source)
    except ScanLoadError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ScanLoadError("This Analyze .hdr/.img pair could not be read.") from exc


# --------------------------------------------------------------------------- #
# DICOM
# --------------------------------------------------------------------------- #
def _looks_like_dicom(raw: bytes) -> bool:
    return len(raw) > 132 and raw[128:132] == b"DICM"


def _dicom_pixels(ds) -> np.ndarray:
    pixels = np.asarray(ds.pixel_array, dtype=np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    if pixels.ndim == 3 and getattr(ds, "SamplesPerPixel", 1) == 3:
        pixels = pixels.mean(axis=-1)  # an RGB secondary capture
    return pixels * slope + intercept


def _load_dicom(members: list[tuple[str, bytes]]) -> LoadedScan:
    import pydicom

    datasets = []
    for name, raw in members:
        try:
            ds = pydicom.dcmread(io.BytesIO(raw), force=True)
        except Exception:  # noqa: BLE001 - skip junk (DICOMDIR, thumbnails, ...)
            continue
        if "PixelData" not in ds:
            continue
        datasets.append(ds)
    if not datasets:
        raise ScanLoadError("No readable DICOM images were found in the upload.")

    # Several series zipped together: take the one with the most slices (the
    # main 3-D acquisition, not a localiser).
    by_series: dict[str, list] = {}
    for ds in datasets:
        by_series.setdefault(str(getattr(ds, "SeriesInstanceUID", "single")), []).append(ds)
    series = max(by_series.values(), key=len)
    notes: list[str] = []
    if len(by_series) > 1:
        notes.append(f"{len(by_series)} DICOM series found; using the largest ({len(series)} images).")

    first = series[0]
    try:
        row_mm, col_mm = (float(v) for v in getattr(first, "PixelSpacing", [1.0, 1.0]))
    except Exception:  # noqa: BLE001
        row_mm, col_mm = 1.0, 1.0

    try:
        if len(series) == 1:
            pixels = _dicom_pixels(first)
            if pixels.ndim == 2:
                # Slices are always (rows, cols), like a picture.
                return LoadedScan(
                    data=pixels.copy(), kind="slice", source_format="dicom",
                    spacing=(row_mm, col_mm), notes=notes + ["Single DICOM slice."],
                )
            # Multi-frame: frames along axis 0.
            thickness = float(getattr(first, "SpacingBetweenSlices", 0) or getattr(first, "SliceThickness", 1.0) or 1.0)
            volume = np.transpose(pixels, (2, 1, 0))
            return LoadedScan(
                data=volume.astype(np.float32), kind="volume", source_format="dicom",
                spacing=(col_mm, row_mm, thickness), notes=notes,
            )

        # A series of 2-D slices: order by position along the slice normal.
        def position(ds) -> float:
            try:
                orient = np.asarray(ds.ImageOrientationPatient, dtype=float)
                normal = np.cross(orient[:3], orient[3:])
                return float(np.dot(normal, np.asarray(ds.ImagePositionPatient, dtype=float)))
            except Exception:  # noqa: BLE001
                return float(getattr(ds, "InstanceNumber", 0) or 0)

        series = sorted(series, key=position)
        slices = [_dicom_pixels(ds) for ds in series]
        shapes = {s.shape for s in slices}
        if len(shapes) != 1:
            common = max(shapes, key=lambda s: sum(x.shape == s for x in slices))
            slices = [s for s in slices if s.shape == common]
            notes.append("Dropped DICOM images whose size didn't match the series.")
        positions = [position(ds) for ds in series][: len(slices)]
        gaps = np.diff(positions)
        gaps = gaps[np.abs(gaps) > 1e-3]
        thickness = float(np.median(np.abs(gaps))) if gaps.size else float(getattr(first, "SliceThickness", 1.0) or 1.0)

        volume = np.stack([s.T for s in slices], axis=-1)  # (cols, rows, slices)
        if int(np.prod(volume.shape)) > MAX_VOXELS:
            raise ScanLoadError("The DICOM series is too large to process.")
        return LoadedScan(
            data=volume.astype(np.float32), kind="volume", source_format="dicom-series",
            spacing=(col_mm, row_mm, thickness), notes=notes,
        )
    except ScanLoadError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ScanLoadError(
            "The DICOM pixel data could not be decoded (it may use a compression "
            "this server can't read). Export the scan as NIfTI and upload that."
        ) from exc


# --------------------------------------------------------------------------- #
# plain images
# --------------------------------------------------------------------------- #
def _load_image(raw: bytes) -> LoadedScan:
    try:
        from PIL import Image, ImageSequence
    except ImportError as exc:  # pragma: no cover - Pillow is in requirements
        raise ScanLoadError("Image uploads need the Pillow package on the server.") from exc

    try:
        image = Image.open(io.BytesIO(raw))
        frames = [frame.convert("L") for frame in ImageSequence.Iterator(image)]
    except Exception as exc:  # noqa: BLE001
        raise ScanLoadError("This image file could not be read.") from exc

    if not frames:
        raise ScanLoadError("The image has no frames.")

    arrays = [np.asarray(frame, dtype=np.float32) for frame in frames]
    if len(arrays) > 8 and len({a.shape for a in arrays}) == 1:
        # A multi-page TIFF / animated GIF of a full stack: treat as a volume
        # with unknown (assumed 1 mm) spacing.
        volume = np.stack([a.T for a in arrays], axis=-1)
        return LoadedScan(
            data=volume, kind="volume", source_format="image-stack",
            spacing=(1.0, 1.0, 1.0),
            notes=["Multi-page image treated as a slice stack; voxel size assumed 1 mm."],
        )
    data = arrays[0]
    if data.shape[0] < 32 or data.shape[1] < 32:
        raise ScanLoadError("The image is too small to be an MRI slice.")
    return LoadedScan(data=data, kind="slice", source_format="image", spacing=None)


__all__ = [
    "ACCEPTED_EXTENSIONS",
    "LoadedScan",
    "ScanLoadError",
    "has_accepted_extension",
    "load_scan",
    "lower_name",
]
