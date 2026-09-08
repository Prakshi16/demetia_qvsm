"""Turn an uploaded scan into something the browser viewer (Niivue) can render.

Niivue reads NIfTI natively but has no DICOM reader, so a `.dcm` upload has to
be decoded server-side. NIfTI / MGH files pass straight through (gzipped if they
aren't already). The DICOM path wraps the pixel volume in a minimal NIfTI-1
image — enough for a clinician to scroll through slices, not a substitute for a
real conversion pipeline (dcm2niix et al.).
"""
from __future__ import annotations

import gzip
import io

VIEWER_MEDIA_TYPE = "application/gzip"
VIEWER_FILENAME = "scan.nii.gz"


def to_viewer_nifti(raw: bytes, filename: str) -> bytes:
    """Return the scan as gzip-compressed NIfTI-1 bytes.

    Raises ``ValueError`` if the format can't be shown in the browser.
    """
    lower = filename.lower()

    if lower.endswith(".nii.gz"):
        return raw
    if lower.endswith(".nii"):
        return gzip.compress(raw)

    if lower.endswith((".mgz", ".mgh")):
        try:
            import nibabel as nib

            mgh = nib.MGHImage.from_bytes(gzip.decompress(raw) if lower.endswith(".mgz") else raw)
            return gzip.compress(nib.Nifti1Image(mgh.get_fdata(), mgh.affine).to_bytes())
        except Exception as exc:  # noqa: BLE001
            raise ValueError("Could not convert this MGH scan for viewing.") from exc

    if lower.endswith((".dcm", ".dicom")):
        try:
            import nibabel as nib
            import numpy as np
            import pydicom

            ds = pydicom.dcmread(io.BytesIO(raw))
            volume = np.asarray(ds.pixel_array).astype(np.float32)
            if volume.ndim == 2:
                volume = volume[:, :, None]
            spacing = [float(v) for v in getattr(ds, "PixelSpacing", [1.0, 1.0])]
            thickness = float(getattr(ds, "SliceThickness", 1.0) or 1.0)
            affine = np.diag([spacing[0], spacing[1], thickness, 1.0])
            return gzip.compress(nib.Nifti1Image(volume, affine).to_bytes())
        except Exception as exc:  # noqa: BLE001
            raise ValueError("Could not decode this DICOM file for viewing.") from exc

    raise ValueError(f"'{filename}' can't be shown in the in-browser viewer.")
