"""Turn a stored MRI upload into something the browser viewer (Niivue) can render.

Niivue reads NIfTI natively but has no DICOM, Analyze-pair or zip reader, so
the scan is decoded server-side with the same loader the feature extractor uses
(``mri_io.load_scan``) and re-encoded as gzip-compressed NIfTI-1. Whatever was
accepted for feature extraction can therefore also be viewed. A NIfTI upload is
passed through untouched (it's already what Niivue wants).

A 2-D picture becomes a one-slice volume, flipped so it shows upright in
Niivue's axial view.
"""
from __future__ import annotations

import gzip

import numpy as np

from app.services.mri_io import ScanLoadError, load_scan, lower_name

VIEWER_MEDIA_TYPE = "application/gzip"
VIEWER_FILENAME = "scan.nii.gz"


def to_viewer_nifti(raw: bytes, filename: str) -> bytes:
    """Return the scan as gzip-compressed NIfTI-1 bytes.

    Raises ``ValueError`` if the stored object can't be decoded.
    """
    import nibabel as nib

    lowered = lower_name(filename)
    if lowered.endswith(".nii.gz") and raw[:2] == b"\x1f\x8b":
        return raw
    if lowered.endswith(".nii"):
        return gzip.compress(raw)

    try:
        scan = load_scan([(filename, raw)])
    except ScanLoadError as exc:
        raise ValueError(str(exc)) from exc

    if scan.is_volume:
        if scan.affine is not None:
            affine = scan.affine
        else:
            affine = np.diag([*(scan.spacing or (1.0, 1.0, 1.0)), 1.0])
        image = nib.Nifti1Image(scan.data.astype(np.float32), affine)
    else:
        # (rows, cols) picture -> (x, y, 1) with y pointing up.
        data = np.flipud(scan.data).T[:, :, None].astype(np.float32)
        spacing = scan.spacing or (1.0, 1.0)
        image = nib.Nifti1Image(data, np.diag([spacing[1], spacing[0], 1.0, 1.0]))
    return gzip.compress(image.to_bytes())
