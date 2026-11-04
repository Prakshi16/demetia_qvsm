/**
 * MRI upload rules shared by the upload component and its tests.
 */
// Mirrors backend/app/services/mri_io.py. One file of any of these, or several
// files that together form one scan (an Analyze .hdr + .img pair, or the
// slices of a DICOM series, which may have no extension at all).
export const ACCEPTED_EXTENSIONS = [
  ".nii", ".nii.gz", ".mgh", ".mgz", ".hdr", ".img",
  ".dcm", ".dicom", ".ima", ".zip",
  ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp",
];
// The native picker matches `accept` against the text after the LAST dot, so
// ".nii.gz" alone never matches — list ".gz" too. validateFiles() still checks
// the full compound list.
export const PICKER_ACCEPT = [...ACCEPTED_EXTENSIONS, ".gz", "application/gzip", "application/dicom", "image/*"].join(",");
// 50 MB in total — the Supabase free-tier per-object limit (the backend stores
// one object per scan) and the backend's own cap.
export const MAX_TOTAL_BYTES = 50 * 1024 * 1024;

function hasAcceptedExtension(fileName) {
  const lower = fileName.toLowerCase();
  return ACCEPTED_EXTENSIONS.some((extension) => lower.endsWith(extension));
}

function formatMegabytes(bytes) {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function validateFiles(files) {
  if (!files || files.length === 0) return "Select an MRI scan file.";

  const total = files.reduce((sum, file) => sum + file.size, 0);
  if (files.every((file) => file.size <= 0)) return "The selected MRI file is empty.";
  if (total > MAX_TOTAL_BYTES) {
    return `The upload is ${formatMegabytes(total)}; the limit is ${formatMegabytes(MAX_TOTAL_BYTES)}. Use a compressed .nii.gz or zip the series.`;
  }

  if (files.length === 1) {
    const [file] = files;
    if (!hasAcceptedExtension(file.name)) {
      return "Unsupported file type. Use NIfTI (.nii/.nii.gz), MGH (.mgz), Analyze (.hdr + .img), DICOM (.dcm or a .zip), or an image of a slice (.png/.jpg).";
    }
    if (file.name.toLowerCase().endsWith(".img") || file.name.toLowerCase().endsWith(".hdr")) {
      return "An Analyze scan is two files: select the .hdr and the .img together.";
    }
  }
  return "";
}

