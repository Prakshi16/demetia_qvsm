/**
 * How an MRI upload was read: what kind of input it was, the two values it
 * produced, their validated typical error, and any caveat (e.g. a 2-D slice
 * can't give eTIV, so the training median was used). Data comes from
 * visit.mri_extraction (migration 006); older visits have none and the note
 * shows just the values.
 */
const FORMAT_LABELS = {
  nifti: "NIfTI",
  "nifti-pair": "NIfTI pair (.hdr/.img)",
  analyze: "Analyze (.hdr/.img)",
  mgh: "MGH/MGZ",
  dicom: "DICOM",
  "dicom-series": "DICOM series",
  image: "image",
  "image-stack": "image stack",
};

export default function MriExtractionNote({ visit }) {
  const vector = visit?.mri_feature_vector;
  if (!Array.isArray(vector) || vector.length !== 2) return null;
  const meta = visit.mri_extraction ?? null;
  const [nwbv, etiv] = vector;

  const kind = meta?.input_kind === "slice" ? "2-D slice" : meta ? "3-D volume" : null;
  const format = meta ? FORMAT_LABELS[meta.source_format] ?? meta.source_format : null;
  const nwbvError = meta?.expected_error?.nwbv?.mae;
  const etivError = meta?.expected_error?.etiv?.mae;
  const warnings = meta?.warnings ?? [];

  return (
    <div className="mri-meta">
      <span>
        <strong>nWBV {Number(nwbv).toFixed(3)}</strong>
        {nwbvError ? ` (typically ±${nwbvError.toFixed(3)})` : ""} ·{" "}
        <strong>eTIV {Math.round(etiv)} mL</strong>
        {meta && !meta.etiv_measured
          ? " (not measurable from a slice: training median used)"
          : etivError
            ? ` (typically ±${Math.round(etivError)} mL)`
            : ""}
      </span>
      {kind ? (
        <span>
          Read as a {kind}
          {format ? ` (${format})` : ""}
          {meta.method === "calibrated" ? ", calibrated extractor" : ", uncalibrated estimate"}
          {meta.skull_stripped_probability != null
            ? meta.skull_stripped_probability > 0.5
              ? "; looks skull-stripped"
              : "; skull present"
            : ""}
          .
        </span>
      ) : null}
      {warnings.length > 0 ? (
        <div className="notice notice--warning" role="note">
          <ul style={{ margin: 0, paddingLeft: "1rem" }}>
            {warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}
