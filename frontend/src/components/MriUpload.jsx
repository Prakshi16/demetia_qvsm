import { useState } from "react";
import { TOKEN_KEY } from "../api/client";

const STATUS = {
  IDLE: "idle",
  UPLOADING: "uploading",
  PROCESSING: "processing",
  DONE: "done",
};

const ACCEPTED_EXTENSIONS = [".nii", ".nii.gz", ".dcm", ".dicom", ".mgh", ".mgz"];
const MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024;

function hasAcceptedExtension(fileName) {
  const normalizedFileName = fileName.toLowerCase();
  return ACCEPTED_EXTENSIONS.some((extension) => normalizedFileName.endsWith(extension));
}

function formatMegabytes(bytes) {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function validateFile(file) {
  if (!file) {
    return "Select an MRI scan file.";
  }

  if (!hasAcceptedExtension(file.name)) {
    return `Unsupported file type. Use one of: ${ACCEPTED_EXTENSIONS.join(", ")}.`;
  }

  if (file.size <= 0) {
    return "The selected MRI file is empty.";
  }

  if (file.size > MAX_FILE_SIZE_BYTES) {
    return `The selected file is too large. Maximum size is ${formatMegabytes(MAX_FILE_SIZE_BYTES)}.`;
  }

  return "";
}

export default function MriUpload({ visitId, onDone }) {
  const [status, setStatus] = useState(STATUS.IDLE);
  const [fileName, setFileName] = useState("");
  const [error, setError] = useState("");
  const [isDragging, setIsDragging] = useState(false);

  const inputId = `mri-upload-${String(visitId || "new").replace(/[^a-zA-Z0-9_-]/g, "-")}`;
  const isBusy = status === STATUS.UPLOADING || status === STATUS.PROCESSING;

  const statusText = {
    [STATUS.IDLE]: "No file uploaded",
    [STATUS.UPLOADING]: "Uploading MRI scan...",
    [STATUS.PROCESSING]: "Extracting imaging features",
    [STATUS.DONE]: "Features extracted",
  }[status];

  async function uploadMriFile(file) {
    const validationMessage = validateFile(file);

    if (validationMessage) {
      setError(validationMessage);
      setStatus(STATUS.IDLE);
      setFileName("");
      return;
    }

    if (!visitId) {
      setError("A visit ID is required before uploading MRI data.");
      setStatus(STATUS.IDLE);
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setError("");
    setFileName(file.name);
    setStatus(STATUS.UPLOADING);

    try {
      const response = await fetch(`/api/v1/visits/${visitId}/mri-upload`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${localStorage.getItem(TOKEN_KEY)}`,
        },
        body: formData,
      });

      setStatus(STATUS.PROCESSING);

      const payload = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(payload.detail || "MRI upload failed.");
      }

      setStatus(STATUS.DONE);
      onDone?.(payload);
    } catch (uploadError) {
      setError(uploadError.message || "MRI upload failed. Please try again.");
      setStatus(STATUS.IDLE);
    }
  }

  function handleInputChange(event) {
    uploadMriFile(event.target.files?.[0]);
    event.target.value = "";
  }

  function handleDrop(event) {
    event.preventDefault();
    setIsDragging(false);

    if (!isBusy) {
      uploadMriFile(event.dataTransfer.files?.[0]);
    }
  }

  return (
    <section className="mri-upload" aria-busy={isBusy}>
      <style>{styles}</style>

      <input
        id={inputId}
        className="mri-upload__input"
        type="file"
        accept={ACCEPTED_EXTENSIONS.join(",")}
        onChange={handleInputChange}
        disabled={isBusy}
      />

      <label
        className={`mri-upload__dropzone${isDragging ? " mri-upload__dropzone--dragging" : ""}`}
        htmlFor={isBusy ? undefined : inputId}
        onDragEnter={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        aria-disabled={isBusy}
      >
        <span className="mri-upload__icon" aria-hidden="true">
          +
        </span>
        <span className="mri-upload__status">
          {statusText}
          {status === STATUS.DONE ? " ✓" : ""}
        </span>
        <span className="mri-upload__hint">Drag and drop an MRI scan here, or click to upload.</span>
      </label>

      {fileName && <p className="mri-upload__file">{fileName}</p>}

      {error && (
        <p className="mri-upload__error" role="alert">
          {error}
        </p>
      )}
    </section>
  );
}

const styles = `
.mri-upload {
  width: 100%;
  max-width: 34rem;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.mri-upload__input {
  display: none;
}

.mri-upload__dropzone {
  min-height: 12rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
  padding: 1.5rem;
  border: 2px dashed #64748b;
  border-radius: 8px;
  background: #f8fafc;
  color: #0f172a;
  cursor: pointer;
  text-align: center;
}

.mri-upload__dropzone:hover,
.mri-upload__dropzone--dragging {
  border-color: #2563eb;
  background: #eff6ff;
}

.mri-upload__dropzone[aria-disabled="true"] {
  cursor: wait;
  opacity: 0.75;
}

.mri-upload__icon {
  width: 2.25rem;
  height: 2.25rem;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  background: #dbeafe;
  color: #1d4ed8;
  font-size: 1.75rem;
  line-height: 1;
}

.mri-upload__status {
  font-size: 1rem;
  font-weight: 700;
}

.mri-upload__hint,
.mri-upload__file,
.mri-upload__error {
  margin: 0;
  font-size: 0.9rem;
  line-height: 1.45;
}

.mri-upload__hint,
.mri-upload__file {
  color: #475569;
}

.mri-upload__file,
.mri-upload__error {
  margin-top: 0.75rem;
}

.mri-upload__error {
  color: #b91c1c;
}
`;
