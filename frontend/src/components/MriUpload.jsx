import { useState } from "react";

// File uploads are the one deliberate exception to "everything goes through
// src/api/client.js": a JSON wrapper cannot carry multipart FormData. So this
// component keeps its raw fetch and borrows the client's two constants instead
// of hardcoding the base URL or the token key.
import { API_BASE_URL, TOKEN_KEY } from "../api/client";
import { PICKER_ACCEPT, validateFiles } from "../utils/mriFiles";

const STATUS = {
  IDLE: "idle",
  UPLOADING: "uploading",
  PROCESSING: "processing",
  DONE: "done",
};

function describeSelection(files) {
  if (files.length === 1) return files[0].name;
  return `${files.length} files (${files.slice(0, 2).map((file) => file.name).join(", ")}${files.length > 2 ? ", …" : ""})`;
}

export default function MriUpload({ visitId, onDone }) {
  const [status, setStatus] = useState(STATUS.IDLE);
  const [fileName, setFileName] = useState("");
  const [error, setError] = useState("");
  const [isDragging, setIsDragging] = useState(false);

  const inputId = `mri-upload-${String(visitId || "new").replace(/[^a-zA-Z0-9_-]/g, "-")}`;
  const isBusy = status === STATUS.UPLOADING || status === STATUS.PROCESSING;

  const statusText = {
    [STATUS.IDLE]: "Drop an MRI scan here",
    [STATUS.UPLOADING]: "Uploading…",
    [STATUS.PROCESSING]: "Reading the scan and extracting features…",
    [STATUS.DONE]: "Features extracted",
  }[status];

  async function uploadMriFiles(fileList) {
    const files = Array.from(fileList ?? []);
    const validationMessage = validateFiles(files);

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
    if (files.length === 1) {
      formData.append("file", files[0]);
    } else {
      files.forEach((file) => formData.append("files", file));
    }

    setError("");
    setFileName(describeSelection(files));
    setStatus(STATUS.UPLOADING);

    try {
      // Do NOT set Content-Type — the browser has to set the multipart
      // boundary itself, and overriding it breaks the upload.
      const request = fetch(`${API_BASE_URL}/visits/${visitId}/mri-upload`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${localStorage.getItem(TOKEN_KEY)}`,
        },
        body: formData,
      });
      // The request body is sent quickly; most of the wait is server-side
      // extraction, so switch the label once the upload is under way.
      const switchLabel = setTimeout(() => setStatus(STATUS.PROCESSING), 600);
      const response = await request;
      clearTimeout(switchLabel);

      const payload = await response.json().catch(() => ({}));

      if (!response.ok) {
        const detail = Array.isArray(payload.detail)
          ? payload.detail.map((item) => item.msg).join(", ")
          : payload.detail;
        throw new Error(detail || "MRI upload failed.");
      }

      setStatus(STATUS.DONE);
      onDone?.(payload);
    } catch (uploadError) {
      setError(
        uploadError instanceof TypeError
          ? "Cannot reach the server. Is the backend running?"
          : uploadError.message || "MRI upload failed. Please try again.",
      );
      setStatus(STATUS.IDLE);
    }
  }

  function handleInputChange(event) {
    uploadMriFiles(event.target.files);
    event.target.value = "";
  }

  function handleDrop(event) {
    event.preventDefault();
    setIsDragging(false);

    if (!isBusy) {
      uploadMriFiles(event.dataTransfer.files);
    }
  }

  return (
    <section className="mri-upload" aria-busy={isBusy}>
      <input
        id={inputId}
        className="mri-upload__input"
        type="file"
        multiple
        accept={PICKER_ACCEPT}
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
        {isBusy ? (
          <span className="spinner" aria-hidden="true" />
        ) : (
          <span className="mri-upload__icon" aria-hidden="true">
            +
          </span>
        )}
        <span className="mri-upload__status" aria-live="polite">
          {statusText}
          {status === STATUS.DONE ? " ✓" : ""}
        </span>
        <span className="mri-upload__hint">or click to choose. Several files are fine for one scan.</span>
        <span className="mri-upload__formats">
          NIfTI · MGZ · Analyze .hdr+.img · DICOM file, series or .zip · PNG/JPG slice
        </span>
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
