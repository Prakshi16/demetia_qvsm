/**
 * Speech upload card for the New Visit — Screening page (§6 screen 5).
 *
 * TEMPORARY, and deliberately minimal: Sheetal owns <SpeechCapture />, the
 * richer component with a record button, live waveform and timer (MediaRecorder
 * API), with file upload only as its fallback. That component does not exist
 * yet and screen 5 cannot be tested without *some* speech card, so this is the
 * fallback half on its own.
 *
 * It takes the same props and calls the same endpoint, so when <SpeechCapture />
 * lands it replaces this as a one-line import swap in NewVisitScreening.jsx —
 * not a merge conflict. Delete this file at that point.
 *
 * Structure mirrors MriUpload.jsx on purpose: both upload cards should look and
 * behave the same, the same way both upload endpoints mirror each other.
 */
import { useState } from "react";

// File uploads are the one deliberate exception to "everything goes through
// src/api/client.js": a JSON wrapper cannot carry multipart FormData. So this
// component uses a raw fetch and borrows the client's two constants instead of
// hardcoding the base URL or the token key.
import { API_BASE_URL, TOKEN_KEY } from "../api/client";

const STATUS = {
  IDLE: "idle",
  UPLOADING: "uploading",
  PROCESSING: "processing",
  DONE: "done",
};

// The backend allowlist is .wav/.mp3 only (Product Rule 11) — anything else is
// rejected server-side before extraction runs, so reject it here first.
const ACCEPTED_EXTENSIONS = [".wav", ".mp3"];

// 50 MB — the Supabase free-tier per-file storage limit, matching the backend.
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
    return "Select an audio recording.";
  }

  if (!hasAcceptedExtension(file.name)) {
    return `Unsupported file type. Use one of: ${ACCEPTED_EXTENSIONS.join(", ")}.`;
  }

  if (file.size <= 0) {
    return "The selected audio file is empty.";
  }

  if (file.size > MAX_FILE_SIZE_BYTES) {
    return `The selected file is too large. Maximum size is ${formatMegabytes(MAX_FILE_SIZE_BYTES)}.`;
  }

  return "";
}

export default function SpeechUpload({ visitId, onDone }) {
  const [status, setStatus] = useState(STATUS.IDLE);
  const [fileName, setFileName] = useState("");
  const [error, setError] = useState("");
  const [isDragging, setIsDragging] = useState(false);

  const inputId = `speech-upload-${String(visitId || "new").replace(/[^a-zA-Z0-9_-]/g, "-")}`;
  const isBusy = status === STATUS.UPLOADING || status === STATUS.PROCESSING;

  const statusText = {
    [STATUS.IDLE]: "No recording uploaded",
    [STATUS.UPLOADING]: "Uploading recording...",
    [STATUS.PROCESSING]: "Extracting speech features",
    [STATUS.DONE]: "Features extracted",
  }[status];

  async function uploadSpeechFile(file) {
    const validationMessage = validateFile(file);

    if (validationMessage) {
      setError(validationMessage);
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setError("");
    setFileName(file.name);
    setStatus(STATUS.UPLOADING);

    try {
      // Do NOT set Content-Type — the browser has to set the multipart
      // boundary itself, and overriding it breaks the upload.
      const response = await fetch(`${API_BASE_URL}/visits/${visitId}/speech-upload`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${localStorage.getItem(TOKEN_KEY)}`,
        },
        body: formData,
      });

      setStatus(STATUS.PROCESSING);

      const payload = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          typeof payload.detail === "string" ? payload.detail : "Speech upload failed.",
        );
      }

      setStatus(STATUS.DONE);
      // The endpoint returns the whole visit (VisitDetailOut), so the parent can
      // read mri_status/speech_status/model_prediction straight off this.
      onDone?.(payload);
    } catch (uploadError) {
      setError(uploadError.message || "Speech upload failed. Please try again.");
      setStatus(STATUS.IDLE);
    }
  }

  function handleInputChange(event) {
    uploadSpeechFile(event.target.files?.[0]);
    event.target.value = "";
  }

  function handleDrop(event) {
    event.preventDefault();
    setIsDragging(false);

    if (!isBusy) {
      uploadSpeechFile(event.dataTransfer.files?.[0]);
    }
  }

  return (
    <section className="speech-upload" aria-busy={isBusy}>
      <style>{styles}</style>

      <input
        id={inputId}
        className="speech-upload__input"
        type="file"
        accept={ACCEPTED_EXTENSIONS.join(",")}
        onChange={handleInputChange}
        disabled={isBusy}
      />

      <label
        className={`speech-upload__dropzone${isDragging ? " speech-upload__dropzone--dragging" : ""}`}
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
        <span className="speech-upload__icon" aria-hidden="true">
          ♪
        </span>
        <span className="speech-upload__status">
          {statusText}
          {status === STATUS.DONE ? " ✓" : ""}
        </span>
        <span className="speech-upload__hint">
          Drag and drop a .wav or .mp3 recording here, or click to upload.
        </span>
      </label>

      {fileName && <p className="speech-upload__file">{fileName}</p>}

      {error && (
        <p className="speech-upload__error" role="alert">
          {error}
        </p>
      )}
    </section>
  );
}

const styles = `
.speech-upload {
  width: 100%;
  font-family: inherit;
}

.speech-upload__input {
  display: none;
}

.speech-upload__dropzone {
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

.speech-upload__dropzone:hover,
.speech-upload__dropzone--dragging {
  border-color: #2563eb;
  background: #eff6ff;
}

.speech-upload__dropzone[aria-disabled="true"] {
  cursor: wait;
  opacity: 0.75;
}

.speech-upload__icon {
  width: 2.25rem;
  height: 2.25rem;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  background: #dbeafe;
  color: #1d4ed8;
  font-size: 1.5rem;
  line-height: 1;
}

.speech-upload__status {
  font-size: 1rem;
  font-weight: 700;
}

.speech-upload__hint {
  font-size: 0.875rem;
  color: #475569;
}

.speech-upload__file {
  margin: 0.75rem 0 0;
  font-size: 0.875rem;
  color: #334155;
  word-break: break-all;
}

.speech-upload__error {
  margin: 0.5rem 0 0;
  font-size: 0.875rem;
  color: #b91c1c;
}
`;
