import { useEffect, useRef, useState } from "react";

// File uploads are the one deliberate exception to "everything goes through
// src/api/client.js": a JSON wrapper cannot carry multipart FormData. So this
// component keeps its raw fetch and borrows the client's two constants instead
// of hardcoding the base URL or the token key.
import { API_BASE_URL, TOKEN_KEY } from "../api/client";

const STATUS = {
  IDLE: "idle",
  RECORDING: "recording",
  PROCESSING: "processing",
  DONE: "done",
};

const ACCEPTED_EXTENSIONS = [".wav", ".mp3", ".m4a", ".webm"];

function hasAcceptedExtension(fileName) {
  const name = fileName.toLowerCase();
  return ACCEPTED_EXTENSIONS.some((extension) => name.endsWith(extension));
}

function validateFile(file) {
  if (!file) {
    return "Select a speech recording.";
  }

  if (!hasAcceptedExtension(file.name)) {
    return `Unsupported file type. Use one of: ${ACCEPTED_EXTENSIONS.join(", ")}.`;
  }

  if (file.size <= 0) {
    return "The selected audio file is empty.";
  }

  return "";
}

export default function SpeechCapture({ visitId, onDone }) {
  const [status, setStatus] = useState(STATUS.IDLE);
  const [timer, setTimer] = useState(0);
  const [error, setError] = useState("");
  const [fileName, setFileName] = useState("");
  const [waveform, setWaveform] = useState(
    Array.from({ length: 32 }, () => 8)
  );

  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const animationRef = useRef(null);

  const inputId = `speech-upload-${String(visitId || "new").replace(
    /[^a-zA-Z0-9_-]/g,
    "-"
  )}`;

  const isBusy =
    status === STATUS.RECORDING || status === STATUS.PROCESSING;

  useEffect(() => {
    return () => {
      stopTracks();
      clearInterval(timerRef.current);
      cancelAnimationFrame(animationRef.current);

      if (audioContextRef.current) {
        audioContextRef.current.close().catch(() => {});
      }
    };
  }, []);

  function stopTracks() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }

  function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;

    return `${String(minutes).padStart(2, "0")}:${String(
      remainingSeconds
    ).padStart(2, "0")}`;
  }

  function startTimer() {
    setTimer(0);

    timerRef.current = setInterval(() => {
      setTimer((current) => current + 1);
    }, 1000);
  }

  function stopTimer() {
    clearInterval(timerRef.current);
    timerRef.current = null;
  }

  function startWaveform(stream) {
    try {
      const AudioContext =
        window.AudioContext || window.webkitAudioContext;

      if (!AudioContext) {
        return;
      }

      const audioContext = new AudioContext();
      const analyser = audioContext.createAnalyser();

      analyser.fftSize = 64;

      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);

      audioContextRef.current = audioContext;
      analyserRef.current = analyser;

      const data = new Uint8Array(analyser.frequencyBinCount);

      function draw() {
        if (!analyserRef.current) {
          return;
        }

        analyser.getByteFrequencyData(data);

        const bars = Array.from({ length: 32 }, (_, index) => {
          const value = data[index % data.length] || 0;
          return Math.max(6, Math.round(value / 4));
        });

        setWaveform(bars);
        animationRef.current = requestAnimationFrame(draw);
      }

      draw();
    } catch {
      // Waveform is visual feedback only.
    }
  }

  function stopWaveform() {
    cancelAnimationFrame(animationRef.current);
    animationRef.current = null;

    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }

    analyserRef.current = null;
    setWaveform(Array.from({ length: 32 }, () => 8));
  }

  async function uploadAudio(file) {
    const validationMessage = validateFile(file);

    if (validationMessage) {
      setError(validationMessage);
      setStatus(STATUS.IDLE);
      return;
    }

    if (!visitId) {
      setError("A visit ID is required before uploading speech data.");
      setStatus(STATUS.IDLE);
      return;
    }

    const token = localStorage.getItem(TOKEN_KEY);

    if (!token) {
      setError("Authentication token not found. Please log in again.");
      setStatus(STATUS.IDLE);
      return;
    }

    const formData = new FormData();

    // IMPORTANT:
    // Do not manually set Content-Type.
    // Browser automatically adds multipart/form-data boundary.
    formData.append("file", file);

    setError("");
    setFileName(file.name);
    setStatus(STATUS.PROCESSING);

    try {
      const response = await fetch(
        `${API_BASE_URL}/visits/${visitId}/speech-upload`,
        {
          method: "POST",
          headers: {
            Authorization: "Bearer " + token,
          },
          body: formData,
        }
      );

      const payload = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(payload.detail || "Speech upload failed.");
      }

      setStatus(STATUS.DONE);
      onDone?.(payload);
    } catch (uploadError) {
      setError(
        uploadError.message || "Speech upload failed. Please try again."
      );
      setStatus(STATUS.IDLE);
    }
  }

  async function startRecording() {
    setError("");

    if (!visitId) {
      setError("A visit ID is required before recording speech.");
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Audio recording is not supported by this browser.");
      return;
    }

    if (!window.MediaRecorder) {
      setError("MediaRecorder is not supported by this browser.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });

      streamRef.current = stream;
      chunksRef.current = [];

      const mimeTypes = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/mp4",
      ];

      const supportedType = mimeTypes.find((type) =>
        MediaRecorder.isTypeSupported(type)
      );

      const recorder = supportedType
        ? new MediaRecorder(stream, { mimeType: supportedType })
        : new MediaRecorder(stream);

      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        stopTimer();
        stopWaveform();
        stopTracks();

        const mimeType = recorder.mimeType || "audio/webm";

        const extension = mimeType.includes("mp4") ? "m4a" : "webm";

        const audioBlob = new Blob(chunksRef.current, {
          type: mimeType,
        });

        const audioFile = new File(
          [audioBlob],
          `speech-${Date.now()}.${extension}`,
          {
            type: mimeType,
          }
        );

        await uploadAudio(audioFile);
      };

      recorder.onerror = () => {
        setError("Recording failed. Please try again.");
        stopTimer();
        stopWaveform();
        stopTracks();
        setStatus(STATUS.IDLE);
      };

      recorder.start();

      setStatus(STATUS.RECORDING);
      setTimer(0);
      startTimer();
      startWaveform(stream);
    } catch (recordingError) {
      stopTracks();

      if (recordingError.name === "NotAllowedError") {
        setError("Microphone permission was denied.");
      } else {
        setError(
          recordingError.message || "Could not start recording."
        );
      }

      setStatus(STATUS.IDLE);
    }
  }

  function stopRecording() {
    const recorder = mediaRecorderRef.current;

    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
    }
  }

  function handleFileChange(event) {
    const file = event.target.files?.[0];

    if (file) {
      uploadAudio(file);
    }

    event.target.value = "";
  }

  function reset() {
    stopTimer();
    stopWaveform();
    stopTracks();

    chunksRef.current = [];
    mediaRecorderRef.current = null;

    setStatus(STATUS.IDLE);
    setTimer(0);
    setFileName("");
    setError("");
  }

  return (
    <section className="speech-capture" aria-busy={isBusy}>
      <style>{styles}</style>

      <div className="speech-capture__card">
        <div className="speech-capture__header">
          <h3>Speech Recording</h3>

          <span className={`speech-capture__status speech-capture__status--${status}`}>
            {status}
          </span>
        </div>

        {status === STATUS.IDLE && (
          <>
            <p className="speech-capture__hint">
              Record a speech sample or upload an existing audio file.
            </p>

            <button
              type="button"
              className="speech-capture__record-button"
              onClick={startRecording}
            >
              ● Start Recording
            </button>

            <div className="speech-capture__or">OR</div>

            <label
              className="speech-capture__upload"
              htmlFor={inputId}
            >
              Upload Audio File
            </label>

            <input
              id={inputId}
              type="file"
              accept={ACCEPTED_EXTENSIONS.join(",")}
              onChange={handleFileChange}
              hidden
            />
          </>
        )}

        {status === STATUS.RECORDING && (
          <>
            <div className="speech-capture__timer">
              {formatTime(timer)}
            </div>

            <div
              className="speech-capture__waveform"
              aria-label="Live audio waveform"
            >
              {waveform.map((height, index) => (
                <span
                  key={index}
                  style={{ height: `${height}px` }}
                />
              ))}
            </div>

            <button
              type="button"
              className="speech-capture__stop-button"
              onClick={stopRecording}
            >
              ■ Stop Recording
            </button>
          </>
        )}

        {status === STATUS.PROCESSING && (
          <div className="speech-capture__processing">
            <div className="speech-capture__spinner" />
            <p>Uploading and processing speech...</p>
            {fileName && <small>{fileName}</small>}
          </div>
        )}

        {status === STATUS.DONE && (
          <div className="speech-capture__done">
            <div className="speech-capture__check">✓</div>
            <strong>Speech uploaded successfully</strong>

            {fileName && <p>{fileName}</p>}

            <button
              type="button"
              className="speech-capture__reset"
              onClick={reset}
            >
              Record Another
            </button>
          </div>
        )}

        {error && (
          <p className="speech-capture__error" role="alert">
            {error}
          </p>
        )}
      </div>
    </section>
  );
}

const styles = `
.speech-capture {
  width: 100%;
  max-width: 34rem;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.speech-capture__card {
  padding: 1.5rem;
  border: 1px solid #cbd5e1;
  border-radius: 12px;
  background: #f8fafc;
}

.speech-capture__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1rem;
}

.speech-capture__header h3 {
  margin: 0;
  color: #0f172a;
}

.speech-capture__status {
  padding: 0.25rem 0.6rem;
  border-radius: 999px;
  background: #e2e8f0;
  color: #475569;
  font-size: 0.75rem;
  font-weight: 700;
  text-transform: uppercase;
}

.speech-capture__hint {
  color: #475569;
  text-align: center;
}

.speech-capture__record-button,
.speech-capture__stop-button,
.speech-capture__upload,
.speech-capture__reset {
  width: 100%;
  padding: 0.8rem 1rem;
  border: 0;
  border-radius: 8px;
  font-size: 0.95rem;
  font-weight: 700;
  cursor: pointer;
  text-align: center;
  box-sizing: border-box;
}

.speech-capture__record-button {
  background: #2563eb;
  color: white;
}

.speech-capture__record-button:hover {
  background: #1d4ed8;
}

.speech-capture__stop-button {
  background: #dc2626;
  color: white;
}

.speech-capture__stop-button:hover {
  background: #b91c1c;
}

.speech-capture__upload {
  display: block;
  background: #e2e8f0;
  color: #0f172a;
}

.speech-capture__upload:hover {
  background: #cbd5e1;
}

.speech-capture__or {
  margin: 0.8rem 0;
  color: #64748b;
  text-align: center;
  font-size: 0.8rem;
  font-weight: 700;
}

.speech-capture__timer {
  margin: 1rem 0;
  font-size: 2rem;
  font-weight: 700;
  text-align: center;
  font-variant-numeric: tabular-nums;
}

.speech-capture__waveform {
  height: 5rem;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 3px;
  margin-bottom: 1.5rem;
  overflow: hidden;
}

.speech-capture__waveform span {
  width: 4px;
  min-height: 6px;
  border-radius: 4px;
  background: #2563eb;
  transition: height 80ms linear;
}

.speech-capture__processing {
  padding: 2rem 0;
  text-align: center;
  color: #475569;
}

.speech-capture__spinner {
  width: 2rem;
  height: 2rem;
  margin: 0 auto 1rem;
  border: 3px solid #cbd5e1;
  border-top-color: #2563eb;
  border-radius: 50%;
  animation: speech-spin 0.8s linear infinite;
}

@keyframes speech-spin {
  to {
    transform: rotate(360deg);
  }
}

.speech-capture__done {
  padding: 1rem 0;
  text-align: center;
}

.speech-capture__check {
  width: 3rem;
  height: 3rem;
  margin: 0 auto 0.75rem;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  background: #dcfce7;
  color: #166534;
  font-size: 1.5rem;
}

.speech-capture__done p {
  color: #64748b;
  font-size: 0.9rem;
}

.speech-capture__reset {
  margin-top: 0.75rem;
  background: #e2e8f0;
  color: #0f172a;
}

.speech-capture__error {
  margin: 1rem 0 0;
  color: #b91c1c;
  font-size: 0.9rem;
  line-height: 1.45;
}
`;