/**
 * Screen 7 (§6) — Visit Detail / Results.
 *
 * Four different screens wearing one route, decided by visit type, status, the
 * calendar day, and the viewer's role:
 *
 *   follow-up            — clinical values only. No modalities, no model, no
 *                          diagnosis section at all.
 *   pending_review       — model output plus, for a clinician, the diagnosis form.
 *   reviewed, same day   — the saved diagnosis, still editable (Rule 5), with the
 *                          revision history for that day.
 *   reviewed, later day  — fully read-only. No dropdown, because a later change
 *                          goes through a new follow-up visit instead.
 *
 * "Same day" is the *server's UTC* calendar day, matching the check in
 * routers/visits.py. Using the browser's local day would put anyone west of UTC
 * into a window where the form is offered and the save then 400s.
 *
 * Back target is fixed to the Patient Profile per §6, not browser history.
 */
import { lazy, Suspense, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { api } from "../api/client";
import { useAuth } from "../auth/useAuth";

// ~1-2 MB of WebGL viewer — only pulled when a clinician actually opens a scan.
const ScanViewer = lazy(() => import("../components/ScanViewer"));

const DIAGNOSIS_OPTIONS = [
  "Nondemented",
  "Demented",
  "Needs further evaluation",
];

const MODALITY_LABELS = {
  idle: "Not uploaded",
  done: "Extracted",
  not_applicable: "Not applicable",
  error: "Failed",
};

function formatDate(value) {
  if (!value) return "—";
  return new Date(value).toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

function formatDateTime(value) {
  if (!value) return "—";
  return new Date(value).toLocaleString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** UTC calendar day, to match the server's boundary rather than the browser's. */
function utcDay(value) {
  return new Date(value).toISOString().slice(0, 10);
}

export default function VisitDetail() {
  const { visitId } = useParams();
  const { user } = useAuth();

  const [visit, setVisit] = useState(null);
  const [patient, setPatient] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState("");

  const [diagnosis, setDiagnosis] = useState("");
  const [notes, setNotes] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [saveError, setSaveError] = useState("");
  const [savedAt, setSavedAt] = useState(null);
  const [showHistory, setShowHistory] = useState(false);
  const [isEditing, setIsEditing] = useState(false);

  // Speech: an object URL for an MP3 blob (transcoded server-side so it plays
  // everywhere), fetched on load.
  const [speechUrl, setSpeechUrl] = useState(null);
  const [speechError, setSpeechError] = useState("");
  // MRI: an object URL for the viewer (NIfTI blob streamed through the API),
  // opened on demand; the download link uses a separate signed URL.
  const [mriScanUrl, setMriScanUrl] = useState(null);
  const [mriError, setMriError] = useState("");
  const [mriLoading, setMriLoading] = useState(false);
  const [mriDownloading, setMriDownloading] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const loadedVisit = await api.getVisit(visitId);
        if (cancelled) return;
        setVisit(loadedVisit);
        setDiagnosis(loadedVisit.doctor_diagnosis ?? "");
        setNotes(loadedVisit.doctor_notes ?? "");

        // Needed for the name in the header and the back link; the visit
        // payload carries only patient_id.
        const loadedPatient = await api.getPatient(loadedVisit.patient_id);
        if (!cancelled) setPatient(loadedPatient);

        // The speech player needs a URL up front; the MRI viewer is opened on
        // demand, so its URL is fetched from the button instead.
        if (loadedVisit.speech_status === "done") {
          try {
            const blob = await api.getVisitAudioBlob(visitId);
            if (!cancelled) setSpeechUrl(URL.createObjectURL(blob));
          } catch (error) {
            if (!cancelled) setSpeechError(error.message);
          }
        }
      } catch (error) {
        if (!cancelled) setLoadError(error.message);
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [visitId]);

  // Revoke object URLs when they change or the page unmounts.
  useEffect(() => {
    if (!mriScanUrl) return undefined;
    return () => URL.revokeObjectURL(mriScanUrl);
  }, [mriScanUrl]);
  useEffect(() => {
    if (!speechUrl) return undefined;
    return () => URL.revokeObjectURL(speechUrl);
  }, [speechUrl]);

  async function toggleMri() {
    setMriError("");
    if (mriScanUrl) {
      setMriScanUrl(null); // close the viewer (URL revoked by the effect)
      return;
    }
    setMriLoading(true);
    try {
      const blob = await api.getVisitScanBlob(visitId);
      setMriScanUrl(URL.createObjectURL(blob));
    } catch (error) {
      setMriError(error.message);
    } finally {
      setMriLoading(false);
    }
  }

  async function downloadMri() {
    setMriError("");
    setMriDownloading(true);
    try {
      const { url } = await api.getVisitFile(visitId, "mri");
      window.open(url, "_blank", "noopener,noreferrer");
    } catch (error) {
      setMriError(error.message);
    } finally {
      setMriDownloading(false);
    }
  }

  async function retrySpeech() {
    setSpeechError("");
    try {
      const blob = await api.getVisitAudioBlob(visitId);
      setSpeechUrl(URL.createObjectURL(blob));
    } catch (error) {
      setSpeechError(error.message);
    }
  }

  async function handleSaveDiagnosis(event) {
    event.preventDefault();
    setSaveError("");
    setIsSaving(true);
    try {
      const updated = await api.saveDiagnosis(visitId, {
        doctor_diagnosis: diagnosis,
        doctor_notes: notes.trim() === "" ? null : notes.trim(),
      });
      setVisit(updated);
      setSavedAt(new Date());
      // Collapse the revise form back to saved values + "Edit diagnosis".
      setIsEditing(false);
      setDiagnosis(updated.doctor_diagnosis ?? "");
      setNotes(updated.doctor_notes ?? "");
    } catch (error) {
      setSaveError(error.message);
    } finally {
      setIsSaving(false);
    }
  }

  if (isLoading) {
    return (
      <div className="page">
        <p className="visit-loading">Loading…</p>
      </div>
    );
  }

  if (loadError) {
    return (
      <div className="page">
        <div className="visit-card">
          <p className="form-error" role="alert">
            {loadError}
          </p>
          <Link className="button-quiet" to="/">
            Back to dashboard
          </Link>
        </div>
      </div>
    );
  }

  const isScreening = visit.visit_type === "screening";
  const isClinician = user.role === "clinician";
  const isReceptionist = user.role === "receptionist";
  const history = visit.diagnosis_history ?? [];

  // Rule 5: first save, or a revision on the same UTC day as the last one.
  const isSameUtcDayAsLastSave =
    visit.diagnosis_saved_at != null &&
    utcDay(visit.diagnosis_saved_at) === utcDay(Date.now());
  const canDiagnose =
    isScreening &&
    isClinician &&
    (visit.status === "pending_review" ||
      (visit.status === "reviewed" && isSameUtcDayAsLastSave));

  // The first diagnosis gets an always-visible form. A same-day revision hides
  // it behind an "Edit diagnosis" button so the doctor doesn't re-save on
  // autopilot; a later day is read-only (handled by canDiagnose above).
  const isFirstSave = canDiagnose && visit.status === "pending_review";
  const isRevision =
    canDiagnose && visit.status === "reviewed" && Boolean(visit.doctor_diagnosis);
  const showForm = isFirstSave || (isRevision && isEditing);

  const normalizedNotes = notes.trim() === "" ? null : notes.trim();
  const isUnchanged =
    isRevision &&
    diagnosis === (visit.doctor_diagnosis ?? "") &&
    normalizedNotes === (visit.doctor_notes ?? null);

  return (
    <div className="page visit-page visit-detail">
      <Link className="back-link" to={`/patients/${visit.patient_id}`}>
        ‹ Back to patient profile
      </Link>

      <header className="visit-header">
        <p className="visit-eyebrow">
          {isScreening ? "Screening visit" : "Follow-up visit"} ·{" "}
          {formatDate(visit.visit_date)}
        </p>
        <h1>{patient?.name ?? "Visit"}</h1>
      </header>

      <div className="profile-actions no-print">
        {/* §6 asks for export-as-PDF on every visit detail. The browser's own
            print-to-PDF is the whole feature: a PDF library would ship ~200 KB
            to reproduce a page we already have, and the print stylesheet drops
            the nav and the controls. */}
        <button type="button" className="button-quiet" onClick={() => window.print()}>
          Export as PDF
        </button>
      </div>

      {/* §6: disclaimer on screening visits only — a follow-up never ran a model. */}
      {isScreening ? (
        <p className="visit-disclaimer">
          Decision-support research tool. The model output is not a diagnosis and
          must be reviewed by a clinician.
        </p>
      ) : null}

      <section className="visit-card">
        <div className="visit-card-head">
          <h2>Clinical measures</h2>
        </div>
        <dl className="detail-grid">
          <div>
            <dt>MMSE</dt>
            <dd>{visit.mmse ?? "—"}</dd>
          </div>
          <div>
            <dt>CDR</dt>
            <dd>{visit.cdr ?? "—"}</dd>
          </div>
          {/* edu/ses are screening-only inputs to the model (§3). */}
          {isScreening ? (
            <>
              <div>
                <dt>Education (years)</dt>
                <dd>{visit.edu ?? "—"}</dd>
              </div>
              <div>
                <dt>Socioeconomic status</dt>
                <dd>{visit.ses ?? "—"}</dd>
              </div>
            </>
          ) : null}
        </dl>
      </section>

      {isScreening ? (
        <section className="visit-card">
          <div className="visit-card-head">
            <h2>Modalities</h2>
          </div>
          <dl className="detail-grid">
            <div>
              <dt>MRI scan</dt>
              <dd>
                {MODALITY_LABELS[visit.mri_status] ?? visit.mri_status}
                {visit.mri_status === "done" ? (
                  <div className="modality-actions no-print">
                    <button
                      type="button"
                      className="button-quiet"
                      onClick={toggleMri}
                      disabled={mriLoading}
                    >
                      {mriLoading
                        ? "Opening…"
                        : mriScanUrl
                          ? "Hide scan"
                          : "View scan"}
                    </button>
                    <button
                      type="button"
                      className="button-quiet"
                      onClick={downloadMri}
                      disabled={mriDownloading}
                    >
                      {mriDownloading ? "Preparing…" : "Download"}
                    </button>
                  </div>
                ) : null}
                {mriError ? (
                  <p className="visit-note no-print">
                    {mriError}{" "}
                    <button type="button" className="link-button" onClick={toggleMri}>
                      Try again
                    </button>
                  </p>
                ) : null}
              </dd>
            </div>
            <div>
              <dt>Speech recording</dt>
              <dd>
                {MODALITY_LABELS[visit.speech_status] ?? visit.speech_status}
                {visit.speech_status === "done" ? (
                  speechUrl ? (
                    <audio
                      className="visit-audio no-print"
                      controls
                      preload="metadata"
                      src={speechUrl}
                    />
                  ) : speechError ? (
                    <p className="visit-note no-print">
                      Couldn’t load the recording.{" "}
                      <button
                        type="button"
                        className="link-button"
                        onClick={retrySpeech}
                      >
                        Try again
                      </button>
                    </p>
                  ) : (
                    <p className="visit-note no-print">Loading player…</p>
                  )
                ) : null}
              </dd>
            </div>
          </dl>

          {mriScanUrl ? (
            <Suspense
              fallback={<p className="visit-note no-print">Loading viewer…</p>}
            >
              <ScanViewer url={mriScanUrl} filename="scan.nii.gz" />
            </Suspense>
          ) : null}

          {visit.status === "awaiting_uploads" ? (
            <p className="visit-note no-print">
              This visit is still incomplete — the model runs once both are in.
              {isReceptionist ? (
                <>
                  {" "}
                  <Link to={`/patients/${visit.patient_id}/new-visit/screening?visitId=${visit.id}`}>
                    Finish the uploads
                  </Link>
                  .
                </>
              ) : null}
            </p>
          ) : null}
        </section>
      ) : null}

      {isScreening && visit.model_prediction ? (
        <section className="visit-card visit-card--result">
          <div className="visit-card-head">
            <h2>Model output</h2>
            {visit.agreement_flag ? (
              <span
                className={`visit-chip visit-chip--${visit.agreement_flag}`}
                title="Whether the doctor's diagnosis matched the model. Informational only."
              >
                {visit.agreement_flag === "match" ? "Doctor agreed" : "Doctor disagreed"}
              </span>
            ) : null}
          </div>

          <div className="visit-predictions">
            <div className="visit-prediction">
              <p className="visit-prediction-label">Quantum SVM</p>
              <p className="visit-prediction-value">{visit.model_prediction}</p>
              {/* NOT a probability: QSVC has no predict_proba, so this is the
                  distance from the decision boundary. Never a percentage. */}
              <p className="visit-prediction-margin">
                margin {visit.model_confidence?.toFixed(2) ?? "—"}
              </p>
            </div>
            <div className="visit-prediction">
              <p className="visit-prediction-label">Classical SVM</p>
              <p className="visit-prediction-value">{visit.svm_prediction ?? "—"}</p>
              <p className="visit-prediction-margin">
                margin {visit.svm_confidence?.toFixed(2) ?? "—"}
              </p>
              <p className="visit-note">Research comparison only.</p>
            </div>
          </div>
        </section>
      ) : null}

      {/* --- diagnosis: screening visits only (§6) ----------------------- */}
      {isScreening ? (
        <section className="visit-card">
          <div className="visit-card-head">
            <h2>Diagnosis</h2>
            {visit.status === "reviewed" ? (
              <span className="visit-chip visit-chip--done">Reviewed</span>
            ) : null}
          </div>

          {visit.doctor_diagnosis ? (
            <dl className="detail-grid">
              <div>
                <dt>Doctor's diagnosis</dt>
                <dd>{visit.doctor_diagnosis}</dd>
              </div>
              <div>
                <dt>Saved</dt>
                <dd>{formatDateTime(visit.diagnosis_saved_at)}</dd>
              </div>
              {visit.doctor_notes ? (
                <div className="detail-grid-wide">
                  <dt>Notes</dt>
                  <dd>{visit.doctor_notes}</dd>
                </div>
              ) : null}
            </dl>
          ) : null}

          {showForm ? (
            <form className="visit-form no-print" onSubmit={handleSaveDiagnosis}>
              <label className="field">
                <span className="field-label">
                  {visit.doctor_diagnosis ? "Revise diagnosis" : "Diagnosis"}
                </span>
                <select
                  value={diagnosis}
                  onChange={(event) => setDiagnosis(event.target.value)}
                  required
                  disabled={isSaving}
                >
                  <option value="">Select…</option>
                  {DIAGNOSIS_OPTIONS.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
                <span className="field-hint">
                  The model never predicts “Needs further evaluation”, so choosing
                  it always records a disagreement.
                </span>
              </label>

              <label className="field">
                <span className="field-label">Notes</span>
                <textarea
                  rows={4}
                  value={notes}
                  onChange={(event) => setNotes(event.target.value)}
                  disabled={isSaving}
                />
              </label>

              {saveError ? (
                <p className="form-error" role="alert">
                  {saveError}
                </p>
              ) : null}

              <div className="visit-form-actions">
                {/* §6: explicit save, nothing auto-saves. */}
                <button
                  type="submit"
                  className="button-primary"
                  disabled={isSaving || isUnchanged}
                >
                  {isSaving ? "Saving…" : "Save diagnosis"}
                </button>
                {isEditing ? (
                  <button
                    type="button"
                    className="button-quiet"
                    disabled={isSaving}
                    onClick={() => {
                      setDiagnosis(visit.doctor_diagnosis ?? "");
                      setNotes(visit.doctor_notes ?? "");
                      setSaveError("");
                      setIsEditing(false);
                    }}
                  >
                    Cancel
                  </button>
                ) : null}
              </div>

              {isUnchanged ? (
                <span className="field-hint">
                  Change the diagnosis or notes to save a revision.
                </span>
              ) : null}

              {savedAt ? (
                <p className="visit-note" role="status">
                  Saved at {formatDateTime(savedAt)}. It can still be revised today;
                  after that, a change goes through a new follow-up visit.
                </p>
              ) : null}
            </form>
          ) : isRevision ? (
            <div className="no-print">
              <button
                type="button"
                className="button-primary"
                onClick={() => {
                  setDiagnosis(visit.doctor_diagnosis ?? "");
                  setNotes(visit.doctor_notes ?? "");
                  setSaveError("");
                  setSavedAt(null);
                  setIsEditing(true);
                }}
              >
                Edit diagnosis
              </button>
              <p className="visit-note">
                This diagnosis can still be revised today; after that, a change
                goes through a new follow-up visit.
              </p>
            </div>
          ) : (
            <p className="visit-note">
              {!isClinician
                ? "Only a clinician can record a diagnosis."
                : visit.status === "awaiting_uploads"
                  ? "This visit can't be diagnosed until both uploads are in and the model has run."
                  : visit.status === "reviewed"
                    ? "Reviewed on an earlier day, so this is now read-only — a revision goes through a new follow-up visit (Product Rule 5)."
                    : "This visit isn't ready for a diagnosis."}
            </p>
          )}

          {/* Rule 5's audit trail. Hidden behind a toggle because on a visit with
              one save it is just that save repeated. */}
          {history.length > 0 ? (
            <div className="diagnosis-history">
              <button
                type="button"
                className="button-quiet no-print"
                onClick={() => setShowHistory((open) => !open)}
                aria-expanded={showHistory}
              >
                {showHistory ? "Hide" : "Show"} diagnosis history ({history.length})
              </button>

              {showHistory ? (
                <ol className="history-list">
                  {history.map((entry) => (
                    <li key={entry.id}>
                      <span className="history-when">{formatDateTime(entry.saved_at)}</span>
                      <span className="history-what">{entry.doctor_diagnosis}</span>
                      {entry.doctor_notes ? (
                        <span className="history-notes">{entry.doctor_notes}</span>
                      ) : null}
                    </li>
                  ))}
                </ol>
              ) : null}
            </div>
          ) : null}
        </section>
      ) : null}
    </div>
  );
}
