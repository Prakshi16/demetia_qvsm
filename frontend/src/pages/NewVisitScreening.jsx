/**
 * Screen 5 (§6) — New Visit: Screening.
 *
 * The demo's critical path: this is the only screen that can put a visit
 * through the model. It hosts the clinical form plus both upload cards.
 *
 * Why two stages rather than one form with three sections: there is no
 * PATCH /visits/{id}. Clinical fields can only be set at creation (POST /visits),
 * and the upload endpoints need a visit_id that doesn't exist until then. So the
 * clinical form is saved first, which creates the visit in `awaiting_uploads`,
 * and the upload cards unlock against the returned id.
 *
 * That is also exactly what Product Rule 2A wants: a half-finished screening is
 * a real, resumable state, not lost work. A receptionist can save the clinical
 * form, walk the patient to the scanner, and come back to
 * /patients/:patientId/new-visit/screening?visitId=... to finish the uploads.
 *
 * Back target is fixed to the Patient Profile per §6, not browser history.
 */
import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";

import { api } from "../api/client";
import MriUpload from "../components/MriUpload";
import SpeechCapture from "../components/SpeechCapture";

// CDR is a fixed clinical scale, and the model was trained on {0, 0.5, 1, 2}
// only — a free number field would let a 3 through to a pipeline that has never
// seen one, so this is a select.
const CDR_OPTIONS = ["0", "0.5", "1", "2"];

const EMPTY_FORM = { mmse: "", cdr: "", edu: "", ses: "" };

/** "" -> null so the API gets a real absence rather than 0 or NaN. */
function toNumberOrNull(value) {
  const trimmed = String(value).trim();
  if (trimmed === "") return null;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : null;
}

export default function NewVisitScreening() {
  const { patientId } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();

  const resumeVisitId = searchParams.get("visitId");

  const [patient, setPatient] = useState(null);
  const [visit, setVisit] = useState(null);
  const [form, setForm] = useState(EMPTY_FORM);

  const [isLoading, setIsLoading] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [formError, setFormError] = useState("");

  // Load the patient, and the in-progress visit if we were sent back to finish
  // one. Both are read-only lookups, so a failure here is fatal to the screen
  // rather than something to retry inline.
  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const [loadedPatient, loadedVisit] = await Promise.all([
          api.getPatient(patientId),
          resumeVisitId ? api.getVisit(resumeVisitId) : Promise.resolve(null),
        ]);

        if (cancelled) return;

        setPatient(loadedPatient);

        if (loadedVisit) {
          // Guard the two ways a hand-edited URL goes wrong. Cross-hospital
          // access is already a 404 from the API (Rule 12); these are the cases
          // that would otherwise silently show the wrong visit.
          if (loadedVisit.patient_id !== loadedPatient.id) {
            setLoadError("That visit belongs to a different patient.");
          } else if (loadedVisit.visit_type !== "screening") {
            setLoadError("That visit is a follow-up, which has no uploads to finish.");
          } else {
            setVisit(loadedVisit);
            setForm({
              mmse: loadedVisit.mmse ?? "",
              cdr: loadedVisit.cdr == null ? "" : String(loadedVisit.cdr),
              edu: loadedVisit.edu ?? "",
              ses: loadedVisit.ses ?? "",
            });
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
  }, [patientId, resumeVisitId]);

  function updateField(name, value) {
    setForm((current) => ({ ...current, [name]: value }));
  }

  async function handleCreateVisit(event) {
    event.preventDefault();
    setFormError("");
    setIsSubmitting(true);

    try {
      const created = await api.createVisit({
        patient_id: patientId,
        visit_type: "screening",
        mmse: toNumberOrNull(form.mmse),
        cdr: toNumberOrNull(form.cdr),
        edu: toNumberOrNull(form.edu),
        ses: toNumberOrNull(form.ses),
      });

      setVisit(created);
      // Put the new id in the URL so a reload — or a receptionist coming back
      // tomorrow — resumes this visit instead of creating a second one.
      setSearchParams({ visitId: created.id }, { replace: true });
    } catch (error) {
      setFormError(error.message);
    } finally {
      setIsSubmitting(false);
    }
  }

  // Both upload endpoints return the whole visit (VisitDetailOut), so the
  // modality states and the prediction all come from the server rather than
  // being tracked separately in the UI.
  const handleUploadDone = useCallback((updatedVisit) => {
    if (updatedVisit?.id) setVisit(updatedVisit);
  }, []);

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
          <Link className="button-quiet" to={`/patients/${patientId}`}>
            Back to patient profile
          </Link>
        </div>
      </div>
    );
  }

  const isClinicalSaved = Boolean(visit);
  const mriDone = visit?.mri_status === "done";
  const speechDone = visit?.speech_status === "done";
  const hasPrediction = Boolean(visit?.model_prediction);

  return (
    <div className="page visit-page">
      <Link className="back-link" to={`/patients/${patientId}`}>
        ‹ Back to patient profile
      </Link>

      <header className="visit-header">
        <p className="visit-eyebrow">Screening visit</p>
        <h1>{patient?.name}</h1>
        <p className="visit-subtitle">
          Clinical measures, an MRI scan and a speech recording. The model runs
          automatically once both uploads are in.
        </p>
      </header>

      {/* AI disclaimer, screening visits only (§6). It sits above the form, not
          next to the result, so it's read before anything is entered. */}
      <p className="visit-disclaimer">
        Decision-support research tool. The model output is not a diagnosis and
        must be reviewed by a clinician.
      </p>

      {/* --- Step 1: clinical form ------------------------------------- */}
      <section className="visit-card">
        <div className="visit-card-head">
          <h2>1. Clinical measures</h2>
          {isClinicalSaved ? <span className="visit-chip visit-chip--done">Saved ✓</span> : null}
        </div>

        {isClinicalSaved ? (
          // Saved values are read-only: there is no endpoint to change them, and
          // letting someone edit a field that won't persist is worse than
          // showing it as fixed.
          <dl className="detail-grid">
            <div>
              <dt>MMSE</dt>
              <dd>{visit.mmse ?? "—"}</dd>
            </div>
            <div>
              <dt>CDR</dt>
              <dd>{visit.cdr ?? "—"}</dd>
            </div>
            <div>
              <dt>Education (years)</dt>
              <dd>{visit.edu ?? "—"}</dd>
            </div>
            <div>
              <dt>Socioeconomic status</dt>
              <dd>{visit.ses ?? "—"}</dd>
            </div>
          </dl>
        ) : (
          <form className="visit-form" onSubmit={handleCreateVisit}>
            <div className="visit-form-row">
              <label className="field">
                <span className="field-label">MMSE</span>
                <input
                  type="number"
                  min="0"
                  max="30"
                  step="1"
                  value={form.mmse}
                  onChange={(event) => updateField("mmse", event.target.value)}
                  required
                  disabled={isSubmitting}
                />
                <span className="field-hint">0–30. Model trained on 12–30.</span>
              </label>

              <label className="field">
                <span className="field-label">CDR</span>
                <select
                  value={form.cdr}
                  onChange={(event) => updateField("cdr", event.target.value)}
                  required
                  disabled={isSubmitting}
                >
                  <option value="">Select…</option>
                  {CDR_OPTIONS.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
                <span className="field-hint">Clinical Dementia Rating.</span>
              </label>
            </div>

            <div className="visit-form-row">
              <label className="field">
                <span className="field-label">Education (years)</span>
                <input
                  type="number"
                  min="0"
                  max="30"
                  step="1"
                  value={form.edu}
                  onChange={(event) => updateField("edu", event.target.value)}
                  required
                  disabled={isSubmitting}
                />
                <span className="field-hint">Model trained on 6–21.</span>
              </label>

              <label className="field">
                <span className="field-label">Socioeconomic status</span>
                <input
                  type="number"
                  min="1"
                  max="5"
                  step="1"
                  value={form.ses}
                  onChange={(event) => updateField("ses", event.target.value)}
                  required
                  disabled={isSubmitting}
                />
                <span className="field-hint">1–5.</span>
              </label>
            </div>

            {formError ? (
              <p className="form-error" role="alert">
                {formError}
              </p>
            ) : null}

            <button type="submit" className="button-primary" disabled={isSubmitting}>
              {isSubmitting ? "Saving…" : "Save and continue to uploads"}
            </button>
            <p className="visit-note">
              Saving creates the visit. You can leave and finish the uploads later —
              it stays in the incomplete-visits queue until both are in.
            </p>
          </form>
        )}
      </section>

      {/* --- Step 2: the two modalities -------------------------------- */}
      <section className={`visit-card${isClinicalSaved ? "" : " visit-card--locked"}`}>
        <div className="visit-card-head">
          <h2>2. Uploads</h2>
          {isClinicalSaved ? null : (
            <span className="visit-chip">Save the clinical measures first</span>
          )}
        </div>

        {isClinicalSaved ? (
          <div className="visit-uploads">
            <div className="visit-upload-slot">
              <h3>MRI scan</h3>
              {mriDone ? (
                <p className="visit-done">Features extracted ✓</p>
              ) : (
                <MriUpload visitId={visit.id} onDone={handleUploadDone} />
              )}
            </div>

            <div className="visit-upload-slot">
              <h3>Speech recording</h3>
              {speechDone ? (
                <p className="visit-done">Features extracted ✓</p>
              ) : (
                <SpeechCapture visitId={visit.id} onDone={handleUploadDone} />
              )}
            </div>
          </div>
        ) : null}
      </section>

      {/* --- Result ---------------------------------------------------- */}
      {/* The model runs server-side inside the second upload request and takes
          ~450 ms, so by the time that response lands the prediction is already
          in it. Nothing to poll. */}
      {hasPrediction ? (
        <section className="visit-card visit-card--result">
          <div className="visit-card-head">
            <h2>3. Model output</h2>
            <span className="visit-chip visit-chip--done">Sent for review</span>
          </div>

          <div className="visit-predictions">
            <div className="visit-prediction">
              <p className="visit-prediction-label">Quantum SVM</p>
              <p className="visit-prediction-value">{visit.model_prediction}</p>
              {/* NOT a probability: QSVC has no predict_proba, so this is the
                  distance from the decision boundary. Never render it as a
                  percentage or a bar. */}
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

          <button
            type="button"
            className="button-primary"
            onClick={() => navigate(`/visits/${visit.id}`)}
          >
            View visit detail
          </button>
        </section>
      ) : null}
    </div>
  );
}
