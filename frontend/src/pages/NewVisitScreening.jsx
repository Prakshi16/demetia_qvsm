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
import MriExtractionNote from "../components/MriExtractionNote";
import MriUpload from "../components/MriUpload";
import PredictionPanel from "../components/PredictionPanel";
import SpeechCapture from "../components/SpeechCapture";
import NumberField from "../components/ui/NumberField";
import {
  CDR_OPTIONS,
  CLINICAL_RULES,
  SES_OPTIONS,
  validateChoice,
  validateWholeNumber,
} from "../utils/validate";

// CDR and SES are fixed ordinal scales, so they're selects: only real scale
// points can be chosen. MMSE and education are whole numbers in a range, typed
// into a digits-only field. The API enforces the same rules (schemas.py).
const SES_VALUES = SES_OPTIONS.map((option) => option.value);

function validateClinical(form) {
  const errors = {
    mmse: validateWholeNumber(form.mmse, CLINICAL_RULES.mmse),
    cdr: validateChoice(form.cdr, { label: "CDR", options: CDR_OPTIONS }),
    edu: validateWholeNumber(form.edu, CLINICAL_RULES.edu),
    ses: validateChoice(form.ses, { label: "socioeconomic status", options: SES_VALUES }),
  };
  return Object.fromEntries(Object.entries(errors).filter(([, message]) => message));
}

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
  // Per-field errors, shown once a field is left (blur) or on submit.
  const [touched, setTouched] = useState({});
  const [submitAttempted, setSubmitAttempted] = useState(false);
  // Re-showing the upload widget for a modality that's already "done", so a
  // receptionist who picked the wrong file can replace it. Reset back to false
  // once the replacement upload succeeds (handleUploadDone below).
  const [replacingMri, setReplacingMri] = useState(false);
  const [replacingSpeech, setReplacingSpeech] = useState(false);

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
              mmse: loadedVisit.mmse == null ? "" : String(loadedVisit.mmse),
              cdr: loadedVisit.cdr == null ? "" : String(loadedVisit.cdr),
              edu: loadedVisit.edu == null ? "" : String(loadedVisit.edu),
              ses: loadedVisit.ses == null ? "" : String(loadedVisit.ses),
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
    setSubmitAttempted(true);
    if (Object.keys(validateClinical(form)).length > 0) return;
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

  const fieldErrors = validateClinical(form);
  // Format/range errors show when a field is left; a "required" error on an
  // empty field waits for submit, so it can't appear (and shift the Save
  // button) the moment someone tabs past a field they haven't reached yet.
  const shownError = (name) =>
    submitAttempted || (touched[name] && String(form[name]).trim() !== "")
      ? fieldErrors[name]
      : undefined;
  const markTouched = (name) => () => setTouched((current) => ({ ...current, [name]: true }));

  const isClinicalSaved = Boolean(visit);
  const mriDone = visit?.mri_status === "done";
  const speechDone = visit?.speech_status === "done";
  const hasPrediction = Boolean(visit?.model_prediction);
  // Matches the backend guard in mri_upload.py/speech.py: once a diagnosis is saved,
  // re-upload is refused outright, so don't offer "Replace file" past that point.
  const canReplaceUploads = visit?.status !== "reviewed";

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
          <form className="visit-form" onSubmit={handleCreateVisit} noValidate>
            <div className="visit-form-row">
              <NumberField
                label="MMSE"
                name="mmse"
                value={form.mmse}
                onChange={(value) => updateField("mmse", value)}
                onBlur={markTouched("mmse")}
                min={CLINICAL_RULES.mmse.min}
                max={CLINICAL_RULES.mmse.max}
                hint="Whole number, 0–30. The model was trained on 12–30."
                error={shownError("mmse")}
                disabled={isSubmitting}
              />

              <label className="field">
                <span className="field-label">CDR</span>
                <select
                  value={form.cdr}
                  onChange={(event) => updateField("cdr", event.target.value)}
                  onBlur={markTouched("cdr")}
                  required
                  disabled={isSubmitting}
                  aria-invalid={shownError("cdr") ? "true" : undefined}
                >
                  <option value="">Select…</option>
                  {CDR_OPTIONS.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
                <span className="field-hint">Clinical Dementia Rating. Recorded, not a model input.</span>
                {shownError("cdr") ? (
                  <span className="field-error" role="alert">{shownError("cdr")}</span>
                ) : null}
              </label>
            </div>

            <div className="visit-form-row">
              <NumberField
                label="Education (years)"
                name="edu"
                value={form.edu}
                onChange={(value) => updateField("edu", value)}
                onBlur={markTouched("edu")}
                min={CLINICAL_RULES.edu.min}
                max={CLINICAL_RULES.edu.max}
                hint="Whole years of schooling, 0–30. The model was trained on 6–23."
                error={shownError("edu")}
                disabled={isSubmitting}
              />

              <label className="field">
                <span className="field-label">Socioeconomic status</span>
                <select
                  value={form.ses}
                  onChange={(event) => updateField("ses", event.target.value)}
                  onBlur={markTouched("ses")}
                  required
                  disabled={isSubmitting}
                  aria-invalid={shownError("ses") ? "true" : undefined}
                >
                  <option value="">Select…</option>
                  {SES_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
                <span className="field-hint">Hollingshead index, 1 (highest) to 5 (lowest).</span>
                {shownError("ses") ? (
                  <span className="field-error" role="alert">{shownError("ses")}</span>
                ) : null}
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
              {mriDone && !replacingMri ? (
                <>
                <div className="visit-done-row">
                  <p className="visit-done">Features extracted ✓</p>
                  {canReplaceUploads ? (
                    <button
                      type="button"
                      className="button-quiet"
                      onClick={() => setReplacingMri(true)}
                    >
                      Wrong file? Replace
                    </button>
                  ) : null}
                </div>
                <MriExtractionNote visit={visit} />
                </>
              ) : (
                <MriUpload
                  visitId={visit.id}
                  onDone={(updated) => {
                    handleUploadDone(updated);
                    setReplacingMri(false);
                  }}
                />
              )}
            </div>

            <div className="visit-upload-slot">
              <h3>Speech recording</h3>
              {speechDone && !replacingSpeech ? (
                <div className="visit-done-row">
                  <p className="visit-done">Features extracted ✓</p>
                  {canReplaceUploads ? (
                    <button
                      type="button"
                      className="button-quiet"
                      onClick={() => setReplacingSpeech(true)}
                    >
                      Wrong file? Replace
                    </button>
                  ) : null}
                </div>
              ) : (
                <SpeechCapture
                  visitId={visit.id}
                  onDone={(updated) => {
                    handleUploadDone(updated);
                    setReplacingSpeech(false);
                  }}
                />
              )}
            </div>
          </div>
        ) : null}
      </section>

      {/* --- Result ---------------------------------------------------- */}
      {/* Both models run server-side inside the second upload request (the
          classical SVM in milliseconds, the QSVM research view in ~0.5 s), so
          by the time that response lands both predictions are in it. Nothing
          to poll. */}
      {hasPrediction ? (
        <PredictionPanel
          visit={visit}
          title="3. Model output"
          badge={<span className="badge badge--success">Sent for review</span>}
        >
          <div className="visit-form-actions" style={{ marginTop: "var(--space-lg)" }}>
            <button
              type="button"
              className="button-primary"
              onClick={() => navigate(`/visits/${visit.id}`)}
            >
              View visit detail
            </button>
          </div>
        </PredictionPanel>
      ) : null}
    </div>
  );
}
