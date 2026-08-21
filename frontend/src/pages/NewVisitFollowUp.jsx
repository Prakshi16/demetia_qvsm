/**
 * Screen 6 (§6) — New Visit: Follow-up.
 *
 * The small one: two numbers, one submit. No file cards, no model run, no
 * review step. It saves as `completed` immediately and adds a point to the
 * patient's trend chart.
 *
 * The one piece of real logic is the guard. A follow-up is only valid when §4
 * permits it — the patient needs a confirmed diagnosis on record and must be
 * inside the re-screen window — and POST /visits rejects it with a 400
 * otherwise. Screen 4's "+ New Visit" is supposed to call next-visit-type and
 * route here only when it says follow_up, but that screen is Govind's and this
 * URL can be reached directly, so this page checks for itself rather than
 * letting the user fill in a form that is going to be refused on submit.
 *
 * Back target is fixed to the Patient Profile per §6, not browser history.
 */
import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { api } from "../api/client";

// Same fixed scale as the screening form: the model was trained on
// {0, 0.5, 1, 2}, and the trend chart plots against those screening values.
const CDR_OPTIONS = ["0", "0.5", "1", "2"];

function toNumberOrNull(value) {
  const trimmed = String(value).trim();
  if (trimmed === "") return null;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : null;
}

export default function NewVisitFollowUp() {
  const { patientId } = useParams();
  const navigate = useNavigate();

  const [patient, setPatient] = useState(null);
  const [allowed, setAllowed] = useState(null); // { visit_type, reason }
  const [mmse, setMmse] = useState("");
  const [cdr, setCdr] = useState("");

  const [isLoading, setIsLoading] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [formError, setFormError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const [loadedPatient, decision] = await Promise.all([
          api.getPatient(patientId),
          api.getNextVisitType(patientId),
        ]);
        if (cancelled) return;
        setPatient(loadedPatient);
        setAllowed(decision);
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
  }, [patientId]);

  async function handleSubmit(event) {
    event.preventDefault();
    setFormError("");
    setIsSubmitting(true);

    try {
      await api.createVisit({
        patient_id: patientId,
        visit_type: "follow_up",
        mmse: toNumberOrNull(mmse),
        cdr: toNumberOrNull(cdr),
        // edu/ses are screening-only (§3): the model isn't re-run on a
        // follow-up, so they're deliberately not collected here.
      });

      // Straight back to the profile, where the new trend point is now visible.
      navigate(`/patients/${patientId}`, { replace: true });
    } catch (submitError) {
      setFormError(submitError.message);
      setIsSubmitting(false);
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
          <Link className="button-quiet" to={`/patients/${patientId}`}>
            Back to patient profile
          </Link>
        </div>
      </div>
    );
  }

  // §4 says this patient needs a full screening instead. Say why, and send them
  // to the right screen rather than showing a form that can only fail.
  if (allowed?.visit_type !== "follow_up") {
    return (
      <div className="page visit-page">
        <Link className="back-link" to={`/patients/${patientId}`}>
          ‹ Back to patient profile
        </Link>

        <div className="visit-card">
          <div className="visit-card-head">
            <h2>A follow-up isn’t available for this patient</h2>
          </div>
          <p className="visit-subtitle">{allowed?.reason}</p>
          <Link
            className="button-primary"
            to={`/patients/${patientId}/new-visit/screening`}
          >
            Start a screening visit instead
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="page visit-page">
      <Link className="back-link" to={`/patients/${patientId}`}>
        ‹ Back to patient profile
      </Link>

      <header className="visit-header">
        <p className="visit-eyebrow">Follow-up visit</p>
        <h1>{patient?.name}</h1>
        <p className="visit-subtitle">
          Two clinical measures, added to the patient’s trend. No scan, no
          recording, and no model run — a follow-up is recorded as complete and
          doesn’t go for review.
        </p>
      </header>

      <section className="visit-card">
        <form className="visit-form" onSubmit={handleSubmit}>
          <div className="visit-form-row">
            <label className="field">
              <span className="field-label">MMSE</span>
              <input
                type="number"
                min="0"
                max="30"
                step="1"
                value={mmse}
                onChange={(event) => setMmse(event.target.value)}
                required
                disabled={isSubmitting}
              />
              <span className="field-hint">0–30.</span>
            </label>

            <label className="field">
              <span className="field-label">CDR</span>
              <select
                value={cdr}
                onChange={(event) => setCdr(event.target.value)}
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

          {formError ? (
            <p className="form-error" role="alert">
              {formError}
            </p>
          ) : null}

          <button type="submit" className="button-primary" disabled={isSubmitting}>
            {isSubmitting ? "Saving…" : "Save follow-up visit"}
          </button>
        </form>
      </section>
    </div>
  );
}
