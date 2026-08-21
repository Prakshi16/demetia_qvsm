/**
 * Screen 3 (§6) — Register New Patient.
 *
 * Demographics plus the consent section, which is the part that matters:
 * Product Rule 7 captures consent once, here, at registration — not per visit.
 * So this is the only place the relationship to a consenting guardian is ever
 * recorded, and getting it wrong means an unusable consent record.
 *
 * Creates the patient only. Starting a visit is a separate, deliberate step
 * from the Patient Profile, so this lands there rather than dropping straight
 * into a screening form.
 *
 * Back target is fixed to the Dashboard per §6, not browser history.
 */
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { api } from "../api/client";

// Matches the consent_given_by Postgres enum ("patient" | "guardian").
const CONSENT_PATIENT = "patient";
const CONSENT_GUARDIAN = "guardian";

// `sex` is a free-text column rather than an enum, so these are the offered
// options and not a constraint the API enforces.
const SEX_OPTIONS = ["Female", "Male", "Other", "Prefer not to say"];

const EMPTY_FORM = {
  name: "",
  dob: "",
  sex: "",
  phone: "",
  address: "",
  consent_given_by: CONSENT_PATIENT,
  consent_relationship: "",
};

/** "" -> null, so an optional field the user left blank is absent, not empty. */
function blankToNull(value) {
  const trimmed = value.trim();
  return trimmed === "" ? null : trimmed;
}

export default function RegisterPatient() {
  const navigate = useNavigate();

  const [form, setForm] = useState(EMPTY_FORM);
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const isGuardianConsent = form.consent_given_by === CONSENT_GUARDIAN;

  function updateField(name, value) {
    setForm((current) => ({ ...current, [name]: value }));
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setIsSubmitting(true);

    try {
      const created = await api.createPatient({
        name: form.name.trim(),
        dob: blankToNull(form.dob),
        sex: blankToNull(form.sex),
        phone: blankToNull(form.phone),
        address: blankToNull(form.address),
        consent_given_by: form.consent_given_by,
        // Only ever sent for guardian consent — carrying a stale relationship
        // over from a toggled-back radio would record consent that never
        // happened. The API rejects a guardian with no relationship too.
        consent_relationship: isGuardianConsent
          ? blankToNull(form.consent_relationship)
          : null,
      });

      navigate(`/patients/${created.id}`, { replace: true });
    } catch (submitError) {
      setError(submitError.message);
      setIsSubmitting(false);
    }
  }

  return (
    <div className="page visit-page">
      <Link className="back-link" to="/">
        ‹ Back to dashboard
      </Link>

      <header className="visit-header">
        <p className="visit-eyebrow">New patient</p>
        <h1>Register a patient</h1>
        <p className="visit-subtitle">
          Registration records the patient and their consent. Visits are started
          separately, from the patient’s profile.
        </p>
      </header>

      <form className="visit-form" onSubmit={handleSubmit}>
        <section className="visit-card">
          <div className="visit-card-head">
            <h2>Patient details</h2>
          </div>

          <div className="visit-form">
            <label className="field">
              <span className="field-label">Full name</span>
              <input
                type="text"
                value={form.name}
                onChange={(event) => updateField("name", event.target.value)}
                autoComplete="off"
                required
                disabled={isSubmitting}
              />
            </label>

            <div className="visit-form-row">
              <label className="field">
                <span className="field-label">Date of birth</span>
                <input
                  type="date"
                  value={form.dob}
                  // A birth date in the future is always a typo, and this is the
                  // field the re-screening interval is reasoned about against.
                  max={new Date().toISOString().slice(0, 10)}
                  onChange={(event) => updateField("dob", event.target.value)}
                  disabled={isSubmitting}
                />
              </label>

              <label className="field">
                <span className="field-label">Sex</span>
                <select
                  value={form.sex}
                  onChange={(event) => updateField("sex", event.target.value)}
                  disabled={isSubmitting}
                >
                  <option value="">Not recorded</option>
                  {SEX_OPTIONS.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <div className="visit-form-row">
              <label className="field">
                <span className="field-label">Phone</span>
                <input
                  type="tel"
                  value={form.phone}
                  onChange={(event) => updateField("phone", event.target.value)}
                  autoComplete="off"
                  disabled={isSubmitting}
                />
              </label>

              <label className="field">
                <span className="field-label">Address</span>
                <input
                  type="text"
                  value={form.address}
                  onChange={(event) => updateField("address", event.target.value)}
                  autoComplete="off"
                  disabled={isSubmitting}
                />
              </label>
            </div>
          </div>
        </section>

        <section className="visit-card">
          <div className="visit-card-head">
            <h2>Consent</h2>
          </div>

          <p className="visit-note consent-intro">
            Recorded once, here. It covers the MRI scan, the speech recording and
            the automated analysis for every visit this patient has.
          </p>

          <fieldset className="consent-choice" disabled={isSubmitting}>
            <legend className="field-label">Who is giving consent?</legend>

            <label className="consent-option">
              <input
                type="radio"
                name="consent_given_by"
                value={CONSENT_PATIENT}
                checked={!isGuardianConsent}
                onChange={(event) => updateField("consent_given_by", event.target.value)}
              />
              <span>The patient</span>
            </label>

            <label className="consent-option">
              <input
                type="radio"
                name="consent_given_by"
                value={CONSENT_GUARDIAN}
                checked={isGuardianConsent}
                onChange={(event) => updateField("consent_given_by", event.target.value)}
              />
              <span>A guardian or carer</span>
            </label>
          </fieldset>

          {/* Only rendered for guardian consent (§6), and required when it is:
              "a guardian consented" without saying who they are is not a
              usable consent record. The API enforces the same rule. */}
          {isGuardianConsent ? (
            <label className="field consent-relationship">
              <span className="field-label">Relationship to the patient</span>
              <input
                type="text"
                value={form.consent_relationship}
                onChange={(event) =>
                  updateField("consent_relationship", event.target.value)
                }
                placeholder="e.g. daughter, spouse, legal guardian"
                autoComplete="off"
                required
                disabled={isSubmitting}
              />
            </label>
          ) : null}
        </section>

        {error ? (
          <p className="form-error" role="alert">
            {error}
          </p>
        ) : null}

        <button type="submit" className="button-primary" disabled={isSubmitting}>
          {isSubmitting ? "Registering…" : "Register patient"}
        </button>
      </form>
    </div>
  );
}
