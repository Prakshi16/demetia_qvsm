/**
 * Screen 4 (§6) — Patient Profile.
 *
 * The hub: demographics, the combined MMSE/CDR trend, and the visit history as
 * clickable date rectangles (§6 is explicit that this is not a flat table).
 * Every other screen's back link lands here, so it is the one screen that has to
 * exist for the app to be navigable at all.
 *
 * "+ New Visit" calls next-visit-type first and routes accordingly, so the
 * receptionist never has to decide between a screening and a follow-up — that is
 * §4's job, server-side. The "Start full screening instead" override sits beside
 * it and is offered regardless of what §4 returned, per §4's closing note.
 *
 * Back target is fixed to the Dashboard per §6, not browser history.
 */
import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { api } from "../api/client";
import TrendChart from "../components/TrendChart";

const VISIT_TYPE_LABELS = {
  screening: "Screening",
  follow_up: "Follow-up",
};

// Wording aimed at the person reading the row, not the column value.
const STATUS_LABELS = {
  awaiting_uploads: "Awaiting uploads",
  pending_review: "Awaiting review",
  reviewed: "Reviewed",
  completed: "Completed",
};

function formatDate(value) {
  if (!value) return "—";
  return new Date(value).toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

function formatAge(dob) {
  if (!dob) return null;
  const born = new Date(dob);
  const now = new Date();
  let age = now.getFullYear() - born.getFullYear();
  const monthDelta = now.getMonth() - born.getMonth();
  if (monthDelta < 0 || (monthDelta === 0 && now.getDate() < born.getDate())) age -= 1;
  return age;
}

export default function PatientProfile() {
  const { patientId } = useParams();
  const navigate = useNavigate();

  const [patient, setPatient] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  const [isRouting, setIsRouting] = useState(false);
  const [routeError, setRouteError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const loaded = await api.getPatient(patientId);
        if (!cancelled) setPatient(loaded);
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

  // §4 decides; this only routes. force=true is the manual override, which is
  // always allowed to return screening.
  async function startVisit(force) {
    setRouteError("");
    setIsRouting(true);
    try {
      const decision = await api.getNextVisitType(patientId, force);
      const path = decision.visit_type === "follow_up" ? "follow-up" : "screening";
      navigate(`/patients/${patientId}/new-visit/${path}`);
    } catch (error) {
      setRouteError(error.message);
      setIsRouting(false);
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

  const age = formatAge(patient.dob);
  const visits = patient.visits ?? [];

  return (
    <div className="page visit-page">
      <Link className="back-link" to="/">
        ‹ Back to dashboard
      </Link>

      <header className="visit-header">
        <p className="visit-eyebrow">Patient</p>
        <h1>{patient.name}</h1>
        <p className="visit-subtitle">
          {[
            patient.sex,
            age == null ? null : `${age} years old`,
            patient.phone,
          ]
            .filter(Boolean)
            .join(" · ") || "No demographics recorded."}
        </p>
      </header>

      <div className="profile-actions">
        <button
          type="button"
          className="button-primary"
          onClick={() => startVisit(false)}
          disabled={isRouting}
        >
          {isRouting ? "Checking…" : "+ New visit"}
        </button>
        {/* §4: offered regardless of what the decision logic returned. */}
        <button
          type="button"
          className="button-quiet"
          onClick={() => startVisit(true)}
          disabled={isRouting}
        >
          Start full screening instead
        </button>
      </div>

      {routeError ? (
        <p className="form-error" role="alert">
          {routeError}
        </p>
      ) : null}

      <section className="visit-card">
        <div className="visit-card-head">
          <h2>Details</h2>
        </div>
        <dl className="detail-grid">
          <div>
            <dt>Date of birth</dt>
            <dd>{formatDate(patient.dob)}</dd>
          </div>
          <div>
            <dt>Sex</dt>
            <dd>{patient.sex || "—"}</dd>
          </div>
          <div>
            <dt>Phone</dt>
            <dd>{patient.phone || "—"}</dd>
          </div>
          <div>
            <dt>Address</dt>
            <dd>{patient.address || "—"}</dd>
          </div>
          <div>
            <dt>Consent given by</dt>
            {/* Rule 7: consent is captured once, at registration, and covers
                every visit — so it belongs on the profile, not on each visit. */}
            <dd>
              {patient.consent_given_by === "guardian"
                ? `Guardian${patient.consent_relationship ? ` (${patient.consent_relationship})` : ""}`
                : "The patient"}
            </dd>
          </div>
          <div>
            <dt>Registered</dt>
            <dd>{formatDate(patient.created_at)}</dd>
          </div>
        </dl>
      </section>

      <section className="visit-card">
        <div className="visit-card-head">
          <h2>Cognitive trend</h2>
        </div>
        <TrendChart points={patient.trend} />
      </section>

      <section className="visit-card">
        <div className="visit-card-head">
          <h2>Visit history</h2>
          <span className="visit-chip">
            {visits.length} visit{visits.length === 1 ? "" : "s"}
          </span>
        </div>

        {visits.length === 0 ? (
          <p className="list-note">
            No visits yet. “+ New visit” starts the first screening.
          </p>
        ) : (
          <ol className="visit-history">
            {visits.map((visit) => (
              <li key={visit.id}>
                {/* §6: each entry is a clickable "visit date rectangle". */}
                <Link className="visit-rect" to={`/visits/${visit.id}`}>
                  <span className="visit-rect-date">{formatDate(visit.visit_date)}</span>
                  <span className="visit-rect-type">
                    {VISIT_TYPE_LABELS[visit.visit_type] ?? visit.visit_type}
                  </span>
                  <span className={`visit-rect-status visit-rect-status--${visit.status}`}>
                    {STATUS_LABELS[visit.status] ?? visit.status}
                  </span>
                  <span className="visit-rect-outcome">
                    {visit.doctor_diagnosis
                      ? visit.doctor_diagnosis
                      : visit.model_prediction
                        ? `Model: ${visit.model_prediction}`
                        : ""}
                  </span>
                </Link>
              </li>
            ))}
          </ol>
        )}
      </section>
    </div>
  );
}
