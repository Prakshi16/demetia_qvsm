/**
 * The deduped patient list both dashboard sections render (§6 screen 2).
 *
 * One row per patient, never per visit — the API already dedupes and orders by
 * most recent visit, so this component does no sorting of its own; re-sorting
 * here would silently disagree with the server on what "most recent" means.
 *
 * `missing` switches the row's right-hand column to what a resumable screening
 * is still waiting for, which is the only thing that makes the receptionist's
 * incomplete-visits queue actionable.
 */
import { Link } from "react-router-dom";

const VISIT_TYPE_LABELS = {
  screening: "Screening",
  follow_up: "Follow-up",
};

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

/** "Speech pending", "MRI pending", or both — §6's example wording. */
function missingModalities(patient) {
  const missing = [];
  if (patient.latest_mri_status && patient.latest_mri_status !== "done") {
    missing.push("MRI");
  }
  if (patient.latest_speech_status && patient.latest_speech_status !== "done") {
    missing.push("Speech");
  }
  return missing.length === 0 ? null : `${missing.join(" + ")} pending`;
}

export default function PatientList({ patients, emptyNote, missing = false }) {
  if (patients.length === 0) {
    return <p className="list-note">{emptyNote}</p>;
  }

  return (
    <ul className="patient-list">
      {patients.map((patient) => (
        <li key={patient.id}>
          <Link className="patient-row" to={`/patients/${patient.id}`}>
            <span className="patient-row-main">
              <span className="patient-row-name">{patient.name}</span>
              <span className="patient-row-meta">
                {[patient.sex, patient.phone].filter(Boolean).join(" · ") ||
                  "No demographics recorded"}
              </span>
            </span>

            <span className="patient-row-visit">
              <span className="patient-row-date">
                {formatDate(patient.latest_visit_date)}
              </span>
              <span className="patient-row-meta">
                {patient.latest_visit_type
                  ? VISIT_TYPE_LABELS[patient.latest_visit_type] ??
                    patient.latest_visit_type
                  : "No visits yet"}
              </span>
            </span>

            <span className="patient-row-tail">
              {missing ? (
                <span className="patient-chip patient-chip--missing">
                  {missingModalities(patient) ?? "Incomplete"}
                </span>
              ) : (
                <span
                  className={`patient-chip patient-chip--${patient.latest_visit_status ?? "none"}`}
                >
                  {patient.latest_doctor_diagnosis ??
                    STATUS_LABELS[patient.latest_visit_status] ??
                    "—"}
                </span>
              )}
            </span>
          </Link>
        </li>
      ))}
    </ul>
  );
}
