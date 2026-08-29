/**
 * Screen 2 (§6) — Dashboard. Home, so it has no back target.
 *
 * Role-aware, and the two roles differ in what sits *above* the general patient
 * list rather than in the list itself:
 *
 *   clinician    — a "pending review" search bar over /patients/pending-review,
 *                  which is the doctor's actual queue (Product Rule 3: complete
 *                  visits only), then a second general search over /dashboard.
 *   receptionist — one general search bar, plus a distinct "Incomplete visits"
 *                  section over /patients/incomplete-visits. §6 is explicit that
 *                  this is a separate list and not a filter toggle on the one
 *                  below it, and the two never overlap: a visit is either
 *                  missing a modality or waiting for a doctor, never both.
 *   hospital_admin — no patient data at all. The admin manages people: the
 *                  clinicians and receptionists at their hospital, over /staff.
 *                  This is a full early return; none of the patient sections
 *                  below render for an admin.
 *
 * The queues are separate endpoints, so they get their own search state. A
 * single shared search box would suggest they are two views of one list.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";

import { api } from "../api/client";
import PatientList from "../components/PatientList";
import { useAuth } from "../auth/useAuth";

const ROLE_LABEL = {
  clinician: "Clinician",
  receptionist: "Receptionist",
  hospital_admin: "Hospital admin",
};

/** Debounced fetch of one list. Returns { items, isLoading, error }. */
function usePatientQuery(fetcher, search, enabled = true) {
  const [items, setItems] = useState([]);
  const [isLoading, setIsLoading] = useState(enabled);
  const [error, setError] = useState("");
  const hasLoaded = useRef(false);

  // The caller passes an inline arrow, so without this the effect below would
  // re-run on every render and the debounce would never settle.
  const stable = useCallback(fetcher, [fetcher]);

  useEffect(() => {
    if (!enabled) {
      setItems([]);
      setIsLoading(false);
      return undefined;
    }

    let cancelled = false;
    setIsLoading(true);

    // 250 ms: long enough that typing a name is one request rather than eight,
    // short enough that the list still feels live. The *first* load is not
    // debounced — there is nothing to coalesce yet, and delaying it just puts
    // a quarter-second of "Loading…" in front of every dashboard mount.
    const timer = setTimeout(async () => {
      try {
        const result = await stable(search);
        if (!cancelled) {
          setItems(result);
          setError("");
        }
      } catch (loadError) {
        if (!cancelled) setError(loadError.message);
      } finally {
        if (!cancelled) {
          setIsLoading(false);
          hasLoaded.current = true;
        }
      }
    }, hasLoaded.current ? 250 : 0);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [stable, search, enabled]);

  return { items, isLoading, error };
}

export default function Dashboard() {
  const { user } = useAuth();
  const isClinician = user.role === "clinician";
  const isReceptionist = user.role === "receptionist";
  const isHospitalAdmin = user.role === "hospital_admin";

  const [generalSearch, setGeneralSearch] = useState("");
  const [queueSearch, setQueueSearch] = useState("");

  // An admin never hits /dashboard — it returns patient data they have no
  // business seeing. Gating the query keeps that request from firing at all.
  const general = usePatientQuery(api.getDashboard, generalSearch, !isHospitalAdmin);
  const review = usePatientQuery(api.getPendingReview, queueSearch, isClinician);
  const incomplete = usePatientQuery(api.getIncompleteVisits, "", isReceptionist);

  // usePatientQuery is list-shaped and role-gated, which is exactly what the
  // staff roster needs too — listStaff ignores the search arg, so `staffRefetch`
  // is just a counter we bump to re-run the effect after adding someone.
  const [staffRefetch, setStaffRefetch] = useState(0);
  const staff = usePatientQuery(api.listStaff, String(staffRefetch), isHospitalAdmin);

  if (isHospitalAdmin) {
    return (
      <AdminDashboard
        name={user.name}
        staff={staff}
        onStaffAdded={() => setStaffRefetch((n) => n + 1)}
      />
    );
  }

  return (
    <div className="page visit-page">
      <header className="visit-header">
        <p className="visit-eyebrow">Dashboard</p>
        <h1>
          {isClinician ? "Patients awaiting your review" : `Welcome, ${user.name}`}
        </h1>
        <p className="visit-subtitle">
          {isClinician
            ? "Screening visits with a model result and no diagnosis yet. Everything else is below."
            : "Search the patient register, or pick up a visit that still needs its uploads."}
        </p>
      </header>

      {/* --- clinician: the review queue (Product Rule 3) ---------------- */}
      {isClinician ? (
        <section className="visit-card">
          <div className="visit-card-head">
            <h2>Pending review</h2>
            {review.isLoading ? null : (
              <span className="visit-chip">{review.items.length}</span>
            )}
          </div>

          <label className="field">
            <span className="field-label">Search patients pending review</span>
            <input
              type="search"
              value={queueSearch}
              onChange={(event) => setQueueSearch(event.target.value)}
              placeholder="Name, phone or ID"
              autoComplete="off"
            />
          </label>

          {review.error ? (
            <p className="form-error" role="alert">
              {review.error}
            </p>
          ) : review.isLoading ? (
            <p className="list-note">Loading…</p>
          ) : (
            <PatientList
              patients={review.items}
              emptyNote={
                queueSearch
                  ? "No patients pending review match that search."
                  : "Nothing is waiting for review right now."
              }
            />
          )}
        </section>
      ) : null}

      {/* --- receptionist: resumable screenings (Product Rule 2A) -------- */}
      {isReceptionist ? (
        <section className="visit-card">
          <div className="visit-card-head">
            <h2>Incomplete visits</h2>
            {incomplete.isLoading ? null : (
              <span className="visit-chip">{incomplete.items.length}</span>
            )}
          </div>
          <p className="visit-note">
            Screenings where the clinical form is saved but a scan or recording is
            still missing. Opening one picks up where it left off.
          </p>

          {incomplete.error ? (
            <p className="form-error" role="alert">
              {incomplete.error}
            </p>
          ) : incomplete.isLoading ? (
            <p className="list-note">Loading…</p>
          ) : (
            <PatientList
              patients={incomplete.items}
              missing
              emptyNote="No half-finished visits. Everything started has been completed."
            />
          )}
        </section>
      ) : null}

      {/* --- both roles: the general register --------------------------- */}
      <section className="visit-card">
        <div className="visit-card-head">
          <h2>All patients</h2>
          {isReceptionist ? (
            <Link className="button-primary" to="/patients/new">
              + Register new patient
            </Link>
          ) : null}
        </div>

        <label className="field">
          <span className="field-label">Search patients</span>
          <input
            type="search"
            value={generalSearch}
            onChange={(event) => setGeneralSearch(event.target.value)}
            placeholder="Name, phone or ID"
            autoComplete="off"
          />
        </label>

        {general.error ? (
          <p className="form-error" role="alert">
            {general.error}
          </p>
        ) : general.isLoading ? (
          <p className="list-note">Loading…</p>
        ) : (
          <PatientList
            patients={general.items}
            emptyNote={
              generalSearch
                ? "No patients match that search."
                : "No patients registered yet."
            }
          />
        )}
      </section>
    </div>
  );
}

const EMPTY_STAFF_FORM = { name: "", email: "", password: "", role: "clinician" };

/**
 * The hospital_admin's whole dashboard. No patient data: an admin's job is the
 * people at their hospital. `staff` is the { items, isLoading, error } shape
 * usePatientQuery returns; `onStaffAdded` re-runs that query after a successful
 * add so the new row shows without a reload.
 */
function AdminDashboard({ name, staff, onStaffAdded }) {
  const [form, setForm] = useState(EMPTY_STAFF_FORM);
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  function updateField(field, value) {
    setForm((current) => ({ ...current, [field]: value }));
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setIsSubmitting(true);
    try {
      await api.addStaff({
        name: form.name.trim(),
        email: form.email.trim(),
        password: form.password,
        role: form.role,
      });
      setForm(EMPTY_STAFF_FORM);
      onStaffAdded();
    } catch (submitError) {
      setError(submitError.message);
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="page visit-page">
      <header className="visit-header">
        <p className="visit-eyebrow">Dashboard</p>
        <h1>Welcome, {name}</h1>
        <p className="visit-subtitle">
          Manage the clinicians and receptionists at your hospital. Patient
          records aren’t part of the admin role.
        </p>
      </header>

      <section className="visit-card">
        <div className="visit-card-head">
          <h2>Staff</h2>
          {staff.isLoading ? null : (
            <span className="visit-chip">{staff.items.length}</span>
          )}
        </div>

        {staff.error ? (
          <p className="form-error" role="alert">
            {staff.error}
          </p>
        ) : staff.isLoading ? (
          <p className="list-note">Loading…</p>
        ) : staff.items.length === 0 ? (
          <p className="list-note">No staff yet. Add your first below.</p>
        ) : (
          <ul className="staff-list">
            {staff.items.map((member) => (
              <li key={member.id} className="staff-row">
                <span className="staff-name">{member.name}</span>
                <span className="staff-email">{member.email}</span>
                <span className={`staff-role staff-role--${member.role}`}>
                  {ROLE_LABEL[member.role] ?? member.role}
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="visit-card">
        <div className="visit-card-head">
          <h2>Add a staff member</h2>
        </div>

        <form className="visit-form" onSubmit={handleSubmit}>
          <div className="visit-form-row">
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

            <label className="field">
              <span className="field-label">Email</span>
              <input
                type="email"
                value={form.email}
                onChange={(event) => updateField("email", event.target.value)}
                autoComplete="off"
                required
                disabled={isSubmitting}
              />
            </label>
          </div>

          <div className="visit-form-row">
            <label className="field">
              <span className="field-label">Temporary password</span>
              <input
                type="text"
                value={form.password}
                onChange={(event) => updateField("password", event.target.value)}
                autoComplete="off"
                required
                minLength={8}
                disabled={isSubmitting}
              />
            </label>

            <label className="field">
              <span className="field-label">Role</span>
              <select
                value={form.role}
                onChange={(event) => updateField("role", event.target.value)}
                disabled={isSubmitting}
              >
                <option value="clinician">Clinician</option>
                <option value="receptionist">Receptionist</option>
              </select>
            </label>
          </div>

          {error ? (
            <p className="form-error" role="alert">
              {error}
            </p>
          ) : null}

          <button type="submit" className="button-primary" disabled={isSubmitting}>
            {isSubmitting ? "Adding…" : "Add staff member"}
          </button>
        </form>
      </section>
    </div>
  );
}
