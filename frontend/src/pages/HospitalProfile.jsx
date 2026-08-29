/**
 * Hospital profile + staff management (faults #4 and #5).
 *
 *   - Every signed-in user sees their hospital's identity (name, city, pincode,
 *     address, logo) and a link to change their own password.
 *   - A hospital_admin additionally gets: an inline edit form, a logo upload, and
 *     the staff roster with "add staff member" (temporary password) and
 *     "reset password" actions.
 *
 * Provisioning is the only way staff accounts are created now — there is no
 * self-service sign-up.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import { api, uploadHospitalLogo } from "../api/client";
import { useAuth } from "../auth/useAuth";
import {
  validateOrgName,
  validatePassword,
  validatePersonName,
  validatePincode,
} from "../utils/validate";

const ROLE_LABELS = {
  receptionist: "Receptionist",
  clinician: "Clinician",
  hospital_admin: "Hospital admin",
};

const EMPTY_STAFF = { name: "", email: "", role: "receptionist", temporary_password: "" };

export default function HospitalProfile() {
  const { user } = useAuth();
  const isAdmin = user.role === "hospital_admin";

  const [hospital, setHospital] = useState(null);
  const [loadError, setLoadError] = useState("");

  const load = useCallback(() => {
    api
      .getHospital()
      .then(setHospital)
      .catch((error) => setLoadError(error.message));
  }, []);

  useEffect(load, [load]);

  if (loadError) {
    return (
      <div className="page visit-page">
        <p className="form-error" role="alert">
          {loadError}
        </p>
      </div>
    );
  }

  if (!hospital) {
    return (
      <div className="page visit-page">
        <p className="visit-loading">Loading hospital…</p>
      </div>
    );
  }

  return (
    <div className="page visit-page">
      <Link className="back-link" to="/">
        ‹ Back to dashboard
      </Link>

      <header className="visit-header">
        <p className="visit-eyebrow">Your hospital</p>
        <h1>{hospital.name}</h1>
        <p className="visit-subtitle">
          {[hospital.city, hospital.pincode].filter(Boolean).join(" · ") || "No location set"}
        </p>
      </header>

      <IdentityCard
        hospital={hospital}
        isAdmin={isAdmin}
        onChange={setHospital}
      />

      {isAdmin ? <StaffCard hospitalId={hospital.id} adminId={user.id} /> : null}

      <section className="visit-card">
        <div className="visit-card-head">
          <h2>Your account</h2>
        </div>
        <p className="visit-note">
          Signed in as {user.name} ({ROLE_LABELS[user.role] ?? user.role}).
        </p>
        <Link className="button-quiet" to="/change-password">
          Change my password
        </Link>
      </section>
    </div>
  );
}

// --------------------------------------------------------------------------- //

function IdentityCard({ hospital, isAdmin, onChange }) {
  const [editing, setEditing] = useState(false);
  const [form, setForm] = useState(hospital);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [logoError, setLogoError] = useState("");

  function update(field, value) {
    setForm((current) => ({ ...current, [field]: value }));
  }

  async function save(event) {
    event.preventDefault();
    setError("");

    const nameError = validateOrgName(form.name);
    const pinError = validatePincode(form.pincode || "");
    if (nameError) return setError(nameError);
    if (pinError) return setError(pinError);

    setBusy(true);
    try {
      const updated = await api.updateHospital({
        name: form.name.trim(),
        address: form.address?.trim() || null,
        city: form.city?.trim() || null,
        pincode: form.pincode.trim(),
      });
      onChange(updated);
      setEditing(false);
    } catch (submitError) {
      setError(submitError.message);
    } finally {
      setBusy(false);
    }
  }

  async function onLogoPick(event) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setLogoError("");
    setBusy(true);
    try {
      onChange(await uploadHospitalLogo(file));
    } catch (uploadError) {
      setLogoError(uploadError.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="visit-card">
      <div className="visit-card-head">
        <h2>Identity</h2>
        {isAdmin && !editing ? (
          <button
            type="button"
            className="button-quiet"
            onClick={() => {
              setForm(hospital);
              setEditing(true);
            }}
          >
            Edit
          </button>
        ) : null}
      </div>

      <div className="hospital-identity">
        {hospital.logo_url ? (
          <img className="hospital-logo" src={hospital.logo_url} alt={`${hospital.name} logo`} />
        ) : (
          <div className="hospital-logo hospital-logo--empty">No logo</div>
        )}

        {isAdmin ? (
          <label className="button-quiet hospital-logo-pick">
            {hospital.logo_url ? "Replace logo" : "Upload logo"}
            <input
              type="file"
              accept=".png,.jpg,.jpeg,.webp,.svg"
              onChange={onLogoPick}
              disabled={busy}
              hidden
            />
          </label>
        ) : null}
      </div>
      {logoError ? (
        <p className="form-error" role="alert">
          {logoError}
        </p>
      ) : null}

      {editing ? (
        <form className="visit-form" onSubmit={save}>
          <label className="field">
            <span className="field-label">Hospital name</span>
            <input value={form.name || ""} onChange={(e) => update("name", e.target.value)} required disabled={busy} />
          </label>
          <div className="visit-form-row">
            <label className="field">
              <span className="field-label">City</span>
              <input value={form.city || ""} onChange={(e) => update("city", e.target.value)} disabled={busy} />
            </label>
            <label className="field">
              <span className="field-label">Pincode</span>
              <input
                value={form.pincode || ""}
                onChange={(e) => update("pincode", e.target.value)}
                inputMode="numeric"
                maxLength={6}
                required
                disabled={busy}
              />
            </label>
          </div>
          <label className="field">
            <span className="field-label">
              Address <span className="field-hint">optional</span>
            </span>
            <input value={form.address || ""} onChange={(e) => update("address", e.target.value)} disabled={busy} />
          </label>

          {error ? (
            <p className="form-error" role="alert">
              {error}
            </p>
          ) : null}

          <div className="visit-form-row">
            <button type="submit" className="button-primary" disabled={busy}>
              {busy ? "Saving…" : "Save changes"}
            </button>
            <button
              type="button"
              className="button-quiet"
              onClick={() => setEditing(false)}
              disabled={busy}
            >
              Cancel
            </button>
          </div>
        </form>
      ) : (
        <dl className="detail-grid">
          <div>
            <dt>City</dt>
            <dd>{hospital.city || "—"}</dd>
          </div>
          <div>
            <dt>Pincode</dt>
            <dd>{hospital.pincode || "—"}</dd>
          </div>
          <div>
            <dt>Address</dt>
            <dd>{hospital.address || "—"}</dd>
          </div>
        </dl>
      )}
    </section>
  );
}

// --------------------------------------------------------------------------- //

function StaffCard({ adminId }) {
  const [staff, setStaff] = useState(null);
  const [error, setError] = useState("");
  const [form, setForm] = useState(EMPTY_STAFF);
  const [busy, setBusy] = useState(false);
  const [justCreated, setJustCreated] = useState(null);

  const load = useCallback(() => {
    api
      .listStaff()
      .then(setStaff)
      .catch((loadError) => setError(loadError.message));
  }, []);

  useEffect(load, [load]);

  const others = useMemo(
    () => (staff || []).filter((member) => member.id !== adminId),
    [staff, adminId],
  );

  function update(field, value) {
    setForm((current) => ({ ...current, [field]: value }));
  }

  async function addStaff(event) {
    event.preventDefault();
    setError("");
    setJustCreated(null);

    const nameError = validatePersonName(form.name);
    const pwError = validatePassword(form.temporary_password);
    if (nameError) return setError(nameError);
    if (pwError) return setError(pwError);

    setBusy(true);
    try {
      const created = await api.createStaff({
        name: form.name.trim(),
        email: form.email.trim(),
        role: form.role,
        temporary_password: form.temporary_password,
      });
      setJustCreated({ email: created.email, password: form.temporary_password });
      setForm(EMPTY_STAFF);
      load();
    } catch (submitError) {
      setError(submitError.message);
    } finally {
      setBusy(false);
    }
  }

  async function resetPassword(member) {
    const next = window.prompt(
      `New temporary password for ${member.name} (min 8 characters):`,
    );
    if (next === null) return;
    const pwError = validatePassword(next);
    if (pwError) {
      setError(pwError);
      return;
    }
    setError("");
    try {
      await api.resetStaffPassword(member.id, { new_password: next });
      setJustCreated({ email: member.email, password: next });
      load();
    } catch (submitError) {
      setError(submitError.message);
    }
  }

  return (
    <section className="visit-card">
      <div className="visit-card-head">
        <h2>Staff</h2>
      </div>

      {error ? (
        <p className="form-error" role="alert">
          {error}
        </p>
      ) : null}

      {justCreated ? (
        <div className="staff-credential">
          <p>
            Share these over a trusted channel — the account must change the
            password on first sign-in.
          </p>
          <p className="mono">
            {justCreated.email}
            <br />
            {justCreated.password}
          </p>
          <button
            type="button"
            className="button-quiet"
            onClick={() =>
              navigator.clipboard?.writeText(
                `${justCreated.email} / ${justCreated.password}`,
              )
            }
          >
            Copy
          </button>
        </div>
      ) : null}

      {staff === null ? (
        <p className="visit-loading">Loading staff…</p>
      ) : (
        <table className="staff-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Email</th>
              <th>Role</th>
              <th>Password</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {others.map((member) => (
              <tr key={member.id}>
                <td>{member.name}</td>
                <td>{member.email}</td>
                <td>{ROLE_LABELS[member.role] ?? member.role}</td>
                <td>{member.must_change_password ? "Temporary — not yet set" : "Set"}</td>
                <td>
                  <button
                    type="button"
                    className="button-quiet"
                    onClick={() => resetPassword(member)}
                  >
                    Reset password
                  </button>
                </td>
              </tr>
            ))}
            {others.length === 0 ? (
              <tr>
                <td colSpan={5} className="list-note">
                  No staff yet. Add a receptionist or clinician below.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      )}

      <form className="visit-form" onSubmit={addStaff}>
        <h3>Add staff member</h3>
        <div className="visit-form-row">
          <label className="field">
            <span className="field-label">Name</span>
            <input value={form.name} onChange={(e) => update("name", e.target.value)} required disabled={busy} />
          </label>
          <label className="field">
            <span className="field-label">Email</span>
            <input type="email" value={form.email} onChange={(e) => update("email", e.target.value)} required disabled={busy} />
          </label>
        </div>
        <div className="visit-form-row">
          <label className="field">
            <span className="field-label">Role</span>
            <select value={form.role} onChange={(e) => update("role", e.target.value)} disabled={busy}>
              <option value="receptionist">Receptionist</option>
              <option value="clinician">Clinician</option>
            </select>
          </label>
          <label className="field">
            <span className="field-label">Temporary password</span>
            <input
              type="password"
              value={form.temporary_password}
              onChange={(e) => update("temporary_password", e.target.value)}
              autoComplete="new-password"
              required
              disabled={busy}
            />
          </label>
        </div>
        <button type="submit" className="button-primary" disabled={busy}>
          {busy ? "Adding…" : "Add staff member"}
        </button>
      </form>
    </section>
  );
}
