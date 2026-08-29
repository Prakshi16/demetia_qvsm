/**
 * Hospital profile (fault #4).
 *
 * Every signed-in user sees their hospital's identity (name, city, pincode,
 * address, logo) and a link to change their own password. A hospital_admin also
 * gets an inline edit form and a logo upload. Staff management lives on the
 * admin's dashboard, not here.
 */
import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { api, uploadHospitalLogo } from "../api/client";
import { useAuth } from "../auth/useAuth";
import { validateOrgName, validatePincode } from "../utils/validate";

const ROLE_LABELS = {
  receptionist: "Receptionist",
  clinician: "Clinician",
  hospital_admin: "Hospital admin",
};

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
