/**
 * Screen 1b (§6) — Sign up, which is two different flows behind one screen:
 *
 *   "Register a new hospital"  -> POST /auth/register-hospital, creates the
 *                                 hospital plus its first user as hospital_admin.
 *   "Join an existing hospital" -> POST /auth/register-staff, pick the hospital
 *                                 from a searchable list, choose receptionist or
 *                                 clinician. No admin-approval step (§6).
 *
 * The role dropdown appears only on the join path: a hospital's first account is
 * always the admin, and `register-staff` rejects `hospital_admin` by schema.
 */
import { useEffect, useMemo, useState } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";

import { api } from "../api/client";
import { useAuth } from "../auth/useAuth";

const MODE = {
  NEW_HOSPITAL: "new-hospital",
  JOIN: "join",
};

export default function SignUp() {
  const { registerHospital, registerStaff, isSignedIn } = useAuth();
  const navigate = useNavigate();

  const [mode, setMode] = useState(MODE.JOIN);
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Shared account fields.
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  // "Register a new hospital" only.
  const [hospitalName, setHospitalName] = useState("");
  const [address, setAddress] = useState("");

  // "Join an existing hospital" only.
  const [hospitals, setHospitals] = useState([]);
  const [hospitalsError, setHospitalsError] = useState("");
  const [isLoadingHospitals, setIsLoadingHospitals] = useState(false);
  const [hospitalSearch, setHospitalSearch] = useState("");
  const [hospitalId, setHospitalId] = useState("");
  const [role, setRole] = useState("receptionist");

  // Load the picker list lazily — only when the join path is actually shown.
  useEffect(() => {
    if (mode !== MODE.JOIN || hospitals.length > 0) return;

    let cancelled = false;
    setIsLoadingHospitals(true);
    setHospitalsError("");

    api
      .listHospitals()
      .then((list) => {
        if (!cancelled) setHospitals(list);
      })
      .catch((loadError) => {
        if (!cancelled) setHospitalsError(loadError.message);
      })
      .finally(() => {
        if (!cancelled) setIsLoadingHospitals(false);
      });

    return () => {
      cancelled = true;
    };
  }, [mode, hospitals.length]);

  const visibleHospitals = useMemo(() => {
    const query = hospitalSearch.trim().toLowerCase();
    if (!query) return hospitals;
    return hospitals.filter((hospital) => hospital.name.toLowerCase().includes(query));
  }, [hospitals, hospitalSearch]);

  if (isSignedIn) return <Navigate to="/" replace />;

  function switchMode(nextMode) {
    setMode(nextMode);
    setError("");
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");

    if (mode === MODE.JOIN && !hospitalId) {
      setError("Select the hospital you're joining.");
      return;
    }

    setIsSubmitting(true);

    try {
      if (mode === MODE.NEW_HOSPITAL) {
        await registerHospital({
          hospital_name: hospitalName.trim(),
          address: address.trim() || null,
          admin_name: name.trim(),
          admin_email: email.trim(),
          password,
        });
      } else {
        await registerStaff({
          hospital_id: hospitalId,
          name: name.trim(),
          email: email.trim(),
          password,
          role,
        });
      }
      // Every auth endpoint returns a token, so registering signs you straight in.
      navigate("/", { replace: true });
    } catch (submitError) {
      setError(submitError.message);
      setIsSubmitting(false);
    }
  }

  const isNewHospital = mode === MODE.NEW_HOSPITAL;

  return (
    <div className="auth-screen">
      <div className="auth-card auth-card-wide">
        <header className="auth-header">
          <p className="auth-brand">Cortex Health Portal</p>
          <h1>Create an account</h1>
        </header>

        <div className="mode-toggle" role="tablist" aria-label="Sign-up type">
          <button
            type="button"
            role="tab"
            aria-selected={!isNewHospital}
            className={!isNewHospital ? "mode-option is-active" : "mode-option"}
            onClick={() => switchMode(MODE.JOIN)}
            disabled={isSubmitting}
          >
            <strong>Join an existing hospital</strong>
            <span>You're staff at a hospital already using the portal</span>
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={isNewHospital}
            className={isNewHospital ? "mode-option is-active" : "mode-option"}
            onClick={() => switchMode(MODE.NEW_HOSPITAL)}
            disabled={isSubmitting}
          >
            <strong>Register a new hospital</strong>
            <span>Sets up the hospital and makes you its admin</span>
          </button>
        </div>

        <form className="auth-form" onSubmit={handleSubmit}>
          {isNewHospital ? (
            <>
              <label className="field">
                <span className="field-label">Hospital name</span>
                <input
                  value={hospitalName}
                  onChange={(event) => setHospitalName(event.target.value)}
                  required
                  disabled={isSubmitting}
                />
              </label>

              <label className="field">
                <span className="field-label">
                  Address <span className="field-hint">optional</span>
                </span>
                <input
                  value={address}
                  onChange={(event) => setAddress(event.target.value)}
                  disabled={isSubmitting}
                />
              </label>
            </>
          ) : (
            <div className="field">
              <span className="field-label">Hospital</span>

              {hospitalsError ? (
                <p className="form-error" role="alert">
                  Couldn't load hospitals — {hospitalsError}
                </p>
              ) : null}

              <input
                type="search"
                placeholder="Search hospitals…"
                value={hospitalSearch}
                onChange={(event) => setHospitalSearch(event.target.value)}
                disabled={isSubmitting || isLoadingHospitals}
              />

              <div className="hospital-list">
                {isLoadingHospitals ? <p className="list-note">Loading hospitals…</p> : null}

                {!isLoadingHospitals && hospitals.length === 0 && !hospitalsError ? (
                  <p className="list-note">
                    No hospitals registered yet — use “Register a new hospital” instead.
                  </p>
                ) : null}

                {!isLoadingHospitals && hospitals.length > 0 && visibleHospitals.length === 0 ? (
                  <p className="list-note">No hospital matches “{hospitalSearch}”.</p>
                ) : null}

                {visibleHospitals.map((hospital) => (
                  <button
                    type="button"
                    key={hospital.id}
                    className={
                      hospital.id === hospitalId ? "hospital-option is-active" : "hospital-option"
                    }
                    onClick={() => setHospitalId(hospital.id)}
                    disabled={isSubmitting}
                  >
                    {hospital.name}
                  </button>
                ))}
              </div>

              <label className="field">
                <span className="field-label">Your role</span>
                <select
                  value={role}
                  onChange={(event) => setRole(event.target.value)}
                  disabled={isSubmitting}
                >
                  <option value="receptionist">Receptionist</option>
                  <option value="clinician">Clinician</option>
                </select>
              </label>
            </div>
          )}

          <label className="field">
            <span className="field-label">Your name</span>
            <input
              value={name}
              onChange={(event) => setName(event.target.value)}
              autoComplete="name"
              required
              disabled={isSubmitting}
            />
          </label>

          <label className="field">
            <span className="field-label">Email</span>
            <input
              type="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              autoComplete="email"
              required
              disabled={isSubmitting}
            />
          </label>

          <label className="field">
            <span className="field-label">Password</span>
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              autoComplete="new-password"
              required
              disabled={isSubmitting}
            />
          </label>

          {error ? (
            <p className="form-error" role="alert">
              {error}
            </p>
          ) : null}

          <button type="submit" className="button-primary" disabled={isSubmitting}>
            {isSubmitting
              ? "Creating account…"
              : isNewHospital
                ? "Register hospital"
                : "Create account"}
          </button>
        </form>

        <p className="auth-switch">
          Already have an account? <Link to="/signin">Sign in</Link>
        </p>
      </div>
    </div>
  );
}
