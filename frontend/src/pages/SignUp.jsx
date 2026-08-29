/**
 * Screen 1b (§6) — Sign up.
 *
 * This screen now does one thing: register a new hospital and become its
 * hospital_admin (POST /auth/register-hospital). Staff no longer self-join —
 * their accounts are provisioned by their hospital's admin (fault #5), so the
 * old "join an existing hospital" flow is gone.
 *
 * Pincode is required and is what tells branches of the same chain apart
 * (fault #7): "Manipal Hospital" at 560017 and at 560066 are two hospitals;
 * "Manipal Hospital" registered twice at 560017 is rejected.
 */
import { useState } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";

import { useAuth } from "../auth/useAuth";
import {
  validateOrgName,
  validatePassword,
  validatePersonName,
  validatePincode,
} from "../utils/validate";

export default function SignUp() {
  const { registerHospital, isSignedIn } = useAuth();
  const navigate = useNavigate();

  const [hospitalName, setHospitalName] = useState("");
  const [pincode, setPincode] = useState("");
  const [city, setCity] = useState("");
  const [address, setAddress] = useState("");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  if (isSignedIn) return <Navigate to="/" replace />;

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");

    const firstError =
      validateOrgName(hospitalName) ||
      validatePincode(pincode) ||
      validatePersonName(name) ||
      validatePassword(password);
    if (firstError) return setError(firstError);

    setIsSubmitting(true);
    try {
      await registerHospital({
        hospital_name: hospitalName.trim(),
        pincode: pincode.trim(),
        city: city.trim() || null,
        address: address.trim() || null,
        admin_name: name.trim(),
        admin_email: email.trim(),
        password,
      });
      navigate("/", { replace: true });
    } catch (submitError) {
      setError(submitError.message);
      setIsSubmitting(false);
    }
  }

  return (
    <div className="auth-screen">
      <div className="auth-card auth-card-wide">
        <header className="auth-header">
          <p className="auth-brand">Cortex Health Portal</p>
          <h1>Register a hospital</h1>
          <p className="auth-subtitle">
            This creates the hospital and makes you its admin. Staff don't sign up
            here — ask your hospital's admin to create your account.
          </p>
        </header>

        <form className="auth-form" onSubmit={handleSubmit}>
          <label className="field">
            <span className="field-label">Hospital name</span>
            <input
              value={hospitalName}
              onChange={(event) => setHospitalName(event.target.value)}
              required
              disabled={isSubmitting}
            />
          </label>

          <div className="visit-form-row">
            <label className="field">
              <span className="field-label">Pincode</span>
              <input
                value={pincode}
                onChange={(event) => setPincode(event.target.value)}
                inputMode="numeric"
                maxLength={6}
                placeholder="6 digits"
                required
                disabled={isSubmitting}
              />
            </label>
            <label className="field">
              <span className="field-label">
                City <span className="field-hint">optional</span>
              </span>
              <input
                value={city}
                onChange={(event) => setCity(event.target.value)}
                disabled={isSubmitting}
              />
            </label>
          </div>

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
            {isSubmitting ? "Creating account…" : "Register hospital"}
          </button>
        </form>

        <p className="auth-switch">
          Already have an account? <Link to="/signin">Sign in</Link>
        </p>
      </div>
    </div>
  );
}
