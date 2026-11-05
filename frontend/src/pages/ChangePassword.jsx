/**
 * Change your own password.
 *
 * Two audiences behind one screen:
 *   - a just-provisioned staff member, sent here by ProtectedRoute and unable to
 *     reach anything else until they replace the admin's temporary password;
 *   - anyone who follows the "Change my password" link from the hospital profile.
 *
 * All three inputs are masked. On success the AuthContext re-stores the fresh
 * session (must_change_password cleared) and we land on the dashboard.
 */
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { useAuth } from "../auth/useAuth";
import { validatePassword } from "../utils/validate";
import FieldError from "../components/ui/FieldError";

export default function ChangePassword() {
  const { user, changePassword } = useAuth();
  const navigate = useNavigate();

  const forced = Boolean(user?.must_change_password);

  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [touched, setTouched] = useState({});
  const [submitAttempted, setSubmitAttempted] = useState(false);

  const values = { newPassword, confirm };
  const fieldErrors = {
    newPassword:
      validatePassword(newPassword) ||
      (currentPassword && newPassword === currentPassword
        ? "The new password must be different from the current one."
        : null),
    confirm: confirm !== newPassword ? "The new passwords don't match." : null,
  };
  // Errors show once a field has been left with something in it, or on submit.
  const shown = (field) =>
    ((submitAttempted || (touched[field] && values[field] !== "")) && fieldErrors[field]) || null;
  const touch = (field) => () => setTouched((current) => ({ ...current, [field]: true }));

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setSubmitAttempted(true);
    if (fieldErrors.newPassword || fieldErrors.confirm) return;

    setIsSubmitting(true);
    try {
      await changePassword({
        current_password: currentPassword,
        new_password: newPassword,
      });
      navigate("/", { replace: true });
    } catch (submitError) {
      setError(submitError.message);
      setIsSubmitting(false);
    }
  }

  return (
    <div className="page visit-page">
      {!forced ? (
        <Link className="back-link" to="/hospital">
          ‹ Back to hospital profile
        </Link>
      ) : null}

      <header className="visit-header">
        <p className="visit-eyebrow">Account</p>
        <h1>{forced ? "Set your password" : "Change your password"}</h1>
        <p className="visit-subtitle">
          {forced
            ? "Your account was created with a temporary password. Choose your own to continue."
            : "Enter your current password and pick a new one."}
        </p>
      </header>

      <form className="visit-form" onSubmit={handleSubmit} style={{ maxWidth: "var(--width-form)" }}>
        <label className="field">
          <span className="field-label">
            {forced ? "Temporary password" : "Current password"}
          </span>
          <input
            type="password"
            value={currentPassword}
            onChange={(event) => setCurrentPassword(event.target.value)}
            autoComplete="current-password"
            required
            disabled={isSubmitting}
          />
        </label>

        <label className="field">
          <span className="field-label">New password</span>
          <input
            type="password"
            value={newPassword}
            onChange={(event) => setNewPassword(event.target.value)}
            onBlur={touch("newPassword")}
            aria-invalid={shown("newPassword") ? "true" : undefined}
            autoComplete="new-password"
            required
            disabled={isSubmitting}
          />
          <FieldError message={shown("newPassword")} />
        </label>

        <label className="field">
          <span className="field-label">Confirm new password</span>
          <input
            type="password"
            value={confirm}
            onChange={(event) => setConfirm(event.target.value)}
            onBlur={touch("confirm")}
            aria-invalid={shown("confirm") ? "true" : undefined}
            autoComplete="new-password"
            required
            disabled={isSubmitting}
          />
          <FieldError message={shown("confirm")} />
        </label>

        {error ? (
          <p className="form-error" role="alert">
            {error}
          </p>
        ) : null}

        <button type="submit" className="button-primary" disabled={isSubmitting}>
          {isSubmitting ? "Saving…" : "Save password"}
        </button>
      </form>
    </div>
  );
}
