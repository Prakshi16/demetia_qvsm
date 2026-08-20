/**
 * Gate for every screen that isn't Sign in / Sign up.
 *
 * `roles` is optional; pass it for screens only one role should reach, e.g.
 *   <ProtectedRoute roles={["clinician"]}>...
 *
 * This is convenience and clarity only — it is NOT security. Every role rule is
 * enforced server-side (deps.py require_clinician / require_receptionist), because
 * anything the browser decides can be bypassed by editing localStorage.
 */
import { Navigate, useLocation } from "react-router-dom";

import { useAuth } from "./useAuth";

export default function ProtectedRoute({ roles, children }) {
  const { user, isSignedIn } = useAuth();
  const location = useLocation();

  if (!isSignedIn) {
    // Remember where they were headed so sign-in can send them back.
    return <Navigate to="/signin" state={{ from: location.pathname }} replace />;
  }

  if (roles && !roles.includes(user.role)) {
    return <Navigate to="/" replace />;
  }

  return children;
}
