/**
 * Chrome around every signed-in screen: the persistent nav bar §6 asks for
 * (a Home target that always returns to the dashboard) plus who's signed in.
 *
 * Screens render into <Outlet />. Per §6 each screen owns its own *fixed* back
 * target — a "< Back to Patient Profile"-style link inside the screen — rather
 * than relying on browser history, so don't add a global back button here.
 */
import { Link, Outlet } from "react-router-dom";

import { useAuth } from "../auth/useAuth";

const ROLE_LABELS = {
  receptionist: "Receptionist",
  clinician: "Clinician",
  hospital_admin: "Hospital admin",
};

export default function AppLayout() {
  const { user, signOut } = useAuth();

  return (
    <div className="app-shell">
      <header className="app-nav">
        <Link to="/" className="nav-home" aria-label="Go to dashboard">
          <span className="nav-home-icon" aria-hidden="true">
            ⌂
          </span>
          <span className="nav-brand">Cortex Health Portal</span>
        </Link>

        <div className="nav-user">
          <span className="nav-user-name">{user.name}</span>
          <span className="nav-user-role">{ROLE_LABELS[user.role] ?? user.role}</span>
          <button type="button" className="button-quiet" onClick={signOut}>
            Sign out
          </button>
        </div>
      </header>

      <main className="app-main">
        <Outlet />
      </main>
    </div>
  );
}
