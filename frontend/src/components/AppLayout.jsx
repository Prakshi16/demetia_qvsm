/**
 * Chrome around every signed-in screen: the persistent nav bar §6 asks for
 * (a Home target that always returns to the dashboard) plus who's signed in and,
 * now, which hospital they belong to (fault #4 — staff shouldn't have to
 * remember). The hospital name + logo are fetched live rather than carried in the
 * JWT so an admin's edits show up without re-login.
 *
 * Screens render into <Outlet />. Per §6 each screen owns its own *fixed* back
 * target — a "< Back to Patient Profile"-style link inside the screen — rather
 * than relying on browser history, so don't add a global back button here.
 */
import { useEffect, useState } from "react";
import { Link, Outlet } from "react-router-dom";

import { api } from "../api/client";
import { useAuth } from "../auth/useAuth";

const ROLE_LABELS = {
  receptionist: "Receptionist",
  clinician: "Clinician",
  hospital_admin: "Hospital admin",
};

function initials(name) {
  return (name || "?")
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((word) => word[0].toUpperCase())
    .join("");
}

export default function AppLayout() {
  const { user, signOut } = useAuth();
  const [hospital, setHospital] = useState(null);

  useEffect(() => {
    let cancelled = false;
    api
      .getHospital()
      .then((data) => {
        if (!cancelled) setHospital(data);
      })
      .catch(() => {
        // A nav-bar label is not worth surfacing an error for; leave it blank.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="app-shell">
      <header className="app-nav">
        <div className="nav-left">
          <Link to="/" className="nav-home" aria-label="Go to dashboard">
            <span className="nav-home-icon" aria-hidden="true">
              ⌂
            </span>
            <span className="nav-brand">Cortex Health Portal</span>
          </Link>

          {hospital ? (
            <Link to="/hospital" className="nav-hospital" title="Hospital profile">
              {hospital.logo_url ? (
                <img className="nav-hospital-logo" src={hospital.logo_url} alt="" />
              ) : (
                <span className="nav-hospital-logo nav-hospital-logo--fallback" aria-hidden="true">
                  {initials(hospital.name)}
                </span>
              )}
              <span className="nav-hospital-name">
                {hospital.name}
                {hospital.city ? <span className="nav-hospital-city"> · {hospital.city}</span> : null}
              </span>
            </Link>
          ) : null}
        </div>

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
