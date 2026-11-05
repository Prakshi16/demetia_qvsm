/**
 * Chrome around every signed-in screen: the persistent nav bar §6 asks for
 * (a Home target that always returns to the dashboard) plus who's signed in and
 * which hospital they belong to (fault #4 — staff shouldn't have to remember).
 * The hospital name + logo are fetched live rather than carried in the JWT so
 * an admin's edits show up without re-login.
 *
 * Responsive: under 720px the user block folds into a menu behind a button.
 * The theme toggle flips light/dark (the choice persists; until one is made the
 * app follows the OS).
 *
 * Screens render into <Outlet />. Per §6 each screen owns its own *fixed* back
 * target — a "< Back to Patient Profile"-style link inside the screen — rather
 * than relying on browser history, so don't add a global back button here.
 */
import { useEffect, useState } from "react";
import { Link, Outlet, useLocation } from "react-router-dom";

import { api } from "../api/client";
import { useAuth } from "../auth/useAuth";
import { useTheme } from "../theme/useTheme";

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

function SunIcon() {
  return (
    <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeLinecap="round" aria-hidden="true">
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
    </svg>
  );
}

function MenuIcon({ open }) {
  return (
    <svg className="icon icon--sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeLinecap="round" aria-hidden="true">
      {open ? <path d="M6 6l12 12M18 6L6 18" /> : <path d="M4 7h16M4 12h16M4 17h16" />}
    </svg>
  );
}

export function ThemeToggle() {
  const { resolved, setTheme } = useTheme();
  const next = resolved === "dark" ? "light" : "dark";
  return (
    <button
      type="button"
      className="icon-button"
      onClick={() => setTheme(next)}
      aria-label={`Switch to ${next} theme`}
      title={`Switch to ${next} theme`}
    >
      {resolved === "dark" ? <SunIcon /> : <MoonIcon />}
    </button>
  );
}

export default function AppLayout() {
  const { user, signOut } = useAuth();
  const [hospital, setHospital] = useState(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();

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

  // Close the mobile menu whenever the route changes.
  useEffect(() => {
    setMenuOpen(false);
  }, [location.pathname]);

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

        <div style={{ display: "flex", alignItems: "center", gap: "var(--space-xs)" }}>
          <ThemeToggle />
          <button
            type="button"
            className="icon-button nav-menu-button"
            aria-expanded={menuOpen}
            aria-controls="nav-user-menu"
            aria-label={menuOpen ? "Close account menu" : "Open account menu"}
            onClick={() => setMenuOpen((open) => !open)}
          >
            <MenuIcon open={menuOpen} />
          </button>
          <div id="nav-user-menu" className={`nav-user${menuOpen ? " is-open" : ""}`}>
            <span className="nav-user-name">{user.name}</span>
            <span className="nav-user-role">{ROLE_LABELS[user.role] ?? user.role}</span>
            <button type="button" className="button-quiet" onClick={signOut}>
              Sign out
            </button>
          </div>
        </div>
      </header>

      <main className="app-main">
        <Outlet />
      </main>

      <footer className="app-footer no-print">
        Cortex Health Portal · decision-support research tool, not a diagnostic device ·{" "}
        <Link to="/design-system">Design system</Link>
      </footer>
    </div>
  );
}
