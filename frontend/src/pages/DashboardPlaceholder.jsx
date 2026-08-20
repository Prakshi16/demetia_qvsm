/**
 * TEMPORARY — replace with Govind's Dashboard.jsx (`feature/dashboard-ui`).
 *
 * This exists only so the router has a landing route and the auth flow can be
 * demonstrated end to end. It deliberately implements none of the real dashboard
 * (§6 screen 2: role-aware search bars, the deduped patient list, the
 * receptionist's "Incomplete visits" section, "+ Register New Patient").
 *
 * To swap it in: drop Dashboard.jsx into src/pages/, change the import and the
 * element in App.jsx, and delete this file.
 *
 * It does confirm one useful thing: that the stored JWT is accepted by the API,
 * since /dashboard is an authenticated endpoint.
 */
import { useEffect, useState } from "react";

import { api } from "../api/client";
import { useAuth } from "../auth/useAuth";

export default function DashboardPlaceholder() {
  const { user } = useAuth();
  const [status, setStatus] = useState("checking");
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    api
      .getDashboard()
      .then(() => {
        if (!cancelled) setStatus("ok");
      })
      .catch((requestError) => {
        if (cancelled) return;
        setStatus("error");
        setError(requestError.message);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="page">
      <div className="placeholder-card">
        <p className="placeholder-tag">Placeholder</p>
        <h1>Signed in as {user.name}</h1>
        <p>
          Authentication works. This screen is a stand-in for the real dashboard —
          Govind&apos;s <code>Dashboard.jsx</code> replaces it on{" "}
          <code>feature/dashboard-ui</code>.
        </p>

        <dl className="detail-grid">
          <div>
            <dt>Role</dt>
            <dd>{user.role}</dd>
          </div>
          <div>
            <dt>Hospital ID</dt>
            <dd className="mono">{user.hospital_id}</dd>
          </div>
          <div>
            <dt>Authenticated API call</dt>
            <dd>
              {status === "checking" ? "Calling GET /dashboard…" : null}
              {status === "ok" ? "GET /dashboard returned 200 — token accepted" : null}
              {status === "error" ? <span className="form-error">{error}</span> : null}
            </dd>
          </div>
        </dl>
      </div>
    </div>
  );
}
