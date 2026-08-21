/**
 * The single place the frontend talks to the backend.
 *
 * Everything goes through `request()` so three things are guaranteed everywhere:
 *   1. the JWT is attached (the API requires `Authorization: Bearer <jwt>` on
 *      every endpoint except GET /hospitals and the three /auth routes),
 *   2. FastAPI error bodies are turned into a readable message,
 *   3. an expired token logs the user out instead of failing silently.
 *
 * Requests use relative paths; vite.config.js proxies /api to the backend in dev.
 */

/**
 * Where the API lives.
 *
 * Unset (the normal case in local dev) -> a relative path, which vite.config.js
 * proxies to localhost:8000. Nothing to configure to run the app locally.
 *
 * Set via VITE_API_BASE_URL -> an absolute origin, which is what a deployed
 * frontend needs: the Vite proxy is a dev-server feature and does not exist in a
 * production build, so once the frontend is on Vercel/Netlify and the backend on
 * Render they are different origins and relative paths would 404.
 *
 * Exported because file uploads can't use this client (multipart FormData can't
 * go through a JSON wrapper) and must build the same URL themselves.
 */
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api/v1";

const BASE_URL = API_BASE_URL;

// Storage keys. Anything reading the token directly (e.g. a file-upload
// component that can't use this client because it posts FormData) must use
// TOKEN_KEY rather than hardcoding the string.
export const TOKEN_KEY = "token";
export const USER_KEY = "user";

// Fired when the API rejects our token. The 24h JWT has no refresh token
// (a deliberate Phase 2 simplification), so expiry is a real, expected event
// rather than an edge case — AuthProvider listens for this and signs the user out.
export const AUTH_EXPIRED_EVENT = "auth:expired";

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

/**
 * FastAPI puts errors in `detail`, but the shape varies: a string for our own
 * HTTPExceptions, an array of objects for 422 validation failures. Rendering the
 * array directly would show "[object Object]" to the user.
 */
function readErrorMessage(payload, status) {
  const detail = payload?.detail;

  if (typeof detail === "string") return detail;

  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        const field = Array.isArray(item?.loc) ? item.loc[item.loc.length - 1] : null;
        return field ? `${field}: ${item.msg}` : item?.msg;
      })
      .filter(Boolean);
    if (messages.length > 0) return messages.join(", ");
  }

  return `Request failed (${status})`;
}

async function request(path, { method = "GET", body, auth = true } = {}) {
  const headers = {};

  if (body !== undefined) headers["Content-Type"] = "application/json";

  const token = getToken();
  if (auth && token) headers.Authorization = `Bearer ${token}`;

  let response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    // fetch only rejects on network-level failure, which in practice means the
    // backend isn't running — worth saying so plainly rather than "failed".
    throw new ApiError("Cannot reach the server. Is the backend running?", 0);
  }

  // 204 and other empty bodies would make .json() throw.
  const payload = await response.json().catch(() => ({}));

  if (!response.ok) {
    if (response.status === 401 && auth && token) {
      localStorage.removeItem(TOKEN_KEY);
      localStorage.removeItem(USER_KEY);
      window.dispatchEvent(new Event(AUTH_EXPIRED_EVENT));
    }
    throw new ApiError(readErrorMessage(payload, response.status), response.status);
  }

  return payload;
}

export const api = {
  // --- auth (§5) ---------------------------------------------------------
  // All three auth calls return { token, user } so the caller is logged in
  // immediately — there is no separate login step after registering.
  listHospitals: () => request("/hospitals", { auth: false }),

  registerHospital: (body) =>
    request("/auth/register-hospital", { method: "POST", body, auth: false }),

  registerStaff: (body) =>
    request("/auth/register-staff", { method: "POST", body, auth: false }),

  login: (body) => request("/auth/login", { method: "POST", body, auth: false }),

  // --- everything else ---------------------------------------------------
  // Add the patient/visit/dashboard calls here as the screens that need them
  // get built, so no component ever calls fetch() directly.
  getDashboard: () => request("/dashboard"),

  // --- patients (§5) ------------------------------------------------------
  getPatient: (patientId) => request(`/patients/${patientId}`),

  // Returns { visit_type, reason } per the §4 decision logic. force_screening
  // is the manual override: screening is always allowed, follow-up isn't.
  getNextVisitType: (patientId, forceScreening = false) =>
    request(`/patients/${patientId}/next-visit-type?force_screening=${forceScreening}`),

  // --- visits (§5) --------------------------------------------------------
  // A screening visit comes back as awaiting_uploads with both modalities idle;
  // the MRI/speech files are posted separately to the upload endpoints, which
  // are the one place that can't use this client (multipart FormData).
  createVisit: (body) => request("/visits", { method: "POST", body }),

  getVisit: (visitId) => request(`/visits/${visitId}`),
};

export { request };
