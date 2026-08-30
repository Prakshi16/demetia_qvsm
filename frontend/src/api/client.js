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

/** "" -> "", "ada" -> "?search=ada". Keeps the query out of the call sites. */
function searchQuery(search) {
  const trimmed = (search ?? "").trim();
  return trimmed === "" ? "" : `?search=${encodeURIComponent(trimmed)}`;
}

export const api = {
  // --- auth (§5) ---------------------------------------------------------
  // All three auth calls return { token, user } so the caller is logged in
  // immediately — there is no separate login step after registering.
  listHospitals: () => request("/hospitals", { auth: false }),

  registerHospital: (body) =>
    request("/auth/register-hospital", { method: "POST", body, auth: false }),

  login: (body) => request("/auth/login", { method: "POST", body, auth: false }),

  // Authenticated. Returns a fresh { token, user } with must_change_password
  // cleared, so the caller must re-store the session.
  changePassword: (body) =>
    request("/auth/change-password", { method: "POST", body }),

  // --- hospital (fault #4) --------------------------------------------------
  getHospital: () => request("/hospital"),
  updateHospital: (body) => request("/hospital", { method: "PATCH", body }),

  // --- staff management, hospital_admin only (§5) ------------------------
  // The admin's dashboard: the clinicians and receptionists at their own
  // hospital. addStaff takes { name, temporary_password, role } — the login
  // email is derived server-side from the name + the hospital's fixed email
  // domain, never sent. The account lands in the caller's hospital (never one
  // named in the body) and the new user must change the password on first
  // sign-in.
  listStaff: () => request("/staff"),
  addStaff: (body) => request("/staff", { method: "POST", body }),
  resetStaffPassword: (id, body) =>
    request(`/staff/${id}/reset-password`, { method: "POST", body }),

  // --- everything else ---------------------------------------------------
  // Add the patient/visit/dashboard calls here as the screens that need them
  // get built, so no component ever calls fetch() directly.
  // Role-aware (§5): the same deduped, most-recent-visit-first patient list for
  // both roles. The specialised queues below are separate endpoints, not filters
  // of this one, and never overlap with each other.
  getDashboard: (search = "") => request(`/dashboard${searchQuery(search)}`),

  // Clinician only. Patients whose latest visit is a screening sitting in
  // pending_review — complete data, no diagnosis yet (Product Rule 3).
  getPendingReview: (search = "") =>
    request(`/patients/pending-review${searchQuery(search)}`),

  // Receptionist only. Patients whose latest visit is still awaiting_uploads,
  // i.e. resumable half-finished screenings (Product Rule 2A).
  getIncompleteVisits: (search = "") =>
    request(`/patients/incomplete-visits${searchQuery(search)}`),

  // --- patients (§5) ------------------------------------------------------
  getPatient: (patientId) => request(`/patients/${patientId}`),

  // Creates the patient only — a visit is a separate, later step (§6 screen 3).
  createPatient: (body) => request("/patients", { method: "POST", body }),

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

  // Clinician only. Valid on a screening in pending_review, or on one reviewed
  // earlier the same UTC day (Rule 5) — the server rejects anything else, so a
  // 400 here is the rule working, not a bug to route around.
  saveDiagnosis: (visitId, body) =>
    request(`/visits/${visitId}/diagnosis`, { method: "POST", body }),
};

/**
 * Hospital logo upload. Multipart FormData can't go through `request()` (same
 * reason as the MRI/speech uploads), so it builds the URL + auth header itself.
 * Returns the updated HospitalDetailOut.
 */
export async function uploadHospitalLogo(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${BASE_URL}/hospital/logo`, {
    method: "POST",
    headers: { Authorization: `Bearer ${getToken()}` },
    body: formData,
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new ApiError(readErrorMessage(payload, response.status), response.status);
  }
  return payload;
}

export { request };
