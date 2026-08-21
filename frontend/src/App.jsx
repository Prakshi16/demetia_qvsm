/**
 * Route table for the whole app.
 *
 * Two groups: the public auth screens, and everything else behind
 * <ProtectedRoute>, rendered inside <AppLayout /> so every signed-in screen gets
 * the same nav bar.
 *
 * ---------------------------------------------------------------------------
 * GOVIND — this is where your three screens plug in (§6 screens 2, 4, 7).
 * Drop your components into src/pages/ and add them to the protected group
 * below; the commented routes are the paths the rest of the app already links to,
 * so please keep these exact paths:
 *
 *     <Route path="/patients/:patientId" element={<PatientProfile />} />
 *     <Route path="/visits/:visitId"     element={<VisitDetail />} />
 *
 * and swap <DashboardPlaceholder /> for your <Dashboard />.
 *
 * Notes so nothing surprises you:
 *   - You do NOT need to fetch or pass the token. Use `api` from src/api/client.js
 *     and it attaches the JWT itself; an expired token signs the user out.
 *   - `useAuth()` from src/auth/AuthContext gives you `user` ({ id, hospital_id,
 *     name, email, role }) for the role-aware bits of the dashboard.
 *   - Screens that only one role should reach take a roles prop, e.g.
 *     <ProtectedRoute roles={["clinician"]}>. That's UI convenience only —
 *     the real rule is enforced server-side.
 * ---------------------------------------------------------------------------
 */
import { Navigate, Route, Routes } from "react-router-dom";

import AppLayout from "./components/AppLayout";
import ProtectedRoute from "./auth/ProtectedRoute";
import DashboardPlaceholder from "./pages/DashboardPlaceholder";
import NewVisitScreening from "./pages/NewVisitScreening";
import SignIn from "./pages/SignIn";
import SignUp from "./pages/SignUp";
import "./App.css";

export default function App() {
  return (
    <Routes>
      {/* Public */}
      <Route path="/signin" element={<SignIn />} />
      <Route path="/signup" element={<SignUp />} />

      {/* Signed in */}
      <Route
        element={
          <ProtectedRoute>
            <AppLayout />
          </ProtectedRoute>
        }
      >
        <Route path="/" element={<DashboardPlaceholder />} />

        {/* §6 screen 5. Deliberately NOT /patients/:patientId or /visits/:visitId
            — those two paths are reserved for Govind's screens, which other
            screens (including this one's back link) already link to.
            ?visitId=... resumes an awaiting_uploads visit (Product Rule 2A). */}
        <Route
          path="/patients/:patientId/new-visit/screening"
          element={<NewVisitScreening />}
        />

        {/* GOVIND: add /patients/:patientId and /visits/:visitId here. */}
      </Route>

      {/* Unknown path -> home, which itself redirects to /signin if signed out. */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
