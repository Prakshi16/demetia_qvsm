/**
 * Route table for the whole app.
 *
 * Two groups: the public auth screens, and everything else behind
 * <ProtectedRoute>, rendered inside <AppLayout /> so every signed-in screen gets
 * the same nav bar.
 *
 * All seven of §6's screens are routed here now. Note that no route carries a
 * `roles` prop: both roles may open a patient, a visit and the dashboard, and
 * the parts only one role should act on (the diagnosis form, the two queues)
 * are gated inside the screens. Role enforcement that matters lives on the
 * server — a `roles` prop hides a link, it does not protect an endpoint.
 */
import { Navigate, Route, Routes } from "react-router-dom";

import AppLayout from "./components/AppLayout";
import ProtectedRoute from "./auth/ProtectedRoute";
import ChangePassword from "./pages/ChangePassword";
import Dashboard from "./pages/Dashboard";
import HospitalProfile from "./pages/HospitalProfile";
import NewVisitFollowUp from "./pages/NewVisitFollowUp";
import NewVisitScreening from "./pages/NewVisitScreening";
import PatientProfile from "./pages/PatientProfile";
import RegisterPatient from "./pages/RegisterPatient";
import SignIn from "./pages/SignIn";
import SignUp from "./pages/SignUp";
import VisitDetail from "./pages/VisitDetail";
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
        <Route path="/" element={<Dashboard />} />

        {/* Forced first-login password change (fault #5) + hospital identity
            and staff management (faults #4/#5). */}
        <Route path="/change-password" element={<ChangePassword />} />
        <Route path="/hospital" element={<HospitalProfile />} />

        {/* §6 screen 3. A static segment, so React Router ranks it above
            Govind's /patients/:patientId and it won't be swallowed by it.
            Patient intake and starting visits are front-desk work — a clinician
            who lands here (or on either new-visit route) is bounced home. The
            server enforces the same with require_receptionist. */}
        <Route
          path="/patients/new"
          element={
            <ProtectedRoute roles={["receptionist"]}>
              <RegisterPatient />
            </ProtectedRoute>
          }
        />

        {/* §6 screens 5 and 6. Deliberately NOT /patients/:patientId or
            /visits/:visitId — those two paths are reserved for Govind's
            screens, which these screens' back links already point at.
            ?visitId=... resumes an awaiting_uploads visit (Product Rule 2A). */}
        <Route
          path="/patients/:patientId/new-visit/screening"
          element={
            <ProtectedRoute roles={["receptionist"]}>
              <NewVisitScreening />
            </ProtectedRoute>
          }
        />
        <Route
          path="/patients/:patientId/new-visit/follow-up"
          element={
            <ProtectedRoute roles={["receptionist"]}>
              <NewVisitFollowUp />
            </ProtectedRoute>
          }
        />

        {/* §6 screens 4 and 7. These two paths are what every other screen's
            back link and every list row already point at, so they are fixed. */}
        <Route path="/patients/:patientId" element={<PatientProfile />} />
        <Route path="/visits/:visitId" element={<VisitDetail />} />
      </Route>

      {/* Unknown path -> home, which itself redirects to /signin if signed out. */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
