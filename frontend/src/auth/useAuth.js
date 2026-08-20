/**
 * Access the signed-in user and the auth actions from any component:
 *
 *   const { user, isSignedIn, signIn, signOut } = useAuth();
 *
 * `user` is { id, hospital_id, name, email, role } — the same UserOut the API
 * returns — or null when signed out.
 */
import { useContext } from "react";

import { AuthContext } from "./authContext";

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === null) {
    throw new Error("useAuth must be used inside <AuthProvider>");
  }
  return context;
}
