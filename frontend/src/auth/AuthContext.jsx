/**
 * Who is signed in, for the whole app.
 *
 * The JWT is stored in localStorage so a refresh doesn't sign you out. It has a
 * 24h expiry and no refresh token (deliberate Phase 2 simplification), so the
 * client also listens for the expiry event the API client fires on a 401 and
 * clears state — otherwise the user would sit on a screen making requests that
 * all silently fail.
 */
import { useCallback, useEffect, useMemo, useState } from "react";

import { api, AUTH_EXPIRED_EVENT, TOKEN_KEY, USER_KEY } from "../api/client";
import { AuthContext } from "./authContext";

function readStoredUser() {
  const raw = localStorage.getItem(USER_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    // Corrupt entry — treat as signed out rather than crashing on boot.
    localStorage.removeItem(USER_KEY);
    return null;
  }
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(readStoredUser);

  const signOut = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    setUser(null);
  }, []);

  useEffect(() => {
    window.addEventListener(AUTH_EXPIRED_EVENT, signOut);
    return () => window.removeEventListener(AUTH_EXPIRED_EVENT, signOut);
  }, [signOut]);

  /** Store the { token, user } every auth endpoint returns. */
  const acceptSession = useCallback((session) => {
    localStorage.setItem(TOKEN_KEY, session.token);
    localStorage.setItem(USER_KEY, JSON.stringify(session.user));
    setUser(session.user);
    return session.user;
  }, []);

  const signIn = useCallback(
    async (credentials) => acceptSession(await api.login(credentials)),
    [acceptSession],
  );

  const registerHospital = useCallback(
    async (body) => acceptSession(await api.registerHospital(body)),
    [acceptSession],
  );

  const registerStaff = useCallback(
    async (body) => acceptSession(await api.registerStaff(body)),
    [acceptSession],
  );

  const value = useMemo(
    () => ({
      user,
      isSignedIn: user !== null,
      signIn,
      signOut,
      registerHospital,
      registerStaff,
    }),
    [user, signIn, signOut, registerHospital, registerStaff],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
