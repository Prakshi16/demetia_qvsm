/**
 * The context object itself, on its own so that AuthContext.jsx exports only a
 * component and useAuth.js exports only a hook — otherwise Vite's fast refresh
 * stops working for the file that mixes them (and oxlint warns about it).
 *
 * Nothing should import this directly except AuthContext.jsx and useAuth.js;
 * components use `useAuth()`.
 */
import { createContext } from "react";

export const AuthContext = createContext(null);
