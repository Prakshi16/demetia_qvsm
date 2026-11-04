/**
 * Theme preference: "system" (follow the OS), "light" or "dark".
 *
 * Stored per browser in localStorage under "theme" (index.html reads it
 * before first paint). The resolved theme is applied as html[data-theme]; with
 * "system" the attribute is removed and tokens.css follows
 * prefers-color-scheme. Storage access is wrapped because it can throw in
 * private windows; the app works without it.
 */
import { useCallback, useEffect, useState } from "react";

const KEY = "theme";
const EVENT = "themechange";
const media = () => window.matchMedia?.("(prefers-color-scheme: dark)");

function readPreference() {
  try {
    const value = localStorage.getItem(KEY);
    return value === "light" || value === "dark" ? value : "system";
  } catch {
    return "system";
  }
}

function apply(preference) {
  const root = document.documentElement;
  if (preference === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", preference);
}

export function resolvedTheme(preference) {
  if (preference !== "system") return preference;
  return media()?.matches ? "dark" : "light";
}

export function useTheme() {
  const [preference, setPreference] = useState(readPreference);
  const [systemDark, setSystemDark] = useState(() => Boolean(media()?.matches));

  useEffect(() => {
    const query = media();
    if (!query) return undefined;
    const onChange = (event) => setSystemDark(event.matches);
    query.addEventListener?.("change", onChange);
    return () => query.removeEventListener?.("change", onChange);
  }, []);

  // Keep every useTheme() instance (nav toggle, design-system page) in sync.
  useEffect(() => {
    const onTheme = (event) => setPreference(event.detail);
    window.addEventListener(EVENT, onTheme);
    return () => window.removeEventListener(EVENT, onTheme);
  }, []);

  const setTheme = useCallback((next) => {
    setPreference(next);
    apply(next);
    window.dispatchEvent(new CustomEvent(EVENT, { detail: next }));
    try {
      if (next === "system") localStorage.removeItem(KEY);
      else localStorage.setItem(KEY, next);
    } catch {
      // preference just won't persist
    }
  }, []);

  const resolved = preference === "system" ? (systemDark ? "dark" : "light") : preference;
  return { preference, resolved, setTheme };
}
