/**
 * Client-side mirror of the backend validators in app/schemas.py (fault #6).
 *
 * These are a UX convenience — they keep obviously-bad input from ever being
 * submitted — not the source of truth. The backend runs the same rules and its
 * 422s render through api/client.js regardless.
 *
 * Each function returns an error string, or null when the value is acceptable.
 */

const MIN_PASSWORD_LENGTH = 8;

function collapseWhitespace(value) {
  return (value ?? "").trim().replace(/\s+/g, " ");
}

function checkName(value, { label, extra, maxLen }) {
  const cleaned = collapseWhitespace(value);
  if (cleaned.length < 2 || cleaned.length > maxLen) {
    return `${label} must be between 2 and ${maxLen} characters.`;
  }
  if (!/\p{L}/u.test(cleaned)) {
    return `${label} must contain at least one letter.`;
  }
  // Allowed: any Unicode letter or number, plus the punctuation in `extra`.
  const allowed = new RegExp(`^[\\p{L}\\p{N}${extra}]+$`, "u");
  if (!allowed.test(cleaned)) {
    return `${label} contains invalid characters.`;
  }
  return null;
}

export function validatePersonName(value) {
  return checkName(value, { label: "Name", extra: " \\-'.", maxLen: 100 });
}

export function validateOrgName(value) {
  return checkName(value, {
    label: "Hospital name",
    extra: " \\-'.,&()/#",
    maxLen: 120,
  });
}

export function validatePincode(value) {
  return /^\d{6}$/.test((value ?? "").replace(/\s+/g, ""))
    ? null
    : "Pincode must be exactly 6 digits.";
}

export function validatePassword(value) {
  return (value ?? "").length >= MIN_PASSWORD_LENGTH
    ? null
    : `Password must be at least ${MIN_PASSWORD_LENGTH} characters.`;
}
