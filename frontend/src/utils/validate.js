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

// Mirror of clean_email_domain in app/services/emails.py — a typo gate, not an
// RFC check. Tolerates a pasted "@domain" or a full address.
const DOMAIN_RE = /^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$/;

export function normaliseEmailDomain(value) {
  let cleaned = (value ?? "").trim().toLowerCase();
  if (cleaned.includes("://")) cleaned = cleaned.split("://")[1];
  cleaned = cleaned.split("/")[0].replace(/^@+/, "");
  if (cleaned.includes("@")) cleaned = cleaned.split("@")[1];
  return cleaned;
}

export function validateEmailDomain(value) {
  const cleaned = normaliseEmailDomain(value);
  return cleaned.length <= 253 && DOMAIN_RE.test(cleaned)
    ? null
    : "Enter a valid email domain, e.g. yourhospital.com";
}

// Mirror of local_part_from_name in app/services/emails.py: first two name
// tokens, letters only, dot-joined. Used to preview the login address a name
// will produce ("" until there is something to show).
export function slugLocalPart(name) {
  const tokens = (name ?? "")
    .split(/\s+/)
    .map((token) => token.toLowerCase().replace(/[^a-z]/g, ""))
    .filter(Boolean)
    .slice(0, 2);
  return tokens.join(".");
}

// --------------------------------------------------------------------------
// Numbers and other structured fields — mirrors of app/schemas.py
// (VisitCreate, PatientCreate). Inputs hold strings; these judge the string.
// --------------------------------------------------------------------------

/** Keep only ASCII digits (and cap the length) — used on every keystroke. */
export function digitsOnly(value, maxLength = Infinity) {
  return String(value ?? "").replace(/\D+/g, "").slice(0, maxLength);
}

/**
 * A whole number in [min, max]. Rejects blanks (when required), signs,
 * decimals, exponents ("1e2"), spaces and anything non-numeric — the same
 * set the API's strict validator refuses.
 */
export function validateWholeNumber(value, { label, min, max, required = true }) {
  const text = String(value ?? "").trim();
  if (text === "") return required ? `${label} is required.` : null;
  if (!/^\d+$/.test(text)) return `${label} must be a whole number.`;
  const number = Number(text);
  if (number < min || number > max) return `${label} must be between ${min} and ${max}.`;
  return null;
}

/** One of a fixed set of option values (e.g. CDR "0" | "0.5" | ...). */
export function validateChoice(value, { label, options, required = true }) {
  const text = String(value ?? "");
  if (text === "") return required ? `Select a ${label}.` : null;
  return options.includes(text) ? null : `${label} must be one of ${options.join(", ")}.`;
}

export function validatePhone(value) {
  const text = String(value ?? "").trim();
  if (text === "") return null; // optional
  if (!/^\+?[\d\s\-().]+$/.test(text)) {
    return "Phone may contain only digits, spaces, +, -, ( and ).";
  }
  const digits = text.replace(/\D/g, "").length;
  return digits >= 7 && digits <= 15 ? null : "Phone must have between 7 and 15 digits.";
}

/** Keystroke filter for phone inputs: drop anything the validator rejects. */
export function phoneCharsOnly(value) {
  return String(value ?? "").replace(/[^\d\s\-().+]/g, "").slice(0, 24);
}

export function validateDob(value, today = new Date()) {
  if (!value) return null; // optional
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return "Enter the date as YYYY-MM-DD.";
  const dob = new Date(`${value}T00:00:00`);
  if (Number.isNaN(dob.getTime())) return "Enter a real date.";
  const todayMidnight = new Date(today.getFullYear(), today.getMonth(), today.getDate());
  if (dob > todayMidnight) return "Date of birth can't be in the future.";
  if (dob.getFullYear() < today.getFullYear() - 120) return "Date of birth is more than 120 years ago.";
  return null;
}

export function validateRelationship(value, { required }) {
  const text = String(value ?? "").trim();
  if (text === "") return required ? "Say who the guardian is to the patient, e.g. Daughter." : null;
  if (text.length > 60) return "Relationship must be at most 60 characters.";
  return /^[\p{L} \-'/]+$/u.test(text) ? null : "Relationship must be words only, e.g. Daughter.";
}

export function validateFreeText(value, { label, maxLength }) {
  const text = String(value ?? "");
  return text.trim().length > maxLength ? `${label} must be at most ${maxLength} characters.` : null;
}

/** Clinical scales, shared by the screening and follow-up forms. */
export const CLINICAL_RULES = {
  mmse: { label: "MMSE", min: 0, max: 30 },
  edu: { label: "Education (years)", min: 0, max: 30 },
};
export const CDR_OPTIONS = ["0", "0.5", "1", "2", "3"];
export const SES_OPTIONS = [
  { value: "1", label: "1 — highest" },
  { value: "2", label: "2" },
  { value: "3", label: "3 — middle" },
  { value: "4", label: "4" },
  { value: "5", label: "5 — lowest" },
];
