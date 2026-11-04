/**
 * A whole-number input that can only ever hold digits.
 *
 * Deliberately `type="text" inputMode="numeric"`, not `type="number"`: a native
 * number input accepts "e", "+", "-" and ".", silently changes value on mouse
 * wheel, and hands React an empty string for half-typed input like "1e", so
 * the form can't tell "blank" from "garbage". Here every keystroke and paste is
 * filtered to digits (capped at the length of `max`), and the range is checked
 * on blur and on submit, with the error tied to the input via
 * aria-describedby.
 */
import { useId } from "react";

import { digitsOnly } from "../../utils/validate";

export default function NumberField({
  label,
  value,
  onChange,
  onBlur,
  min,
  max,
  hint,
  error,
  required = true,
  disabled,
  name,
}) {
  const id = useId();
  const hintId = `${id}-hint`;
  const errorId = `${id}-error`;
  const describedBy = [hint ? hintId : null, error ? errorId : null].filter(Boolean).join(" ") || undefined;

  return (
    <label className="field" htmlFor={id}>
      <span className="field-label">
        {label}
        {required ? null : <span className="field-optional"> (optional)</span>}
      </span>
      <input
        id={id}
        name={name}
        type="text"
        inputMode="numeric"
        pattern="[0-9]*"
        autoComplete="off"
        value={value}
        onChange={(event) => onChange(digitsOnly(event.target.value, String(max).length))}
        onBlur={onBlur}
        required={required}
        disabled={disabled}
        aria-invalid={error ? "true" : undefined}
        aria-describedby={describedBy}
        data-min={min}
        data-max={max}
      />
      {hint ? (
        <span className="field-hint" id={hintId}>
          {hint}
        </span>
      ) : null}
      {error ? (
        <span className="field-error" id={errorId} role="alert">
          {error}
        </span>
      ) : null}
    </label>
  );
}
