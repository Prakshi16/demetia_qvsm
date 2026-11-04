/**
 * Segmented control: a single-choice switch with a sliding thumb.
 *
 * Accessible as a radio group (role="radiogroup" + role="radio" with
 * aria-checked; arrow keys move between options), and animated by moving one
 * pseudo-element (the thumb) via the --segmented-index custom property, so the
 * switch is one continuous motion rather than two buttons repainting.
 */
import { useRef } from "react";

export default function SegmentedControl({ label, options, value, onChange, size }) {
  const refs = useRef([]);
  const index = Math.max(0, options.findIndex((option) => option.value === value));

  function move(delta) {
    const enabled = options.map((option, i) => (option.disabled ? null : i)).filter((i) => i !== null);
    if (enabled.length === 0) return;
    const position = enabled.indexOf(index);
    const next = enabled[(position + delta + enabled.length) % enabled.length];
    onChange(options[next].value);
    refs.current[next]?.focus();
  }

  return (
    <div
      className={`segmented${size === "sm" ? " segmented--sm" : ""}`}
      role="radiogroup"
      aria-label={label}
      style={{ "--segmented-index": index, "--segmented-count": options.length }}
      onKeyDown={(event) => {
        if (event.key === "ArrowRight" || event.key === "ArrowDown") {
          event.preventDefault();
          move(1);
        } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
          event.preventDefault();
          move(-1);
        }
      }}
    >
      {options.map((option, i) => (
        <button
          key={option.value}
          ref={(node) => {
            refs.current[i] = node;
          }}
          type="button"
          role="radio"
          aria-checked={option.value === value}
          tabIndex={option.value === value ? 0 : -1}
          disabled={option.disabled}
          title={option.title}
          onClick={() => onChange(option.value)}
        >
          {option.color ? (
            <span className="segmented-dot" style={{ "--dot": option.color }} aria-hidden="true" />
          ) : null}
          {option.shortLabel ? (
            <>
              <span className="segmented-label-long">{option.label}</span>
              <span className="segmented-label-short" aria-hidden="true">
                {option.shortLabel}
              </span>
            </>
          ) : (
            <span className="segmented-label">{option.label}</span>
          )}
        </button>
      ))}
    </div>
  );
}
