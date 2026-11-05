/** The inline error under a field. Renders nothing when there's no message. */
export default function FieldError({ message }) {
  return message ? (
    <span className="field-error" role="alert">
      {message}
    </span>
  ) : null;
}
