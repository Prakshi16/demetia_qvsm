/**
 * MMSE + CDR over time, as one chart (§6 screen 4).
 *
 * Inline SVG rather than a charting library: this is the only chart in the app,
 * and recharts would be a bigger dependency than the sixty lines below. It also
 * keeps §9's "plain CSS, no framework" constraint honest.
 *
 * The two measures share an x-axis but not a y-axis — MMSE runs 0–30 and higher
 * is better, CDR runs 0–3 and higher is worse. Plotting them on one scale would
 * flatten CDR into the baseline and imply the two lines move together, when in a
 * declining patient they move in opposite directions. So: MMSE on the left axis,
 * CDR on the right, and a legend naming which is which.
 */

const WIDTH = 640;
const HEIGHT = 220;
const PAD = { top: 16, right: 44, bottom: 34, left: 40 };

const MMSE_MAX = 30;
// The CDR scale is 0/0.5/1/2/3. The form only offers up to 2, but a visit
// created before that constraint existed could hold 3, and a point above the
// axis ceiling would be drawn outside the plot area.
const CDR_MAX = 3;

const PLOT_W = WIDTH - PAD.left - PAD.right;
const PLOT_H = HEIGHT - PAD.top - PAD.bottom;

function formatDate(value) {
  return new Date(value).toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

export default function TrendChart({ points }) {
  const series = (points ?? []).filter((p) => p.mmse != null || p.cdr != null);

  if (series.length === 0) {
    return (
      <p className="list-note">
        No clinical measures recorded yet. The chart appears once a visit is saved.
      </p>
    );
  }

  // A single visit has no line to draw, only a dot. Guard the divide-by-zero
  // rather than special-casing further down.
  const lastIndex = Math.max(series.length - 1, 1);
  const x = (index) =>
    series.length === 1
      ? PAD.left + PLOT_W / 2 // a lone point pinned to the axis reads as an error
      : PAD.left + (index / lastIndex) * PLOT_W;
  const yFor = (value, max) => PAD.top + PLOT_H - (value / max) * PLOT_H;

  function path(key, max) {
    const drawn = series
      .map((point, index) => ({ point, index }))
      .filter(({ point }) => point[key] != null);
    if (drawn.length === 0) return null;
    return drawn
      .map(({ point, index }, position) =>
        `${position === 0 ? "M" : "L"} ${x(index)} ${yFor(point[key], max)}`)
      .join(" ");
  }

  const mmsePath = path("mmse", MMSE_MAX);
  const cdrPath = path("cdr", CDR_MAX);

  return (
    <div className="trend">
      <ul className="trend-legend">
        <li>
          <span className="trend-swatch trend-swatch--mmse" aria-hidden="true" />
          MMSE <span className="trend-legend-hint">0–30, higher is better</span>
        </li>
        <li>
          <span className="trend-swatch trend-swatch--cdr" aria-hidden="true" />
          CDR <span className="trend-legend-hint">0–3, higher is worse</span>
        </li>
      </ul>

      <div className="trend-scroll">
        <svg
          className="trend-svg"
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          role="img"
          aria-label={
            `MMSE and CDR across ${series.length} visit` +
            `${series.length === 1 ? "" : "s"}, ` +
            `${formatDate(series[0].visit_date)} to ` +
            `${formatDate(series[series.length - 1].visit_date)}`
          }
        >
          {/* Thirds rather than quarters: both axes land on whole numbers
              (30/20/10/0 and 3/2/1/0), where quarters give 22.5 and 7.5. */}
          {[0, 1 / 3, 2 / 3, 1].map((fraction) => {
            const y = PAD.top + PLOT_H * fraction;
            return (
              <g key={fraction}>
                <line
                  className="trend-grid"
                  x1={PAD.left}
                  x2={WIDTH - PAD.right}
                  y1={y}
                  y2={y}
                />
                <text className="trend-axis" x={PAD.left - 8} y={y + 4} textAnchor="end">
                  {Math.round(MMSE_MAX * (1 - fraction))}
                </text>
                <text
                  className="trend-axis trend-axis--right"
                  x={WIDTH - PAD.right + 8}
                  y={y + 4}
                >
                  {Math.round(CDR_MAX * (1 - fraction))}
                </text>
              </g>
            );
          })}

          {mmsePath ? <path className="trend-line trend-line--mmse" d={mmsePath} /> : null}
          {cdrPath ? <path className="trend-line trend-line--cdr" d={cdrPath} /> : null}

          {series.map((point, index) => (
            <g key={point.visit_date + index}>
              {point.mmse != null ? (
                <circle
                  className="trend-dot trend-dot--mmse"
                  cx={x(index)}
                  cy={yFor(point.mmse, MMSE_MAX)}
                  r="4"
                >
                  <title>{`${formatDate(point.visit_date)} — MMSE ${point.mmse}`}</title>
                </circle>
              ) : null}
              {point.cdr != null ? (
                <circle
                  className="trend-dot trend-dot--cdr"
                  cx={x(index)}
                  cy={yFor(point.cdr, CDR_MAX)}
                  r="4"
                >
                  <title>{`${formatDate(point.visit_date)} — CDR ${point.cdr}`}</title>
                </circle>
              ) : null}
            </g>
          ))}

          {/* Only the first and last dates are labelled: with visits bunched
              together every label would overlap its neighbour. */}
          <text className="trend-axis" x={PAD.left} y={HEIGHT - 10}>
            {formatDate(series[0].visit_date)}
          </text>
          {series.length > 1 ? (
            <text
              className="trend-axis"
              x={WIDTH - PAD.right}
              y={HEIGHT - 10}
              textAnchor="end"
            >
              {formatDate(series[series.length - 1].visit_date)}
            </text>
          ) : null}
        </svg>
      </div>
    </div>
  );
}
