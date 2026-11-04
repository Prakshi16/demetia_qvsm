/**
 * The model-output card on the screening and visit-detail screens.
 *
 * A three-way switch — Classical SVM · Quantum SVM · Compare — over the same
 * patient's two predictions:
 *
 *   - Verdict: the selected model's label, cross-faded on switch.
 *   - Decision axis: where this patient sits relative to the decision boundary,
 *     drawn against each model's typical range (95th-percentile |decision value|
 *     on the training patients). On a switch the marker springs from one
 *     model's position to the other's, so the motion itself shows how far
 *     apart the two models put this patient.
 *   - Cross-validated metrics for both models (5-fold GroupKFold, real data),
 *     bars animating between the selected model and the other.
 *
 * Honesty constraints carried over from the old card: the decision value is a
 * margin, never shown as a percentage or a "confidence bar"; the axis is a
 * diverging scale centred on the boundary, not a 0–100% fill. The classical
 * SVM is canonical (clinician queue + agreement flag); the QSVM is labelled as
 * a research view.
 */
import { useEffect, useMemo, useState } from "react";

import { api } from "../api/client";
import {
  AXIS_EXTENT,
  DEFAULT_SCALES,
  METRIC_ROWS,
  MODEL_META,
  formatPercent,
  formatSigned,
  isOffScale,
  markerPercent,
  modelAgreement,
  modelOutputs,
} from "../utils/prediction";
import SegmentedControl from "./ui/SegmentedControl";

const VIEW_STORAGE_KEY = "predictionView";

function readStoredView() {
  try {
    const stored = sessionStorage.getItem(VIEW_STORAGE_KEY);
    return stored === "svm" || stored === "qsvm" || stored === "compare" ? stored : "svm";
  } catch {
    return "svm";
  }
}

function useModelInfo(override) {
  const [info, setInfo] = useState(override ?? null);
  useEffect(() => {
    if (override) return undefined;
    let cancelled = false;
    api
      .getModelInfo()
      .then((data) => {
        if (!cancelled) setInfo(data);
      })
      .catch(() => {
        // The panel still works without it (default scales, no metric bars).
      });
    return () => {
      cancelled = true;
    };
  }, [override]);
  return info;
}

export default function PredictionPanel({ visit, title = "Model output", badge, modelInfo, children }) {
  const info = useModelInfo(modelInfo);
  const outputs = useMemo(() => modelOutputs(visit), [visit]);
  const hasQsvm = Boolean(outputs.qsvm);

  const [view, setView] = useState(readStoredView);
  const effectiveView = !hasQsvm ? "svm" : view;

  // First paint puts the markers on the boundary; the next frame moves them to
  // their values, so a freshly arrived result animates in.
  const [settled, setSettled] = useState(false);
  useEffect(() => {
    const id = requestAnimationFrame(() => setSettled(true));
    return () => cancelAnimationFrame(id);
  }, []);

  function changeView(next) {
    setView(next);
    try {
      sessionStorage.setItem(VIEW_STORAGE_KEY, next);
    } catch {
      // per-tab convenience only
    }
  }

  const cards = Object.fromEntries((info?.models ?? []).map((card) => [card.key, card]));
  const scales = {
    svm: cards.svm?.decision_scale ?? DEFAULT_SCALES.svm,
    qsvm: cards.qsvm?.decision_scale ?? DEFAULT_SCALES.qsvm,
  };
  const activeModel = effectiveView === "qsvm" ? "qsvm" : "svm";
  const active = outputs[activeModel];
  const agreement = modelAgreement(outputs);

  if (!outputs.svm) return null;

  const qsvmUnavailable = "Not computed for this visit (predicted before the quantum model was re-enabled).";
  const options = [
    { value: "svm", label: "Classical SVM", shortLabel: "SVM", color: "var(--color-model-svm)" },
    {
      value: "qsvm",
      label: "Quantum SVM",
      shortLabel: "QSVM",
      color: "var(--color-model-qsvm)",
      disabled: !hasQsvm,
      title: hasQsvm ? undefined : qsvmUnavailable,
    },
    { value: "compare", label: "Compare", disabled: !hasQsvm, title: hasQsvm ? undefined : qsvmUnavailable },
  ];

  return (
    <section
      className="visit-card prediction-panel"
      data-model={activeModel}
      aria-labelledby={`prediction-title-${visit.id}`}
    >
      <div className="prediction-head">
        <h2 id={`prediction-title-${visit.id}`}>
          {title}
          {badge}
        </h2>
        <div className="no-print prediction-switch">
          <SegmentedControl
            label="Which model's prediction to show"
            options={options}
            value={effectiveView}
            onChange={changeView}
          />
        </div>
      </div>

      <div className="prediction-body">
        <div className="verdict" aria-live="polite">
          {effectiveView === "compare" ? (
            <CompareVerdict outputs={outputs} agreement={agreement} />
          ) : (
            <>
              <p className="verdict-model">
                {MODEL_META[activeModel].name}
                {activeModel === "svm" ? (
                  <span className="badge badge--info" title="Sent to the clinician's queue and used for the agreement flag">
                    Canonical
                  </span>
                ) : (
                  <span className="badge badge--quantum" title="Shown for academic comparison; feeds no decision">
                    Research view
                  </span>
                )}
              </p>
              <p className="verdict-label" key={`${activeModel}-${active.label}`} data-class={active.label}>
                {active.label}
              </p>
              <p className="verdict-margin">
                decision value <strong>{formatSigned(active.signed)}</strong>
              </p>
              <p className="verdict-agreement">
                {agreement === "agree"
                  ? "Both models give the same label for this patient."
                  : agreement === "disagree"
                    ? "The two models disagree on this patient — see Compare."
                    : "Quantum model output isn't available for this visit."}
              </p>
            </>
          )}
        </div>

        <DecisionAxis
          outputs={outputs}
          scales={scales}
          view={effectiveView}
          settled={settled}
        />
      </div>

      {info ? <MetricCompare info={info} view={effectiveView} /> : null}

      <p className="prediction-foot">
        The decision value is the signed distance from the model&apos;s decision boundary
        (positive = Demented side). It is not a probability.
        {outputs.legacy ? " This visit was predicted under the earlier QSVM-canonical layout." : ""}
      </p>
      {children}
    </section>
  );
}

function CompareVerdict({ outputs, agreement }) {
  return (
    <>
      <p className="verdict-model" style={{ color: "var(--color-text-muted)" }}>
        Side by side
      </p>
      {["svm", "qsvm"].map((key) => (
        <div key={key} style={{ display: "grid", gap: 2 }}>
          <span className="verdict-model" style={{ color: `var(--color-model-${key})` }}>
            {MODEL_META[key].name}
          </span>
          <span
            className="verdict-label"
            data-class={outputs[key].label}
            style={{ fontSize: "var(--p-font-24)" }}
            key={`cmp-${key}-${outputs[key].label}`}
          >
            {outputs[key].label}
          </span>
          <span className="verdict-margin">
            decision value <strong>{formatSigned(outputs[key].signed)}</strong>
          </span>
        </div>
      ))}
      <p className="verdict-agreement">
        {agreement === "agree" ? "The models agree." : "The models disagree on this patient."}
      </p>
    </>
  );
}

function DecisionAxis({ outputs, scales, view, settled }) {
  const activeModel = view === "qsvm" ? "qsvm" : "svm";
  const rangeLabel =
    view === "compare"
      ? `each model's own typical range (×${AXIS_EXTENT})`
      : `±${(scales[activeModel] * AXIS_EXTENT).toFixed(2)} (${MODEL_META[activeModel].short})`;

  const markers = ["svm", "qsvm"]
    .filter((key) => outputs[key])
    .map((key) => {
      const value = outputs[key].signed;
      const state = view === "compare" || key === activeModel ? "active" : "muted";
      return {
        key,
        value,
        x: settled ? markerPercent(value, scales[key]) : 50,
        state,
        off: isOffScale(value, scales[key]),
      };
    });

  return (
    <div className="decision-chart">
      <p className="decision-chart-title">
        Position relative to the decision boundary <span>axis: {rangeLabel}</span>
      </p>
      <div className="decision-plot" role="img" aria-label={describeAxis(outputs, view)}>
        <span className="decision-boundary-label">boundary</span>
        <div className="decision-track" />
        <span className="decision-boundary" aria-hidden="true" />
        {markers.map((marker) => (
          <span
            key={marker.key}
            className="decision-marker"
            data-model={marker.key}
            data-state={marker.state}
            data-tag={view === "compare" && marker.key === "qsvm" ? "below" : "above"}
            style={{ "--x": `${marker.x}%`, zIndex: marker.state === "active" ? 2 : 1 }}
          >
            <span className="decision-marker-tag">
              {MODEL_META[marker.key].short} {formatSigned(marker.value)}
              {marker.off ? (marker.value > 0 ? " ›" : " ‹") : ""}
            </span>
          </span>
        ))}
        <div className="decision-ticks" aria-hidden="true">
          <span>← Nondemented</span>
          <span>Demented →</span>
        </div>
      </div>
      <div className="decision-legend" aria-hidden="true">
        <span>
          <i style={{ "--c": "var(--color-model-svm)" }} /> Classical SVM
        </span>
        <span>
          <i style={{ "--c": "var(--color-model-qsvm)" }} /> Quantum SVM
        </span>
      </div>
    </div>
  );
}

function describeAxis(outputs, view) {
  const parts = ["svm", "qsvm"]
    .filter((key) => outputs[key] && (view === "compare" || key === (view === "qsvm" ? "qsvm" : "svm")))
    .map((key) => `${MODEL_META[key].name}: ${outputs[key].label}, decision value ${formatSigned(outputs[key].signed)}`);
  return `Decision axis. ${parts.join(". ")}.`;
}

function MetricCompare({ info, view }) {
  const cards = Object.fromEntries(info.models.map((card) => [card.key, card]));
  const keys = ["svm", "qsvm"].filter((key) => cards[key]);
  const activeModel = view === "qsvm" ? "qsvm" : "svm";

  return (
    <div className="metric-compare">
      <div className="metric-compare-head">
        <h3>How each model scored in validation</h3>
        <p>{info.evaluation}</p>
      </div>
      <div className="metric-rows">
        {METRIC_ROWS.map((row) => (
          <div key={row.key} className="metric-row">
            <div className="metric-name">
              {row.label}
              {row.hint ? <small>{row.hint}</small> : null}
            </div>
            <div className="metric-bars">
              {keys.map((key) => {
                const metric = cards[key].metrics[row.key];
                if (!metric) return null;
                const state = view === "compare" || key === activeModel ? "active" : "muted";
                return (
                  <div
                    key={key}
                    className="metric-bar"
                    data-model={key}
                    data-state={state}
                    style={{ "--value": metric.mean, "--std": metric.std }}
                    title={`${MODEL_META[key].name}: ${formatPercent(metric.mean)} ± ${formatPercent(metric.std)} across folds`}
                  >
                    <div className="metric-bar-track">
                      <div className="metric-bar-fill" />
                      <div className="metric-bar-std" />
                    </div>
                    <span className="metric-bar-value">
                      <span className="visually-hidden">{MODEL_META[key].name} </span>
                      {formatPercent(metric.mean)}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
