/**
 * /design-system — the living reference for the Cortex design system.
 *
 * Tokens -> components -> patterns, rendered from the real CSS (nothing here
 * is a screenshot), so it can't drift from what the screens use. The
 * prediction panel is shown with fixture visits in each state it can be in.
 */
import { useState } from "react";

import MriExtractionNote from "../components/MriExtractionNote";
import PredictionPanel from "../components/PredictionPanel";
import NumberField from "../components/ui/NumberField";
import SegmentedControl from "../components/ui/SegmentedControl";
import { useTheme } from "../theme/useTheme";
import { validateWholeNumber } from "../utils/validate";

const RAMPS = ["deep", "sage", "tide", "ochre", "red"];
const STEPS = {
  deep: [25, 50, 100, 150, 200, 300, 500, 600, 650, 700, 750, 800, 850, 900, 950],
  tide: [50, 300, 400, 500, 600],
  ochre: [50, 100, 200, 600, 700],
  default: [50, 100, 200, 300, 400, 500, 600, 700, 800, 900],
};

const SEMANTIC = [
  ["--color-bg", "Page background"],
  ["--color-surface", "Card / input surface"],
  ["--color-surface-sunken", "Wells, tracks"],
  ["--color-border", "Hairlines"],
  ["--color-text", "Body text"],
  ["--color-text-muted", "Secondary text (≥4.5:1)"],
  ["--color-accent", "Primary action"],
  ["--color-accent-soft", "Selected / info wash"],
  ["--color-success-fg", "Success / reviewed"],
  ["--color-warning-fg", "Attention / pending"],
  ["--color-danger-fg", "Errors only"],
  ["--color-class-negative", "Model: Nondemented"],
  ["--color-class-positive", "Model: Demented"],
  ["--color-model-svm", "Classical SVM"],
  ["--color-model-qsvm", "Quantum SVM"],
];

const TYPE = [
  ["--text-display", "Display — verdicts", 700],
  ["--text-h1", "Heading 1 — page titles", 700],
  ["--text-h2", "Heading 2 — card titles", 600],
  ["--text-h3", "Heading 3 — sub-sections", 600],
  ["--text-lead", "Lead — data values", 500],
  ["--text-body", "Body — default text", 400],
  ["--text-body-sm", "Small body — buttons, lists", 400],
  ["--text-small", "Small — hints, meta", 400],
  ["--text-caption", "Caption — labels, badges", 600],
];

const SPACE = ["--space-2xs", "--space-xs", "--space-sm", "--space-md", "--space-lg", "--space-xl", "--space-2xl"];

const MODEL_INFO_FIXTURE = {
  evaluation: "GroupKFold(n_splits=5) on the composite group, fused 24-feature row, N=233 real patients",
  models: [
    {
      key: "svm", name: "Classical SVM", kernel: "RBF kernel", canonical: true, decision_scale: 1.8415,
      metrics: {
        accuracy: { mean: 0.8114, std: 0.0301 }, recall: { mean: 0.8593, std: 0.0516 },
        specificity: { mean: 0.7487, std: 0.0805 }, f1: { mean: 0.8336, std: 0.0225 },
        roc_auc: { mean: 0.9077, std: 0.0228 },
      },
    },
    {
      key: "qsvm", name: "Quantum SVM", kernel: "ZZFeatureMap", canonical: false, decision_scale: 1.4137,
      metrics: {
        accuracy: { mean: 0.7814, std: 0.0777 }, recall: { mean: 0.7944, std: 0.1058 },
        specificity: { mean: 0.7687, std: 0.0498 }, f1: { mean: 0.7988, std: 0.0699 },
        roc_auc: { mean: 0.8453, std: 0.0513 },
      },
    },
  ],
};

const FIXTURES = {
  agree: {
    id: "fixture-agree", model_prediction: "Demented", model_confidence: 0.84,
    qsvm_prediction: "Demented", qsvm_confidence: 0.41,
  },
  disagree: {
    id: "fixture-disagree", model_prediction: "Nondemented", model_confidence: 0.32,
    qsvm_prediction: "Demented", qsvm_confidence: 0.18,
  },
  svmOnly: { id: "fixture-svm-only", model_prediction: "Nondemented", model_confidence: 1.37 },
};

const MRI_FIXTURE = {
  mri_feature_vector: [0.7412, 1476],
  mri_extraction: {
    input_kind: "slice", source_format: "image", method: "calibrated", etiv_measured: false,
    expected_error: { nwbv: { mae: 0.026 } },
    warnings: ["Single 2-D image: eTIV can't be measured from one slice, so the training median (1475 mL) is used."],
  },
};

function Section({ id, title, intro, children }) {
  return (
    <section className="visit-card ds-section" aria-labelledby={id}>
      <h2 id={id}>{title}</h2>
      {intro ? <p>{intro}</p> : null}
      {children}
    </section>
  );
}

export default function DesignSystem() {
  const { preference, setTheme } = useTheme();
  const [segment, setSegment] = useState("svm");
  const [demoValue, setDemoValue] = useState("31");
  const [fixture, setFixture] = useState("agree");

  return (
    <div className="page visit-page">
      <header className="visit-header">
        <p className="visit-eyebrow">Cortex design system</p>
        <h1>Tokens, components and patterns</h1>
        <p className="visit-subtitle">
          Three token tiers (primitive → semantic → component) feed a small set of components,
          which compose into the screens. Everything below is rendered live from the same CSS
          the app uses, in whichever theme is active.
        </p>
        <div className="ds-row" style={{ marginTop: "var(--space-xs)" }}>
          <SegmentedControl
            label="Theme"
            value={preference}
            onChange={setTheme}
            options={[
              { value: "system", label: "System" },
              { value: "light", label: "Light" },
              { value: "dark", label: "Dark" },
            ]}
          />
        </div>
      </header>

      <Section
        id="ds-principles"
        title="Principles"
        intro="The rules every decision below follows."
      >
        <ul style={{ margin: 0, paddingLeft: "var(--list-indent)", display: "grid", gap: "var(--space-xs)" }}>
          <li><strong>Clinical calm.</strong> The interface stays in one muted teal-sage family; saturated colour is kept for clinical signal. Red only means an error. A “Demented” model output is a finding to review, so it is amber, not red.</li>
          <li><strong>Never colour alone.</strong> Every status carries a word (and badges a dot); every chart has labels.</li>
          <li><strong>Honest numbers.</strong> A model margin is shown on a diverging axis around its decision boundary, never as a percentage.</li>
          <li><strong>AA contrast.</strong> Body and muted text are ≥4.5:1 on every surface in both themes (checked by <code>npm run check:contrast</code>).</li>
          <li><strong>Motion explains.</strong> Animation is used where it carries meaning (a patient’s marker sliding between two models), and turns off under reduced-motion.</li>
        </ul>
      </Section>

      <Section id="ds-color" title="Colour" intro="Primitive ramps (tier 1) — never used directly by components.">
        <div className="ds-grid-2">
          {RAMPS.map((ramp) => (
            <div key={ramp}>
              <p className="ds-ramp-label">{ramp}</p>
              <div className="ds-ramp" style={{ "--steps": (STEPS[ramp] ?? STEPS.default).length }}>
                {(STEPS[ramp] ?? STEPS.default).map((step) => (
                  <span key={step} style={{ "--c": `var(--p-${ramp}-${step})` }} title={`--p-${ramp}-${step}`} />
                ))}
              </div>
            </div>
          ))}
        </div>
        <p style={{ color: "var(--color-text-muted)" }}>Semantic roles (tier 2) — what components read, and what the dark theme redefines.</p>
        <div className="ds-swatches">
          {SEMANTIC.map(([token, role]) => (
            <div key={token} className="ds-swatch">
              <span className="ds-swatch-chip" style={{ "--c": `var(${token})` }} />
              <strong>{role}</strong>
              <code>{token}</code>
            </div>
          ))}
        </div>
      </Section>

      <Section id="ds-type" title="Typography" intro="Inter for text, JetBrains Mono for raw numbers (margins, measurements). Headings use fluid clamp() sizes.">
        <div className="ds-type">
          {TYPE.map(([token, label, weight]) => (
            <div key={token} className="ds-type-row">
              <code>{token}</code>
              <span style={{ fontSize: `var(${token})`, fontWeight: weight, lineHeight: "var(--leading-tight)" }}>{label}</span>
            </div>
          ))}
        </div>
      </Section>

      <Section id="ds-space" title="Spacing, shape and elevation" intro="A 4px grid. Radii: 2 / 4 / 6 / 8 / 12 / 16 / pill. Shadows soften to borders in dark mode.">
        <div className="ds-space">
          {SPACE.map((token) => (
            <div key={token} className="ds-space-row">
              <code>{token}</code>
              <span style={{ "--w": `var(${token})` }} />
            </div>
          ))}
        </div>
        <div className="ds-row">
          {["--shadow-xs", "--shadow-sm", "--shadow-md", "--shadow-lg"].map((shadow) => (
            <div
              key={shadow}
              style={{
                width: "var(--ds-demo-width)", height: "var(--ds-demo-height)", display: "grid", placeItems: "center",
                background: "var(--color-surface)", borderRadius: "var(--radius-lg)",
                border: "var(--border-width) solid var(--color-border)", boxShadow: `var(${shadow})`,
                fontSize: "var(--text-caption)",
              }}
            >
              <code>{shadow}</code>
            </div>
          ))}
        </div>
      </Section>

      <Section id="ds-components" title="Components" intro="Buttons, fields, badges, notices and the segmented control.">
        <div className="ds-row">
          <button type="button" className="button-primary">Primary</button>
          <button type="button" className="button-secondary">Secondary</button>
          <button type="button" className="button-quiet">Quiet</button>
          <button type="button" className="button-danger">Destructive</button>
          <button type="button" className="button-primary" disabled>Disabled</button>
          <button type="button" className="link-button">Link button</button>
        </div>

        <div className="visit-form-row">
          <NumberField
            label="MMSE (digits only)"
            value={demoValue}
            onChange={setDemoValue}
            min={0}
            max={30}
            hint="Try typing letters, a minus sign or 31."
            error={validateWholeNumber(demoValue, { label: "MMSE", min: 0, max: 30 }) ?? undefined}
          />
          <label className="field">
            <span className="field-label">Select</span>
            <select defaultValue="">
              <option value="">Select…</option>
              <option>0.5</option>
            </select>
            <span className="field-hint">Fixed scales are selects.</span>
          </label>
          <label className="field">
            <span className="field-label">Read-only</span>
            <input readOnly value="Saved value" />
          </label>
        </div>

        <div className="ds-row">
          <span className="badge">Neutral</span>
          <span className="badge badge--success">Reviewed</span>
          <span className="badge badge--warning">Pending review</span>
          <span className="badge badge--info">Canonical</span>
          <span className="badge badge--quantum">Research view</span>
          <span className="badge badge--danger">Failed</span>
        </div>

        <p className="notice">Information notice — the decision-support disclaimer uses this.</p>
        <p className="notice notice--warning">Warning notice — e.g. an MRI read from a single slice.</p>
        <p className="form-error">Form error — a failed save.</p>

        <div className="ds-row">
          <SegmentedControl
            label="Demo segmented control"
            value={segment}
            onChange={setSegment}
            options={[
              { value: "svm", label: "Classical SVM", shortLabel: "SVM", color: "var(--color-model-svm)" },
              { value: "qsvm", label: "Quantum SVM", shortLabel: "QSVM", color: "var(--color-model-qsvm)" },
              { value: "compare", label: "Compare" },
            ]}
          />
          <span className="spinner" aria-label="Loading" />
          <span className="skeleton" style={{ width: "var(--ds-demo-width)" }} />
        </div>
      </Section>

      <Section
        id="ds-patterns"
        title="Pattern: prediction panel"
        intro="The model-output card used on the screening and visit-detail screens. Switch between models: the marker springs between the two models' positions for the same patient, and the validation bars follow."
      >
        <SegmentedControl
          label="Fixture"
          value={fixture}
          onChange={setFixture}
          options={[
            { value: "agree", label: "Models agree", shortLabel: "Agree" },
            { value: "disagree", label: "Models disagree", shortLabel: "Disagree" },
            { value: "svmOnly", label: "SVM only (older visit)", shortLabel: "SVM only" },
          ]}
        />
        <PredictionPanel key={fixture} visit={FIXTURES[fixture]} modelInfo={MODEL_INFO_FIXTURE} />
      </Section>

      <Section id="ds-mri" title="Pattern: MRI extraction note" intro="Shown under a processed scan: the two values, their validated error, and any caveat.">
        <MriExtractionNote visit={MRI_FIXTURE} />
      </Section>
    </div>
  );
}
