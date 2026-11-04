/**
 * Pure helpers behind the prediction panel (no React, so `node --test` can
 * cover them directly; see prediction.test.js).
 *
 * Conventions:
 *   - A model's stored "confidence" is |decision_function|, a margin, not a
 *     probability. Its sign comes from the label: Demented (class 1) is
 *     positive, Nondemented negative.
 *   - The chart axis spans ±(AXIS_EXTENT × the model's decision_scale), where
 *     decision_scale is the 95th percentile |decision value| over training
 *     patients (results/model_info.json). Each model is drawn against its own
 *     typical range, because the raw margins of an RBF-SVM and a quantum-kernel
 *     SVM aren't in the same units.
 */

export const AXIS_EXTENT = 1.25;
export const MODEL_KEYS = ["svm", "qsvm"];

export const MODEL_META = {
  svm: { name: "Classical SVM", short: "SVM", kernel: "RBF kernel" },
  qsvm: { name: "Quantum SVM", short: "QSVM", kernel: "ZZFeatureMap kernel · 6 qubits" },
};

// Fallback scales if /models/info is unreachable (values from model_info.json
// at the time of writing; the live endpoint wins when it answers).
export const DEFAULT_SCALES = { svm: 1.8415, qsvm: 1.4137 };

export function signedDecision(label, confidence) {
  if (!label || confidence == null || !Number.isFinite(Number(confidence))) return null;
  const magnitude = Math.abs(Number(confidence));
  return label === "Demented" ? magnitude : -magnitude;
}

/** Signed decision value -> marker position along the axis, in percent. */
export function markerPercent(signed, scale, extent = AXIS_EXTENT) {
  if (signed == null || !(scale > 0)) return 50;
  const ratio = Math.max(-1, Math.min(1, signed / (scale * extent)));
  return 50 + ratio * 50;
}

/** True when the value sits beyond the drawn axis (marker pinned at the end). */
export function isOffScale(signed, scale, extent = AXIS_EXTENT) {
  return signed != null && scale > 0 && Math.abs(signed) > scale * extent;
}

function entry(label, confidence) {
  const signed = signedDecision(label, confidence);
  return signed == null ? null : { label, confidence: Math.abs(Number(confidence)), signed };
}

/**
 * Both models' outputs for a visit, as { svm, qsvm } (either may be null).
 *
 * Column history:
 *   - since 2026-11-04: model_* = SVM (canonical), qsvm_* = QSVM (research view)
 *   - 2026-09-25 to 2026-11-04: model_* = SVM only
 *   - before 2026-09-25: model_* = QSVM, svm_* = SVM. Detected by svm_* being
 *     set while qsvm_* isn't, so old visits still show the right model per tab.
 */
export function modelOutputs(visit) {
  if (!visit) return { svm: null, qsvm: null, legacy: false };
  const legacy = visit.svm_prediction != null && visit.qsvm_prediction == null;
  if (legacy) {
    return {
      svm: entry(visit.svm_prediction, visit.svm_confidence),
      qsvm: entry(visit.model_prediction, visit.model_confidence),
      legacy: true,
    };
  }
  return {
    svm: entry(visit.model_prediction, visit.model_confidence),
    qsvm: entry(visit.qsvm_prediction, visit.qsvm_confidence),
    legacy: false,
  };
}

/** "agree" | "disagree" | null (when either model is missing). */
export function modelAgreement(outputs) {
  if (!outputs?.svm || !outputs?.qsvm) return null;
  return outputs.svm.label === outputs.qsvm.label ? "agree" : "disagree";
}

export function formatSigned(value, digits = 2) {
  if (value == null || !Number.isFinite(value)) return "—";
  const fixed = Math.abs(value).toFixed(digits);
  return value >= 0 ? `+${fixed}` : `−${fixed}`;
}

export function formatPercent(fraction, digits = 1) {
  if (fraction == null || !Number.isFinite(fraction)) return "—";
  return `${(fraction * 100).toFixed(digits)}%`;
}

export const METRIC_ROWS = [
  { key: "accuracy", label: "Accuracy" },
  { key: "recall", label: "Sensitivity", hint: "recall: dementia caught" },
  { key: "specificity", label: "Specificity", hint: "healthy correctly cleared" },
  { key: "f1", label: "F1 score" },
  { key: "roc_auc", label: "ROC-AUC", hint: "ranking quality" },
];
