import assert from "node:assert/strict";
import test from "node:test";

import { validateFiles } from "./mriFiles.js";
import {
  AXIS_EXTENT,
  formatSigned,
  isOffScale,
  markerPercent,
  modelAgreement,
  modelOutputs,
  signedDecision,
} from "./prediction.js";

test("decision value sign follows the label", () => {
  assert.equal(signedDecision("Demented", 0.8), 0.8);
  assert.equal(signedDecision("Nondemented", 0.8), -0.8);
  assert.equal(signedDecision("Demented", -0.8), 0.8); // stored as a magnitude
  assert.equal(signedDecision(null, 0.8), null);
  assert.equal(signedDecision("Demented", null), null);
  assert.equal(signedDecision("Demented", Number.NaN), null);
});

test("marker sits at 50% on the boundary and clamps at the axis ends", () => {
  assert.equal(markerPercent(0, 2), 50);
  assert.equal(markerPercent(2 * AXIS_EXTENT, 2), 100);
  assert.equal(markerPercent(-2 * AXIS_EXTENT, 2), 0);
  assert.equal(markerPercent(99, 2), 100);
  assert.equal(markerPercent(null, 2), 50);
  assert.equal(markerPercent(1, 0), 50); // a bad scale never produces NaN
  assert.ok(isOffScale(99, 2));
  assert.ok(!isOffScale(1, 2));
});

test("current column layout: model_* is the SVM, qsvm_* the QSVM", () => {
  const out = modelOutputs({
    model_prediction: "Demented", model_confidence: 0.5,
    qsvm_prediction: "Nondemented", qsvm_confidence: 0.2,
  });
  assert.equal(out.legacy, false);
  assert.equal(out.svm.signed, 0.5);
  assert.equal(out.qsvm.signed, -0.2);
  assert.equal(modelAgreement(out), "disagree");
});

test("pre-2026-09-25 layout: model_* was the QSVM, svm_* the SVM", () => {
  const out = modelOutputs({
    model_prediction: "Demented", model_confidence: 0.3,
    svm_prediction: "Demented", svm_confidence: 1.1,
  });
  assert.equal(out.legacy, true);
  assert.equal(out.svm.confidence, 1.1);
  assert.equal(out.qsvm.confidence, 0.3);
  assert.equal(modelAgreement(out), "agree");
});

test("SVM-only visits have no QSVM and no agreement verdict", () => {
  const out = modelOutputs({ model_prediction: "Nondemented", model_confidence: 1 });
  assert.equal(out.qsvm, null);
  assert.equal(modelAgreement(out), null);
  assert.deepEqual(modelOutputs(null), { svm: null, qsvm: null, legacy: false });
});

test("formatSigned uses a real minus sign", () => {
  assert.equal(formatSigned(0.5), "+0.50");
  assert.equal(formatSigned(-0.5), "−0.50");
  assert.equal(formatSigned(null), "—");
});

const file = (name, size = 10) => ({ name, size });

test("MRI upload: accepted single files", () => {
  for (const name of ["a.nii", "a.nii.gz", "a.mgz", "a.dcm", "a.zip", "slice.png", "slice.JPG", "s.tiff"]) {
    assert.equal(validateFiles([file(name)]), "", name);
  }
});

test("MRI upload: rejected inputs", () => {
  assert.match(validateFiles([]), /Select/);
  assert.match(validateFiles([file("notes.txt")]), /Unsupported/);
  assert.match(validateFiles([file("scan.img")]), /\.hdr and the \.img/);
  assert.match(validateFiles([file("a.nii", 0)]), /empty/);
  assert.match(validateFiles([file("a.nii", 60 * 1024 * 1024)]), /limit/);
});

test("MRI upload: multi-file scans (Analyze pair, DICOM series without extensions)", () => {
  assert.equal(validateFiles([file("s.hdr"), file("s.img")]), "");
  assert.equal(validateFiles([file("IM0001"), file("IM0002"), file("IM0003")]), "");
});
