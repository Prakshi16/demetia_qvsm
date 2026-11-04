// Run with: npm test   (node's built-in test runner, no extra dependencies)
import assert from "node:assert/strict";
import test from "node:test";

import {
  CDR_OPTIONS,
  CLINICAL_RULES,
  digitsOnly,
  phoneCharsOnly,
  validateChoice,
  validateDob,
  validatePersonName,
  validatePhone,
  validatePincode,
  validateRelationship,
  validateWholeNumber,
} from "./validate.js";

test("digitsOnly strips everything that isn't a digit and caps the length", () => {
  assert.equal(digitsOnly("2a8"), "28");
  assert.equal(digitsOnly("-5"), "5");
  assert.equal(digitsOnly("1e3"), "13");
  assert.equal(digitsOnly("2.5"), "25");
  assert.equal(digitsOnly(" 30 "), "30");
  assert.equal(digitsOnly("12345", 2), "12");
  assert.equal(digitsOnly("٣٠"), ""); // non-ASCII digits are not accepted
  assert.equal(digitsOnly(null), "");
});

test("MMSE accepts whole numbers 0-30 only", () => {
  const mmse = CLINICAL_RULES.mmse;
  for (const ok of ["0", "12", "30"]) assert.equal(validateWholeNumber(ok, mmse), null, ok);
  assert.match(validateWholeNumber("", mmse), /required/);
  assert.match(validateWholeNumber("31", mmse), /between 0 and 30/);
  for (const bad of ["-1", "2.5", "1e1", "abc", "12a", "+5", "0x1"]) {
    assert.match(validateWholeNumber(bad, mmse), /whole number/, bad);
  }
});

test("education accepts whole years 0-30", () => {
  const edu = CLINICAL_RULES.edu;
  assert.equal(validateWholeNumber("16", edu), null);
  assert.match(validateWholeNumber("45", edu), /between 0 and 30/);
  assert.match(validateWholeNumber("12.5", edu), /whole number/);
});

test("CDR must be a real scale point", () => {
  for (const ok of CDR_OPTIONS) assert.equal(validateChoice(ok, { label: "CDR", options: CDR_OPTIONS }), null);
  assert.match(validateChoice("", { label: "CDR", options: CDR_OPTIONS }), /Select/);
  assert.ok(validateChoice("0.7", { label: "CDR", options: CDR_OPTIONS }));
  assert.ok(validateChoice("4", { label: "CDR", options: CDR_OPTIONS }));
});

test("phone allows separators but needs 7-15 digits", () => {
  assert.equal(validatePhone(""), null);
  assert.equal(validatePhone("+91 98765 43210"), null);
  assert.equal(validatePhone("(011) 2345-6789"), null);
  assert.match(validatePhone("12345"), /between 7 and 15/);
  assert.match(validatePhone("98765abc12"), /only digits/);
  assert.equal(phoneCharsOnly("98a76-5x"), "9876-5");
});

test("date of birth: not in the future, not over 120 years ago", () => {
  const today = new Date(2026, 10, 4);
  assert.equal(validateDob("", today), null);
  assert.equal(validateDob("1950-06-01", today), null);
  assert.match(validateDob("2027-01-01", today), /future/);
  assert.match(validateDob("1890-01-01", today), /120 years/);
  assert.match(validateDob("1950-13-45", today), /real date|YYYY/);
});

test("guardian relationship is required for guardian consent and must be words", () => {
  assert.match(validateRelationship("", { required: true }), /guardian/);
  assert.equal(validateRelationship("", { required: false }), null);
  assert.equal(validateRelationship("Daughter", { required: true }), null);
  assert.equal(validateRelationship("Legal guardian", { required: true }), null);
  assert.match(validateRelationship("123", { required: true }), /words only/);
});

test("names and pincodes", () => {
  assert.equal(validatePersonName("Asha Rao"), null);
  assert.ok(validatePersonName("12345"));
  assert.ok(validatePersonName("A"));
  assert.equal(validatePincode("560001"), null);
  assert.ok(validatePincode("56001"));
  assert.ok(validatePincode("56000a"));
});
