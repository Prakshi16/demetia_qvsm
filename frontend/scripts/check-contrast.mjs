// WCAG 2.2 contrast check for the design tokens, in both themes.
//
//   npm run check:contrast
//
// Resolves every semantic token through its var() chain to a primitive,
// composites translucent backgrounds over the surface they sit on, and checks
// the foreground/background pairs the UI actually uses. Fails (exit 1) if any
// pair is below its threshold: 4.5:1 for text, 3:1 for large text and UI
// boundaries.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const css = readFileSync(fileURLToPath(new URL("../src/styles/tokens.css", import.meta.url)), "utf8");

function declarations(block) {
  const out = {};
  for (const match of block.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)) out[match[1]] = match[2].trim();
  return out;
}

function blockAfter(selector, from = 0) {
  const start = css.indexOf(selector, from);
  if (start < 0) throw new Error(`selector not found: ${selector}`);
  const open = css.indexOf("{", start);
  let depth = 0;
  for (let i = open; i < css.length; i += 1) {
    if (css[i] === "{") depth += 1;
    if (css[i] === "}") depth -= 1;
    if (depth === 0) return { text: css.slice(open + 1, i), end: i };
  }
  throw new Error("unbalanced braces");
}

const primitives = blockAfter(":root {");
const light = blockAfter(":root {", primitives.end);
const dark = blockAfter(':root[data-theme="dark"] {');
const base = { ...declarations(primitives.text), ...declarations(light.text) };
const themes = { light: base, dark: { ...base, ...declarations(dark.text) } };

function resolve(vars, value, depth = 0) {
  if (depth > 20) throw new Error(`var() cycle at ${value}`);
  const ref = value.match(/^var\((--[\w-]+)\)$/);
  if (ref) return resolve(vars, vars[ref[1]], depth + 1);
  return value;
}

function parseColor(value) {
  let m = value.match(/^#([0-9a-f]{6})$/i);
  if (m) {
    const n = parseInt(m[1], 16);
    return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255, a: 1 };
  }
  m = value.match(/^rgb\(\s*(\d+)\s+(\d+)\s+(\d+)\s*(?:\/\s*([\d.]+))?\s*\)$/);
  if (m) return { r: +m[1], g: +m[2], b: +m[3], a: m[4] === undefined ? 1 : +m[4] };
  throw new Error(`can't parse colour: ${value}`);
}

function over(fg, bg) {
  const a = fg.a;
  return { r: fg.r * a + bg.r * (1 - a), g: fg.g * a + bg.g * (1 - a), b: fg.b * a + bg.b * (1 - a), a: 1 };
}

function luminance({ r, g, b }) {
  const channel = (c) => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
}

function ratio(a, b) {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

// [foreground, background, sits-on (for translucent bg), minimum, where]
const PAIRS = [
  ["--color-text", "--color-bg", null, 4.5, "body text on page"],
  ["--color-text", "--color-surface", null, 4.5, "body text on card"],
  ["--color-text", "--color-surface-sunken", null, 4.5, "text in wells"],
  ["--color-text-muted", "--color-bg", null, 4.5, "muted text on page"],
  ["--color-text-muted", "--color-surface", null, 4.5, "muted text on card"],
  ["--color-text-muted", "--color-surface-sunken", null, 4.5, "muted text in wells"],
  ["--color-text-link", "--color-surface", null, 4.5, "links"],
  ["--color-on-accent", "--color-accent", null, 4.5, "primary button"],
  ["--color-on-accent", "--color-danger-solid", null, 4.5, "destructive button"],
  ["--color-accent-text", "--color-accent-soft", "--color-surface", 4.5, "secondary button / info"],
  ["--color-success-fg", "--color-success-bg", "--color-surface", 4.5, "success badge"],
  ["--color-warning-fg", "--color-warning-bg", "--color-surface", 4.5, "warning badge"],
  ["--color-danger-fg", "--color-danger-bg", "--color-surface", 4.5, "error message"],
  ["--color-info-fg", "--color-info-bg", "--color-surface", 4.5, "info badge"],
  ["--color-neutral-fg", "--color-neutral-bg", "--color-surface", 4.5, "neutral badge"],
  ["--color-text", "--color-info-bg", "--color-surface", 4.5, "notice body"],
  ["--color-model-svm", "--color-model-svm-soft", "--color-surface", 4.5, "SVM caption on verdict"],
  ["--color-model-qsvm", "--color-model-qsvm-soft", "--color-surface", 4.5, "QSVM caption on verdict"],
  ["--color-model-qsvm", "--color-surface", null, 4.5, "QSVM badge text"],
  ["--color-text-muted", "--color-model-svm-soft", "--color-surface", 4.5, "muted text on SVM verdict"],
  ["--color-text-muted", "--color-model-qsvm-soft", "--color-surface", 4.5, "muted text on QSVM verdict"],
  ["--color-class-positive", "--color-model-svm-soft", "--color-surface", 3, "Demented verdict (large)"],
  ["--color-class-negative", "--color-model-svm-soft", "--color-surface", 3, "Nondemented verdict (large)"],
  ["--color-class-positive", "--color-model-qsvm-soft", "--color-surface", 3, "Demented verdict on QSVM (large)"],
  ["--color-class-negative", "--color-model-qsvm-soft", "--color-surface", 3, "Nondemented verdict on QSVM (large)"],
  ["--color-on-model", "--color-model-svm", null, 4.5, "SVM marker tag"],
  ["--color-on-model", "--color-model-qsvm", null, 4.5, "QSVM marker tag"],
  ["--color-border-strong", "--color-surface", null, 1.5, "field border (supplementary)"],
  ["--color-focus-ring", "--color-surface", null, 3, "focus ring"],
  ["--color-model-svm", "--color-surface-sunken", null, 3, "SVM bar on track"],
  ["--color-model-qsvm", "--color-surface-sunken", null, 3, "QSVM bar on track"],
];

let failures = 0;
for (const [theme, vars] of Object.entries(themes)) {
  console.log(`\n${theme} theme`);
  for (const [fgName, bgName, onName, min, where] of PAIRS) {
    const fg = parseColor(resolve(vars, `var(${fgName})`));
    let bg = parseColor(resolve(vars, `var(${bgName})`));
    if (bg.a < 1) bg = over(bg, parseColor(resolve(vars, `var(${onName ?? "--color-surface"})`)));
    const fgSolid = fg.a < 1 ? over(fg, bg) : fg;
    const r = ratio(fgSolid, bg);
    const ok = r >= min;
    if (!ok) failures += 1;
    console.log(`  ${ok ? "pass" : "FAIL"}  ${r.toFixed(2).padStart(5)}:1  (needs ${min})  ${where}`);
  }
}
console.log(failures ? `\n${failures} pair(s) below threshold` : "\nAll pairs pass.");
process.exit(failures ? 1 : 0);
