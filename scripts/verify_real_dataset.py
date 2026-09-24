"""Phase 1.7 — automated dataset verification GATE (GO/NO-GO).

Per real_dataset_setup.md §Phase 1.7. The spec originally split this gate into an
automated half (Prakshi) and a manual half (Sheetal: hand-check 25 random rows against
the OASIS xlsx + listen to clips, sign off in DATASHEET.md). Track C is no longer
staffed, so this script is now the *entire* gate — there is no human eyeball step.
That substitution is deliberate (agreed with the user 2026-09-24) and is recorded in
DATASHEET.md's sign-off log rather than silently dropped.

Checks, each printed as [PASS]/[WARN]/[FAIL]:
  1. schema        — exact column set, row count, no NaN in feature columns, Label in {0,1}.
  2. ranges         — MMSE 0-30, nWBV 0.6-0.85, eTIV 1100-2100, ASF 0.7-1.6, all mfcc finite.
  3. duplication    — every assembled row is a distinct (speaker, clip, window); no OASIS
                       subject reused (matcher assigns without replacement); speaker reuse
                       capped at --r-max.
  4. leak probe     — depth-1 DecisionTreeClassifier per clinical/MRI feature vs Label
                       (resubstitution accuracy, most sensitive to a hard leak). Flag >= 0.95.
  5. representativeness — matched-OASIS-subset vs full oasis_pool.csv, mean_diff_in_std_units
                       per clinical/MRI feature (same metric as validation_checks.ipynb Task 1).
                       WARN > 0.3 std, FAIL > 1.0 std.
  6. age QC         — every row's oasis_age falls in [age_lo, age_hi]; report the
                       |oasis_age - speaker_age_final| distribution per (label, sex) stratum
                       and per age_source, against the tol=8/12/15(widened) design.

Exit code 0 iff there is no FAIL. WARNs are printed but do not block.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"

CLINICAL_COLS = ["MMSE", "ASF", "EDUC", "SES"]
MRI_COLS = ["nWBV", "eTIV"]
SPEECH_COLS = [
    "pause_rate", "speech_rate", "pitch_mean", "jitter", "shimmer",
    *[f"mfcc_{i}" for i in range(1, 14)],
]
DATASET_COLUMNS = ["Subject_ID", *CLINICAL_COLS, *MRI_COLS, *SPEECH_COLS, "Label"]

RANGES = {
    "MMSE": (0, 30),
    "nWBV": (0.6, 0.85),
    "eTIV": (1100, 2100),
    "ASF": (0.7, 1.6),
}

LEAK_STUMP_THRESHOLD = 0.95
REPRESENTATIVENESS_WARN = 0.3
REPRESENTATIVENESS_FAIL = 1.0
AGE_GAP_HARD_MAX = 15  # widened tol ceiling per Appendix B


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failed = False

    def _emit(self, status: str, msg: str) -> None:
        line = f"[{status}] {msg}"
        print(line)
        self.lines.append(line)
        if status == "FAIL":
            self.failed = True

    def ok(self, msg: str) -> None:
        self._emit("PASS", msg)

    def warn(self, msg: str) -> None:
        self._emit("WARN", msg)

    def fail(self, msg: str) -> None:
        self._emit("FAIL", msg)

    def section(self, title: str) -> None:
        header = f"\n=== {title} ==="
        print(header)
        self.lines.append(header)


def check_schema(df: pd.DataFrame, r: Report) -> None:
    r.section("1. Schema")
    actual = df.columns.tolist()
    if actual != DATASET_COLUMNS:
        missing = set(DATASET_COLUMNS) - set(actual)
        extra = set(actual) - set(DATASET_COLUMNS)
        r.fail(f"column set mismatch — missing={sorted(missing)} extra={sorted(extra)}")
    else:
        r.ok(f"columns match the Appendix A contract exactly ({len(actual)} cols)")

    if len(df) == 0:
        r.fail("dataset has 0 rows")
    else:
        r.ok(f"row count = {len(df)}")

    feature_cols = [c for c in DATASET_COLUMNS if c not in ("Subject_ID", "Label")]
    present_feature_cols = [c for c in feature_cols if c in df.columns]
    n_nan = df[present_feature_cols].isna().sum().sum()
    if n_nan:
        bad = df[present_feature_cols].isna().sum()
        r.fail(f"{n_nan} NaN values in feature columns: {bad[bad > 0].to_dict()}")
    else:
        r.ok("no NaN in any feature column")

    if "Label" in df.columns:
        bad_labels = set(df["Label"].unique()) - {0, 1}
        if bad_labels:
            r.fail(f"Label has values outside {{0,1}}: {bad_labels}")
        else:
            counts = df["Label"].value_counts().to_dict()
            r.ok(f"Label in {{0,1}}, counts={counts}")


def check_ranges(df: pd.DataFrame, r: Report) -> None:
    r.section("2. Feature ranges")
    for col, (lo, hi) in RANGES.items():
        if col not in df.columns:
            continue
        out = df[(df[col] < lo) | (df[col] > hi)]
        if len(out):
            r.fail(f"{col}: {len(out)} row(s) outside [{lo}, {hi}] — "
                    f"e.g. Subject_ID={out['Subject_ID'].iloc[0]!r} {col}={out[col].iloc[0]}")
        else:
            r.ok(f"{col}: all values within [{lo}, {hi}]")

    mfcc_cols = [c for c in df.columns if c.startswith("mfcc_")]
    finite = np.isfinite(df[mfcc_cols].to_numpy()).all()
    if finite:
        r.ok(f"all {len(mfcc_cols)} mfcc columns finite")
    else:
        r.fail("non-finite values found in mfcc columns")


def check_duplication(prov: pd.DataFrame, r: Report, r_max: int) -> None:
    r.section("3. Duplication / reuse sanity")
    window_key = prov[["speaker_id", "clip_path", "window_idx"]]
    n_distinct_windows = window_key.drop_duplicates().shape[0]
    if n_distinct_windows != len(prov):
        r.fail(f"{len(prov) - n_distinct_windows} row(s) reuse an identical "
                f"(speaker_id, clip_path, window_idx) window")
    else:
        r.ok(f"all {len(prov)} rows are distinct speech windows")

    oasis_reuse = prov["oasis_subject_id"].value_counts()
    max_oasis_reuse = int(oasis_reuse.max()) if len(oasis_reuse) else 0
    if max_oasis_reuse > 1:
        r.fail(f"OASIS subject(s) reused across rows (max reuse={max_oasis_reuse}) — "
               f"matcher is supposed to assign without replacement on the OASIS side")
    else:
        r.ok(f"every OASIS subject used at most once ({prov['oasis_subject_id'].nunique()} distinct)")

    speaker_reuse = prov["speaker_id"].value_counts()
    max_speaker_reuse = int(speaker_reuse.max()) if len(speaker_reuse) else 0
    if max_speaker_reuse > r_max:
        r.fail(f"speaker reuse {max_speaker_reuse} exceeds R_max={r_max} "
               f"(speaker={speaker_reuse.idxmax()})")
    else:
        r.ok(f"max speaker reuse = {max_speaker_reuse} (R_max={r_max}); "
             f"{prov['speaker_id'].nunique()} distinct speakers used")

    if "no_contradiction" in prov.columns:
        if not bool(prov["no_contradiction"].all()):
            r.fail("provenance has row(s) with no_contradiction == False")
        else:
            r.ok("no_contradiction holds for every row")


def check_leak_probe(df: pd.DataFrame, r: Report) -> None:
    r.section("4. Leak probe (depth-1 stump per clinical/MRI feature)")
    y = df["Label"].to_numpy()
    for col in [*CLINICAL_COLS, *MRI_COLS]:
        if col not in df.columns:
            continue
        X = df[[col]].to_numpy()
        clf = DecisionTreeClassifier(max_depth=1, random_state=42)
        clf.fit(X, y)
        acc = clf.score(X, y)
        if acc >= LEAK_STUMP_THRESHOLD:
            r.fail(f"{col}: single-feature stump resub accuracy {acc:.3f} >= "
                   f"{LEAK_STUMP_THRESHOLD} — looks like a label leak, not real signal")
        else:
            r.ok(f"{col}: stump accuracy {acc:.3f} (< {LEAK_STUMP_THRESHOLD})")


def check_representativeness(prov: pd.DataFrame, pool: pd.DataFrame, r: Report) -> None:
    r.section("5. Representativeness — matched OASIS subset vs full oasis_pool")
    matched_ids = set(prov["oasis_subject_id"])
    subset = pool[pool["oasis_subject_id"].isin(matched_ids)]
    r.ok(f"matched subset = {len(subset)}/{len(pool)} OASIS subjects "
         f"({len(subset) / len(pool):.1%})")

    rows = []
    for feat in [*CLINICAL_COLS, *MRI_COLS]:
        full = pool[feat].dropna()
        sub = subset[feat].dropna()
        fm, fs = full.mean(), full.std()
        sm, ss = sub.mean(), sub.std()
        pooled = np.sqrt((fs**2 + ss**2) / 2)
        diff = abs(fm - sm) / pooled if pooled > 0 else np.nan
        rows.append((feat, fm, fs, sm, ss, diff))

    tbl = pd.DataFrame(rows, columns=["feature", "pool_mean", "pool_std",
                                       "matched_mean", "matched_std", "mean_diff_in_std_units"])
    print(tbl.to_string(index=False))
    r.lines.append(tbl.to_string(index=False))

    for _, row in tbl.iterrows():
        d = row["mean_diff_in_std_units"]
        if pd.isna(d):
            continue
        if d > REPRESENTATIVENESS_FAIL:
            r.fail(f"{row['feature']}: {d:.2f} std divergence (> {REPRESENTATIVENESS_FAIL}) — "
                   f"matched subset is not representative of the full pool")
        elif d > REPRESENTATIVENESS_WARN:
            r.warn(f"{row['feature']}: {d:.2f} std divergence (> {REPRESENTATIVENESS_WARN}, "
                   f"<= {REPRESENTATIVENESS_FAIL})")
        else:
            r.ok(f"{row['feature']}: {d:.2f} std divergence (good match)")


def check_age_qc(prov: pd.DataFrame, r: Report) -> None:
    r.section("6. Age QC")
    violations = prov[(prov["oasis_age"] < prov["age_lo"]) | (prov["oasis_age"] > prov["age_hi"])]
    if len(violations):
        r.fail(f"{len(violations)} row(s) have oasis_age outside [age_lo, age_hi] — "
               f"e.g. {violations['Subject_ID'].iloc[0]!r}")
    else:
        r.ok(f"all {len(prov)} rows satisfy age_lo <= oasis_age <= age_hi")

    hard_violations = prov[prov["age_gap"] > AGE_GAP_HARD_MAX]
    if len(hard_violations):
        r.fail(f"{len(hard_violations)} row(s) exceed the widened age_gap ceiling "
               f"({AGE_GAP_HARD_MAX})")
    else:
        r.ok(f"max age_gap = {prov['age_gap'].max():.1f} (<= {AGE_GAP_HARD_MAX})")

    print("\nage_gap by (label, sex) stratum:")
    strat = prov.groupby(["label", "sex"])["age_gap"].agg(["count", "mean", "median", "max"])
    print(strat.to_string())
    r.lines.append(strat.to_string())

    print("\nage_gap by speaker_age_source:")
    src = prov.groupby("speaker_age_source")["age_gap"].agg(["count", "mean", "median", "max"])
    print(src.to_string())
    r.lines.append(src.to_string())

    widened_n = int(prov["window_widened"].sum())
    print(f"\nwindow_widened rows: {widened_n}/{len(prov)}")
    r.lines.append(f"window_widened rows: {widened_n}/{len(prov)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DATA_DIR / "multimodal_dementia_dataset.csv")
    ap.add_argument("--provenance", type=Path, default=DATA_DIR / "multimodal_real_provenance.csv")
    ap.add_argument("--pool", type=Path, default=DATA_DIR / "oasis_pool.csv")
    ap.add_argument("--r-max", type=int, default=18)
    ap.add_argument("--report-out", type=Path, default=DATA_DIR / "phase17_verification_report.md")
    args = ap.parse_args()

    df = pd.read_csv(args.dataset)
    prov = pd.read_csv(args.provenance)
    pool = pd.read_csv(args.pool)

    r = Report()
    check_schema(df, r)
    check_ranges(df, r)
    check_duplication(prov, r, args.r_max)
    check_leak_probe(df, r)
    check_representativeness(prov, pool, r)
    check_age_qc(prov, r)

    r.section("VERDICT")
    if r.failed:
        r.fail("Phase 1.7 gate: NO-GO — see FAIL lines above")
    else:
        r.ok("Phase 1.7 gate: GO — no FAIL, proceed to Phase 1.9")

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text("\n".join(r.lines) + "\n", encoding="utf-8")
    print(f"\nFull report written to {args.report_out}")

    return 1 if r.failed else 0


if __name__ == "__main__":
    sys.exit(main())
