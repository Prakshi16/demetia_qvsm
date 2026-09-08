"""Build the OASIS pool and deterministically match it to speech windows.

The script has two independent stages:

* harmonise the OASIS-1/OASIS-2 workbooks, run the label-blind imputation
  bake-off, and write ``data/oasis_pool.csv``;
* when ``speakers_meta.csv`` is available, match speech windows to OASIS
  subjects and write the real multimodal dataset plus its provenance.

All generated files live under ``data/``. That directory is intentionally
gitignored because the OASIS source material is DUA-bound.

Run from the repository root:

    python scripts/build_real_dataset.py

The first run can build the pool immediately. The matcher runs automatically
once ``data/speakers_meta.csv`` and the speech-window CSVs are present.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

SPEECH_FEATURES = [
    "pause_rate",
    "speech_rate",
    "pitch_mean",
    "jitter",
    "shimmer",
    *[f"mfcc_{i}" for i in range(1, 14)],
]
POOL_COLUMNS = [
    "oasis_source",
    "oasis_subject_id",
    "mri_id",
    "label",
    "sex",
    "age",
    "MMSE",
    "ASF",
    "EDUC",
    "SES",
    "nWBV",
    "eTIV",
    "CDR",
    "imputed_flags",
]
MODEL_COLUMNS = ["MMSE", "ASF", "EDUC", "SES", "nWBV", "eTIV"]
CLINICAL_COLUMNS = ["MMSE", "EDUC", "SES"]
IMPUTATION_PREDICTORS = ["age", "EDUC", "eTIV", "nWBV", "ASF"]
S3_IMPUTER_COLUMNS = ["age", "sex_code", "EDUC", "eTIV", "nWBV", "ASF", "MMSE", "SES"]
IMPUTE_TARGETS = ["MMSE", "EDUC", "SES"]
PROVENANCE_COLUMNS = [
    "Subject_ID",
    "oasis_source",
    "oasis_subject_id",
    "oasis_age",
    "label",
    "sex",
    "speaker_id",
    "clip_path",
    "window_idx",
    "speaker_age_final",
    "speaker_age_source",
    "age_lo",
    "age_hi",
    "age_gap",
    "window_widened",
    "speaker_row_count",
    "oasis_reuse_count",
    "group_id",
    "orig_split",
    "no_contradiction",
]
DATASET_COLUMNS = [
    "Subject_ID",
    "MMSE",
    "ASF",
    "EDUC",
    "SES",
    "nWBV",
    "eTIV",
    *SPEECH_FEATURES,
    "Label",
]


def _normalise_column_name(name: object) -> str:
    value = str(name).strip().replace("/", "_")
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9_]+", "", value)
    return value.strip("_")


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [_normalise_column_name(column) for column in result.columns]
    return result


def _find_column(frame: pd.DataFrame, *names: str) -> str:
    normalised = {_normalise_column_name(name) for name in names}
    for column in frame.columns:
        if _normalise_column_name(column) in normalised:
            return column
    raise ValueError(f"Missing one of columns {sorted(normalised)}; found {list(frame.columns)}")


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def _normalise_sex(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.upper()
    values = values.replace({"MALE": "M", "FEMALE": "F"})
    values = values.where(values.isin(["M", "F"]), pd.NA)
    return values.astype("string")


def _empty_standard_rows(source: str, frame: pd.DataFrame) -> pd.DataFrame:
    mri_col = _find_column(frame, "MRI_ID", "ID")
    subject_col = _find_column(frame, "Subject_ID", "SubjectID") if source == "oasis2" else None
    sex_col = _find_column(frame, "M_F", "MF", "Sex")
    age_col = _find_column(frame, "Age")
    educ_col = _find_column(frame, "EDUC", "Educ")
    result = pd.DataFrame(index=frame.index)
    result["oasis_source"] = source
    result["oasis_subject_id"] = (
        frame[subject_col].astype("string").str.strip()
        if subject_col
        else frame[mri_col].astype("string").str.replace(r"_MR1$", "", regex=True)
    )
    result["mri_id"] = frame[mri_col].astype("string").str.strip()
    result["sex"] = _normalise_sex(frame[sex_col])
    result["age"] = _numeric(frame[age_col])
    result["EDUC"] = _numeric(frame[educ_col])
    for field in ["MMSE", "SES", "nWBV", "eTIV", "ASF", "CDR"]:
        source_col = _find_column(frame, field)
        result[field] = _numeric(frame[source_col])
    result["label"] = (result["CDR"] > 0).astype("Int64")
    return result


def load_oasis_tables(oasis1_path: Path, oasis2_path: Path) -> pd.DataFrame:
    """Read both workbooks and return one un-imputed, subject-level table."""
    oasis2 = _normalise_columns(pd.read_excel(oasis2_path))
    subject_col = _find_column(oasis2, "Subject_ID", "SubjectID")
    visit_col = _find_column(oasis2, "Visit")
    age_col = _find_column(oasis2, "Age")
    cdr_col = _find_column(oasis2, "CDR")
    group_col = _find_column(oasis2, "Group")
    oasis2["_visit_order"] = _numeric(oasis2[visit_col]).fillna(-np.inf)
    oasis2["_age_order"] = _numeric(oasis2[age_col]).fillna(-np.inf)

    # The handoff defines a Converted subject by its final demented phase.
    # The supplied workbook contains OAS2_0131 with a later CDR=0 row after
    # an earlier CDR=0.5 row, so selecting the final row blindly violates the
    # subject-level Converted label. Select the latest positive-CDR phase for
    # every Converted subject, and fail loudly if one has no such phase.
    is_converted = oasis2[group_col].astype("string").str.strip().str.casefold().eq("converted")
    converted_ids = set(oasis2.loc[is_converted, subject_col].astype(str))
    converted_demented = oasis2[is_converted & (_numeric(oasis2[cdr_col]) > 0)]
    missing_demented = converted_ids - set(converted_demented[subject_col].astype(str))
    if missing_demented:
        raise ValueError(
            "Converted subject(s) have no demented phase (CDR > 0): "
            + ", ".join(sorted(missing_demented))
        )
    ordinary = oasis2[~is_converted]
    oasis2 = pd.concat([ordinary, converted_demented], ignore_index=True)
    oasis2 = oasis2.sort_values(
        [subject_col, "_visit_order", "_age_order"],
        ascending=[True, True, True],
        kind="mergesort",
    ).drop_duplicates(subject_col, keep="last")
    oasis2_rows = _empty_standard_rows("oasis2", oasis2)

    oasis1 = _normalise_columns(pd.read_excel(oasis1_path))
    mri_col = _find_column(oasis1, "ID")
    oasis1 = oasis1[oasis1[mri_col].astype("string").str.upper().str.endswith("_MR1", na=False)]
    cdr_col = _find_column(oasis1, "CDR")
    oasis1 = oasis1[oasis1[cdr_col].notna()].copy()
    oasis1_rows = _empty_standard_rows("oasis1", oasis1)

    result = pd.concat([oasis1_rows, oasis2_rows], ignore_index=True)
    result["label"] = result["label"].astype("Int64")
    return result.sort_values(
        ["oasis_source", "oasis_subject_id", "age"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def _sex_code(series: pd.Series) -> pd.Series:
    return series.map({"F": 0.0, "M": 1.0}).astype(float)


def _group_median_fill(frame: pd.DataFrame, field: str) -> None:
    age_band = np.floor(frame["age"] / 10.0) * 10.0
    grouped = frame.assign(_age_band=age_band)
    by_band = grouped.groupby(["sex", "_age_band"], dropna=False)[field].transform("median")
    by_sex = grouped.groupby("sex", dropna=False)[field].transform("median")
    fallback = frame[field].median()
    frame[field] = frame[field].fillna(by_band).fillna(by_sex).fillna(fallback)


def apply_imputation(frame: pd.DataFrame, scheme: str) -> pd.DataFrame:
    """Apply one label-blind scheme without changing label/CDR/Group columns."""
    scheme = scheme.upper()
    result = frame.copy()
    if scheme == "S0":
        return result.dropna(subset=CLINICAL_COLUMNS).copy()

    if scheme == "S1":
        result = result.dropna(subset=["MMSE"]).copy()
        for field in ["SES", "EDUC"]:
            result[field] = result[field].fillna(result[field].median())
        return result

    if scheme in {"S2", "S4"}:
        for field in IMPUTE_TARGETS:
            _group_median_fill(result, field)
        if scheme == "S2":
            return result

        # S4 keeps S2 as its deterministic fallback, then replaces the two
        # targeted estimates where a regression has enough complete rows.
        regression_frame = result.copy()
        for field in IMPUTATION_PREDICTORS:
            regression_frame[field] = regression_frame[field].fillna(regression_frame[field].median())
        sex = _sex_code(regression_frame["sex"])
        regressions = {
            "SES": ["EDUC", "age", "sex"],
            "MMSE": ["EDUC", "age", "nWBV"],
        }
        for target, predictors in regressions.items():
            predictor_values = regression_frame[predictors].copy()
            if "sex" in predictor_values:
                predictor_values["sex"] = sex
            known = frame[target].notna()
            complete = known & predictor_values.notna().all(axis=1)
            if int(complete.sum()) < 3:
                continue
            model = LinearRegression().fit(predictor_values.loc[complete], frame.loc[complete, target])
            missing = frame[target].isna() & predictor_values.notna().all(axis=1)
            if not missing.any():
                continue
            prediction = model.predict(predictor_values.loc[missing])
            if target == "SES":
                observed = frame[target].dropna()
                prediction = np.rint(prediction).clip(observed.min(), observed.max())
            result.loc[missing, target] = prediction
        return result

    if scheme == "S3":
        numeric_fields = ["age", "EDUC", "eTIV", "nWBV", "ASF", "MMSE", "SES"]
        matrix = result[numeric_fields].copy()
        matrix.insert(1, "sex_code", _sex_code(result["sex"]))
        imputer = IterativeImputer(random_state=42, max_iter=20, initial_strategy="median")
        transformed = imputer.fit_transform(matrix)
        transformed = pd.DataFrame(transformed, columns=S3_IMPUTER_COLUMNS, index=result.index)
        for field in numeric_fields:
            result[field] = transformed[field]
        return result

    raise ValueError(f"Unknown imputation scheme: {scheme}")


def _mask_known_values(frame: pd.DataFrame, field: str, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    masked = frame.copy()
    known = masked[field].notna()
    indices = masked.index[known]
    count = max(1, int(np.floor(len(indices) * 0.20))) if len(indices) else 0
    rng = np.random.RandomState(seed)
    selected = pd.Series(False, index=masked.index)
    if count:
        chosen = rng.choice(indices.to_numpy(), size=count, replace=False)
        selected.loc[chosen] = True
        masked.loc[chosen, field] = np.nan
    return masked, selected


def score_imputation_scheme(frame: pd.DataFrame, scheme: str) -> dict[str, object]:
    """Score masked known values; labels and CDR never reach the imputer."""
    metrics: dict[str, object] = {"scheme": scheme, "mae_mmse": np.nan, "mae_educ": np.nan, "macro_f1_ses": np.nan}
    coverage: dict[str, float] = {}
    masked_frames: dict[str, tuple[pd.DataFrame, pd.Series]] = {
        field: _mask_known_values(frame, field) for field in IMPUTE_TARGETS
    }
    for field, (masked, selected) in masked_frames.items():
        imputed = apply_imputation(masked, scheme)
        available = selected.index[selected].intersection(imputed.index)
        coverage[field] = float(len(available) / max(1, int(selected.sum())))
        if not len(available):
            continue
        actual = frame.loc[available, field].astype(float)
        predicted = imputed.loc[available, field].astype(float)
        if field == "SES":
            metrics["macro_f1_ses"] = float(
                f1_score(np.rint(actual), np.rint(predicted), average="macro", zero_division=0)
            )
        else:
            metrics[f"mae_{field.lower()}"] = float(np.abs(actual - predicted).mean())
    metrics["coverage"] = coverage
    valid = [metrics["mae_mmse"], metrics["mae_educ"], 1.0 - metrics["macro_f1_ses"]]
    metrics["composite_error"] = float(np.mean(valid)) if all(np.isfinite(valid)) else np.inf
    return metrics


def choose_imputation_scheme(scores: Iterable[dict[str, object]], frame: pd.DataFrame) -> str:
    """Choose the lowest masked error, leaving a downstream-CV hook for 1.7."""
    score_list = list(scores)
    if not frame[CLINICAL_COLUMNS].isna().any().any():
        return "S0"
    eligible = [row for row in score_list if np.isfinite(float(row["composite_error"]))]
    if not eligible:
        return "S2"
    return str(min(eligible, key=lambda row: (float(row["composite_error"]), str(row["scheme"])))['scheme'])


def _imputed_flags(original: pd.DataFrame, imputed: pd.DataFrame) -> pd.Series:
    flags = []
    for index in imputed.index:
        filled = [field for field in ["age", *MODEL_COLUMNS] if pd.isna(original.loc[index, field]) and pd.notna(imputed.loc[index, field])]
        flags.append(";".join(filled))
    return pd.Series(flags, index=imputed.index, dtype="string")


def write_bakeoff(path: Path, scores: list[dict[str, object]], selected: str) -> None:
    lines = [
        "# OASIS missing-value bake-off",
        "",
        "Known values were masked at 20% with seed 42. The imputation inputs were label-blind: `label`, `CDR`, and `Group` were never supplied.",
        "",
        "| Scheme | MMSE MAE | EDUC MAE | SES macro-F1 | Composite error | Coverage |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in scores:
        coverage = ", ".join(f"{field}={value:.2f}" for field, value in row["coverage"].items())
        def fmt(value: object) -> str:
            return "n/a" if not np.isfinite(float(value)) else f"{float(value):.4f}"
        lines.append(
            f"| {row['scheme']} | {fmt(row['mae_mmse'])} | {fmt(row['mae_educ'])} | {fmt(row['macro_f1_ses'])} | {fmt(row['composite_error'])} | {coverage} |"
        )
    lines.extend([
        "",
        f"**Selected scheme:** `{selected}` (lowest finite composite masked error).",
        "",
        "S0 is retained as a complete-case comparison. The downstream grouped-CV tie-break remains a Phase 1.7 hook; no label or CDR is used to select the masked-value winner.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_oasis_pool(oasis1_path: Path, oasis2_path: Path, output_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, str, list[dict[str, object]]]:
    """Build and write the frozen ``oasis_pool.csv`` contract."""
    raw = load_oasis_tables(oasis1_path, oasis2_path)
    scores = [score_imputation_scheme(raw, scheme) for scheme in ["S0", "S1", "S2", "S3", "S4"]]
    selected = choose_imputation_scheme(scores, raw)
    pool = apply_imputation(raw, selected)
    pool["imputed_flags"] = _imputed_flags(raw, pool)
    pool = pool.dropna(subset=["age", *MODEL_COLUMNS]).copy()
    pool["label"] = pool["label"].astype(int)
    pool = pool[POOL_COLUMNS].sort_values(
        ["label", "sex", "age", "oasis_subject_id"], kind="mergesort"
    ).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    pool.to_csv(output_dir / "oasis_pool.csv", index=False)
    write_bakeoff(output_dir / "oasis_imputation_bakeoff.md", scores, selected)
    return pool, selected, scores


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _prepare_speakers(speakers: pd.DataFrame) -> pd.DataFrame:
    required = ["speaker_id", "label", "sex", "age_final", "age_lo", "age_hi"]
    _require_columns(speakers, required, "speakers_meta.csv")
    result = speakers.copy()
    result["speaker_id"] = result["speaker_id"].astype("string")
    result["label"] = _numeric(result["label"]).astype("Int64")
    result["sex"] = _normalise_sex(result["sex"])
    for field in ["age_final", "age_lo", "age_hi"]:
        result[field] = _numeric(result[field])
    result["age_source"] = result.get("age_source", pd.Series("unknown", index=result.index)).fillna("unknown").astype("string")
    result["orig_split"] = result.get("orig_split", pd.Series(pd.NA, index=result.index)).astype("string")
    if result[required].isna().any().any():
        raise ValueError("speakers_meta.csv contains missing matching fields")
    return result.drop_duplicates("speaker_id", keep="first")


def _prepare_pool(pool: pd.DataFrame) -> pd.DataFrame:
    _require_columns(pool, POOL_COLUMNS, "oasis_pool.csv")
    result = pool.copy()
    result["label"] = _numeric(result["label"]).astype("Int64")
    result["sex"] = _normalise_sex(result["sex"])
    result["age"] = _numeric(result["age"])
    return result


def _prepare_windows(windows: pd.DataFrame, speakers: pd.DataFrame) -> pd.DataFrame:
    _require_columns(windows, ["speaker_id", "clip_path", "window_idx", *SPEECH_FEATURES], "speech windows")
    result = windows.copy().reset_index(drop=True)
    result["speaker_id"] = result["speaker_id"].astype("string")
    result["window_idx"] = _numeric(result["window_idx"]).astype(int)
    for field in SPEECH_FEATURES:
        result[field] = _numeric(result[field])
    result = result.merge(
        speakers[["speaker_id", "label", "sex", "age_final", "age_lo", "age_hi", "age_source", "orig_split"]],
        on="speaker_id",
        how="inner",
        validate="many_to_one",
    )
    if result.empty:
        raise ValueError("No speech windows matched speakers_meta.csv")
    return result


def match_windows(
    oasis_pool: pd.DataFrame,
    speakers_meta: pd.DataFrame,
    speech_windows: pd.DataFrame,
    r_max: int = 18,
    widened_tolerance: float = 15.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Match each OASIS subject to at most one speech window, without RNG."""
    if r_max < 1:
        raise ValueError("r_max must be at least 1")
    pool = _prepare_pool(oasis_pool)
    speakers = _prepare_speakers(speakers_meta)
    windows = _prepare_windows(speech_windows, speakers)
    used_window_rows: set[int] = set()
    speaker_counts = {str(speaker): 0 for speaker in speakers["speaker_id"]}
    matched_rows: list[dict[str, object]] = []
    matched_oasis: set[str] = set()

    for (label, sex), oasis_cell in pool.groupby(["label", "sex"], dropna=False, sort=True):
        cell_windows = windows[(windows["label"] == label) & (windows["sex"] == sex)].copy()
        oasis_cell = oasis_cell.sort_values(["age", "oasis_subject_id"], kind="mergesort")
        for oasis_index, oasis_row in oasis_cell.iterrows():
            available = cell_windows.loc[
                ~cell_windows.index.isin(used_window_rows)
                & cell_windows["speaker_id"].map(lambda value: speaker_counts[str(value)] < r_max)
            ].copy()
            if available.empty:
                continue
            strict = available[
                (available["age_lo"] <= oasis_row["age"])
                & (oasis_row["age"] <= available["age_hi"])
            ].copy()
            widened = False
            candidates = strict
            if candidates.empty:
                candidates = available[
                    (available["age_final"] - widened_tolerance <= oasis_row["age"])
                    & (oasis_row["age"] <= available["age_final"] + widened_tolerance)
                ].copy()
                widened = not candidates.empty
            if candidates.empty:
                continue
            candidates["_age_gap"] = (candidates["age_final"] - oasis_row["age"]).abs()
            candidates = candidates.sort_values(
                ["_age_gap", "speaker_id", "window_idx"], kind="mergesort"
            )
            window_index = int(candidates.index[0])
            window = candidates.iloc[0]
            speaker_id = str(window["speaker_id"])
            used_window_rows.add(window_index)
            speaker_counts[speaker_id] += 1
            oasis_id = str(oasis_row["oasis_subject_id"])
            matched_oasis.add(oasis_id)
            effective_lo = float(window["age_final"] - widened_tolerance) if widened else float(window["age_lo"])
            effective_hi = float(window["age_final"] + widened_tolerance) if widened else float(window["age_hi"])
            row: dict[str, object] = {
                "oasis_source": oasis_row["oasis_source"],
                "oasis_subject_id": oasis_id,
                "mri_id": oasis_row["mri_id"],
                "oasis_age": float(oasis_row["age"]),
                "label": int(label),
                "sex": str(sex),
                "speaker_id": speaker_id,
                "clip_path": window["clip_path"],
                "window_idx": int(window["window_idx"]),
                "speaker_age_final": float(window["age_final"]),
                "speaker_age_source": str(window["age_source"]),
                "age_lo": effective_lo,
                "age_hi": effective_hi,
                "age_gap": float(window["_age_gap"]),
                "window_widened": bool(widened),
                "speaker_row_count": speaker_counts[speaker_id],
                "oasis_reuse_count": 1,
                "group_id": f"{speaker_id}|{oasis_id}",
                "orig_split": window["orig_split"],
                "no_contradiction": bool(
                    int(label) == int(window["label"])
                    and str(sex) == str(window["sex"])
                    and effective_lo <= float(oasis_row["age"]) <= effective_hi
                ),
            }
            row.update({field: window[field] for field in SPEECH_FEATURES})
            row.update({field: oasis_row[field] for field in ["MMSE", "ASF", "EDUC", "SES", "nWBV", "eTIV"]})
            matched_rows.append(row)

    matched = pd.DataFrame(matched_rows)
    if matched.empty:
        matched = pd.DataFrame(columns=[*POOL_COLUMNS, *SPEECH_FEATURES, *PROVENANCE_COLUMNS])
    else:
        counts = matched["speaker_id"].value_counts()
        matched["speaker_row_count"] = matched["speaker_id"].map(counts).astype(int)
        matched["oasis_reuse_count"] = matched["oasis_subject_id"].map(matched["oasis_subject_id"].value_counts()).astype(int)
        matched["no_contradiction"] = matched["no_contradiction"].astype(bool)

    unmatched_oasis = pool[~pool["oasis_subject_id"].astype(str).isin(matched_oasis)].copy()
    unmatched_oasis["reason"] = "no eligible speech window"
    used_speakers = set(matched["speaker_id"].astype(str)) if not matched.empty else set()
    unmatched_speakers = speakers[~speakers["speaker_id"].astype(str).isin(used_speakers)].copy()
    unmatched_speakers["rows_assigned"] = 0
    if not matched.empty:
        assigned = matched["speaker_id"].value_counts()
        unmatched_speakers["rows_assigned"] = unmatched_speakers["speaker_id"].map(assigned).fillna(0).astype(int)
    return matched.reset_index(drop=True), unmatched_oasis.reset_index(drop=True), unmatched_speakers.reset_index(drop=True)


def assemble_outputs(matched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert match rows into the frozen dataset and provenance contracts."""
    if matched.empty:
        return pd.DataFrame(columns=DATASET_COLUMNS), pd.DataFrame(columns=PROVENANCE_COLUMNS)
    result = matched.copy()
    result["Subject_ID"] = result.apply(
        lambda row: f"{row['oasis_source']}_{row['oasis_subject_id']}_w{int(row['window_idx'])}", axis=1
    )
    dataset = result[DATASET_COLUMNS[:-1]].copy()
    dataset["Label"] = result["label"].astype(int)
    dataset = dataset[DATASET_COLUMNS]
    provenance = result[PROVENANCE_COLUMNS].copy()
    provenance["no_contradiction"] = provenance["no_contradiction"].astype(bool)
    if not provenance["no_contradiction"].all():
        raise AssertionError("A matched row violates the label/sex/age contradiction check")
    return dataset, provenance


def _write_match_variant(
    pool: pd.DataFrame,
    speakers_path: Path,
    speech_path: Path,
    variant: str,
    output_dir: Path,
    r_max: int,
) -> None:
    speakers = pd.read_csv(speakers_path)
    speech = pd.read_csv(speech_path)
    matched, unmatched_oasis, unmatched_speakers = match_windows(pool, speakers, speech, r_max=r_max)
    dataset, provenance = assemble_outputs(matched)
    suffix = f"_{variant}"
    dataset.to_csv(output_dir / f"multimodal_dementia_dataset{suffix}.csv", index=False)
    provenance.to_csv(output_dir / f"multimodal_real_provenance{suffix}.csv", index=False)
    dataset_1to1, provenance_1to1 = assemble_outputs(
        match_windows(pool, speakers, speech, r_max=1)[0]
    )
    dataset_1to1.to_csv(output_dir / f"multimodal_real_1to1{suffix}.csv", index=False)
    unmatched_oasis.to_csv(output_dir / f"unmatched_oasis{suffix}.csv", index=False)
    unmatched_speakers.to_csv(output_dir / f"unmatched_speakers{suffix}.csv", index=False)
    if variant == "raw":
        dataset.to_csv(output_dir / "multimodal_dementia_dataset.csv", index=False)
        provenance.to_csv(output_dir / "multimodal_real_provenance.csv", index=False)
        dataset_1to1.to_csv(output_dir / "multimodal_real_1to1.csv", index=False)
        unmatched_oasis.to_csv(output_dir / "unmatched_oasis.csv", index=False)
        unmatched_speakers.to_csv(output_dir / "unmatched_speakers.csv", index=False)
    reuse = provenance["speaker_row_count"].value_counts().sort_index().to_dict() if not provenance.empty else {}
    widened = int(provenance["window_widened"].sum()) if not provenance.empty else 0
    print(
        f"{variant}: rows={len(dataset)} speakers={provenance['speaker_id'].nunique() if not provenance.empty else 0} "
        f"oasis_subjects={provenance['oasis_subject_id'].nunique() if not provenance.empty else 0} "
        f"dropped_oasis={len(unmatched_oasis)} widened={widened} speaker_row_count_hist={reuse}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oasis1", type=Path, default=DATA_DIR / "oasis1_clinical data.xlsx")
    parser.add_argument("--oasis2", type=Path, default=DATA_DIR / "oasis2_mri_clinical.xlsx")
    parser.add_argument("--speakers-meta", type=Path, default=DATA_DIR / "speakers_meta.csv")
    parser.add_argument("--speech-raw", type=Path, default=DATA_DIR / "speech_windows_raw.csv")
    parser.add_argument("--speech-clean", type=Path, default=DATA_DIR / "speech_windows_clean.csv")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--pool-only", action="store_true", help="Build only oasis_pool.csv and the bake-off")
    parser.add_argument("--match-only", action="store_true", help="Use an existing oasis_pool.csv and only run matching")
    parser.add_argument("--speech-variant", choices=["raw", "clean", "both"], default="both")
    parser.add_argument("--r-max", type=int, default=18)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pool_path = args.output_dir / "oasis_pool.csv"
    if not args.match_only:
        if not args.oasis1.exists() or not args.oasis2.exists():
            raise FileNotFoundError("Both OASIS workbook paths are required to build the pool")
        pool, selected, _ = build_oasis_pool(args.oasis1, args.oasis2, args.output_dir)
        print(f"oasis_pool: rows={len(pool)} selected_imputation={selected}")
    else:
        pool = pd.read_csv(pool_path)

    if args.pool_only:
        return 0
    if not args.speakers_meta.exists():
        print("Matcher pending: data/speakers_meta.csv is not available yet.")
        return 0
    paths = {"raw": args.speech_raw, "clean": args.speech_clean}
    variants = [args.speech_variant] if args.speech_variant != "both" else ["raw", "clean"]
    for variant in variants:
        if not paths[variant].exists():
            raise FileNotFoundError(f"Missing speech input: {paths[variant]}")
        _write_match_variant(pool, args.speakers_meta, paths[variant], variant, args.output_dir, args.r_max)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
