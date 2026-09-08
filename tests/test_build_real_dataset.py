from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_real_dataset import (
    CLINICAL_COLUMNS,
    DATASET_COLUMNS,
    IMPUTATION_PREDICTORS,
    POOL_COLUMNS,
    S3_IMPUTER_COLUMNS,
    SPEECH_FEATURES,
    _imputed_flags,
    _mask_known_values,
    apply_imputation,
    assemble_outputs,
    build_oasis_pool,
    choose_imputation_scheme,
    load_oasis_tables,
    match_windows,
    score_imputation_scheme,
)


def _pool() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["oasis1", "OAS1_0001", "OAS1_0001_MR1", 0, "F", 50, 28, 1.1, 12, 2, 0.8, 1400, 0, ""],
            ["oasis2", "OAS2_0001", "OAS2_0001_MR1", 1, "F", 51, 20, 1.2, 10, 2, 0.7, 1300, 1, ""],
            ["oasis2", "OAS2_0002", "OAS2_0002_MR1", 0, "M", 70, 27, 1.0, 14, 1, 0.8, 1450, 0, ""],
        ],
        columns=POOL_COLUMNS,
    )


def _speakers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["speaker_f0", 0, "F", 50, 48, 52, "lookup", "test"],
            ["speaker_f1", 1, "F", 50, 48, 52, "lookup", "test"],
            ["speaker_m0", 0, "M", 70, 68, 72, "lookup", "test"],
        ],
        columns=["speaker_id", "label", "sex", "age_final", "age_lo", "age_hi", "age_source", "orig_split"],
    )


def _windows() -> pd.DataFrame:
    rows = []
    for speaker, count in [("speaker_f0", 2), ("speaker_f1", 1), ("speaker_m0", 1)]:
        for window_idx in range(count):
            row = {"speaker_id": speaker, "clip_path": f"{speaker}.wav", "window_idx": window_idx}
            row.update({feature: float(window_idx + 1) for feature in SPEECH_FEATURES})
            rows.append(row)
    return pd.DataFrame(rows)


def _oasis1_sheet() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            ["OAS1_0001_MR1", "female", 50, 12, 28, 2, 0.8, 1400, 1.1, 0.0],
            ["OAS1_0002_MR1", "M", 60, 14, 20, None, 0.7, 1300, 1.2, 1.0],
            ["OAS1_0003_MR2", "F", 61, 15, 21, 1, 0.7, 1301, 1.2, 0.5],
            ["OAS1_0004_MR1", "M", 62, 13, 22, 1, 0.7, 1302, 1.2, None],
        ],
        columns=["ID", "M/F", "Age", "Educ", "MMSE", "SES", "nWBV", "eTIV", "ASF", "CDR"],
    )
    return frame


def _oasis2_sheet() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            ["OAS2_0001", "OAS2_0001_MR1", "Nondemented", 1, "F", 70, 27, 0, 0.8, 1400, 1.0],
            ["OAS2_0001", "OAS2_0001_MR2", "Nondemented", 2, "F", 72, 26, 0, 0.79, 1401, 1.01],
            ["OAS2_0131", "OAS2_0131_MR1", "Converted", 1, "M", 65, 30, 0.5, 0.75, 1340, 1.30],
            ["OAS2_0131", "OAS2_0131_MR2", "Converted", 2, "M", 67, 25, 0, 0.76, 1331, 1.31],
        ],
        columns=["Subject ID", "MRI ID", "Group", "Visit", "M/F", "Age", "MMSE", "CDR", "nWBV", "eTIV", "ASF"],
    )
    frame["Educ"] = [12, 12, 10, 10]
    frame["SES"] = [1, 1, 2, 2]
    return frame


def _stub_raw_frame() -> pd.DataFrame:
    frame = pd.concat([_pool()] * 6, ignore_index=True)
    frame["Group"] = frame["label"].map({0: "Nondemented", 1: "Demented"})
    frame.loc[0, "SES"] = pd.NA
    frame.loc[1, "MMSE"] = pd.NA
    frame.loc[2, "EDUC"] = pd.NA
    return frame


def test_phase_1_3_harmonises_both_oasis_workbooks(monkeypatch):
    def fake_read_excel(path):
        return _oasis2_sheet() if "oasis2" in str(path) else _oasis1_sheet()

    monkeypatch.setattr("scripts.build_real_dataset.pd.read_excel", fake_read_excel)
    result = load_oasis_tables(Path("oasis1_clinical data.xlsx"), Path("oasis2_mri_clinical.xlsx"))

    oasis1 = result[result["oasis_source"] == "oasis1"]
    assert oasis1["mri_id"].tolist() == ["OAS1_0001_MR1", "OAS1_0002_MR1"]
    assert oasis1["oasis_subject_id"].tolist() == ["OAS1_0001", "OAS1_0002"]
    assert oasis1["label"].tolist() == [0, 1]
    assert oasis1["sex"].tolist() == ["F", "M"]
    assert oasis1["EDUC"].tolist() == [12.0, 14.0]

    oasis2 = result[result["oasis_source"] == "oasis2"].set_index("oasis_subject_id")
    assert len(oasis2) == 2
    assert oasis2.loc["OAS2_0001", "mri_id"] == "OAS2_0001_MR2"
    assert oasis2.loc["OAS2_0131", "mri_id"] == "OAS2_0131_MR1"
    assert oasis2.loc["OAS2_0131", "CDR"] == 0.5
    assert oasis2.loc["OAS2_0131", "label"] == 1
    assert oasis2.loc["OAS2_0131", "ASF"] == 1.30


def test_all_imputation_schemes_execute_without_diagnosis_predictors():
    frame = _stub_raw_frame()
    assert not set(IMPUTATION_PREDICTORS).intersection({"label", "CDR", "Group"})
    assert not set(S3_IMPUTER_COLUMNS).intersection({"label", "CDR", "Group"})
    assert set(CLINICAL_COLUMNS) == {"MMSE", "EDUC", "SES"}
    for scheme in ["S0", "S1", "S2", "S3", "S4"]:
        result = apply_imputation(frame, scheme)
        assert "label" in result.columns
        assert "CDR" in result.columns
        assert "Group" in result.columns


def test_masking_uses_exact_seed_42_deterministically():
    frame = _stub_raw_frame()
    first_frame, first_mask = _mask_known_values(frame, "SES", seed=42)
    second_frame, second_mask = _mask_known_values(frame, "SES", seed=42)
    assert first_mask.equals(second_mask)
    pd.testing.assert_frame_equal(first_frame, second_frame)


def test_bakeoff_scores_all_schemes_and_selection_uses_measured_scores():
    frame = _stub_raw_frame()
    scores = [score_imputation_scheme(frame, scheme) for scheme in ["S0", "S1", "S2", "S3", "S4"]]
    assert [row["scheme"] for row in scores] == ["S0", "S1", "S2", "S3", "S4"]
    measured = [row for row in scores if row["composite_error"] != float("inf")]
    assert measured
    assert choose_imputation_scheme(
        [{"scheme": "S2", "composite_error": 0.1}, {"scheme": "S3", "composite_error": 0.9}],
        frame,
    ) == "S2"


def test_imputed_flags_identify_fields_individually():
    original = _stub_raw_frame()
    imputed = apply_imputation(original, "S3")
    flags = _imputed_flags(original, imputed)
    assert flags.loc[0] == "SES"
    assert flags.loc[1] == "MMSE"
    assert flags.loc[2] == "EDUC"


def test_pool_builder_writes_exact_contract_from_inputs(monkeypatch, tmp_path):
    def fake_read_excel(path):
        return _oasis2_sheet() if "oasis2" in str(path) else _oasis1_sheet()

    monkeypatch.setattr("scripts.build_real_dataset.pd.read_excel", fake_read_excel)
    pool, selected, scores = build_oasis_pool(
        Path("oasis1_clinical data.xlsx"), Path("oasis2_mri_clinical.xlsx"), tmp_path
    )
    assert list(pool.columns) == POOL_COLUMNS
    assert pool["oasis_subject_id"].is_unique
    assert set(pool["sex"]) == {"F", "M"}
    assert set(pool["label"]) == {0, 1}
    assert len(scores) == 5
    assert selected == min(
        (row for row in scores if row["composite_error"] != float("inf")),
        key=lambda row: (row["composite_error"], row["scheme"]),
    )["scheme"]
    assert (tmp_path / "oasis_pool.csv").exists()
    assert (tmp_path / "oasis_imputation_bakeoff.md").exists()


def test_match_is_deterministic_and_obeys_one_to_one_oasis_assignment():
    first = match_windows(_pool(), _speakers(), _windows(), r_max=18)
    second = match_windows(_pool(), _speakers(), _windows(), r_max=18)
    first_matched = first[0].sort_values("oasis_subject_id").reset_index(drop=True)
    second_matched = second[0].sort_values("oasis_subject_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(first_matched, second_matched)
    assert first_matched["oasis_subject_id"].is_unique
    assert first_matched["no_contradiction"].all()


def test_match_widens_once_and_records_the_effective_bounds():
    pool = _pool().iloc[[0]].copy()
    pool["age"] = 60
    speakers = _speakers().iloc[[0]].copy()
    windows = _windows().query("speaker_id == 'speaker_f0'").iloc[[0]].copy()
    matched, unmatched_oasis, _ = match_windows(pool, speakers, windows, r_max=18)
    assert len(matched) == 1
    assert len(unmatched_oasis) == 0
    assert bool(matched.iloc[0]["window_widened"])
    assert matched.iloc[0]["age_lo"] == 35
    assert matched.iloc[0]["age_hi"] == 65
    assert bool(matched.iloc[0]["no_contradiction"])


def test_assembly_has_frozen_contract_columns():
    matched = match_windows(_pool(), _speakers(), _windows())[0]
    dataset, provenance = assemble_outputs(matched)
    assert list(dataset.columns) == DATASET_COLUMNS
    assert list(provenance.columns) == [
        "Subject_ID", "oasis_source", "oasis_subject_id", "oasis_age", "label", "sex",
        "speaker_id", "clip_path", "window_idx", "speaker_age_final", "speaker_age_source",
        "age_lo", "age_hi", "age_gap", "window_widened", "speaker_row_count",
        "oasis_reuse_count", "group_id", "orig_split", "no_contradiction",
    ]
    assert dataset["Subject_ID"].str.contains("_w").all()


def test_imputation_does_not_modify_label_or_cdr():
    frame = _pool()
    frame.loc[0, "SES"] = pd.NA
    frame.loc[1, "MMSE"] = pd.NA
    result = apply_imputation(frame, "S2")
    assert result["label"].tolist() == frame["label"].tolist()
    assert result["CDR"].tolist() == frame["CDR"].tolist()
    assert result["SES"].notna().all()
    assert result["MMSE"].notna().all()
