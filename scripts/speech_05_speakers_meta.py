"""Phase 1.5 — merge the speech-model age/sex with Sheetal's researched ground
truth into data/speakers_meta.csv (spec real_dataset_setup.md Phase 1.5 + Appendix A).

Inputs (all data/, gitignored):
  _speaker_research_worksheet_filled.csv   Sheetal's filled Track-C worksheet (131 rows)
  speech_model_agesex.csv                  wav2vec2 age/sex per speaker (Phase 1.2)
  speech_clips.csv                         per-clip orig_split hint (Phase 1.2)

Outputs (data/):
  speaker_birthdates.csv        Appendix A  (split out of the worksheet)
  speaker_sex_lookup.csv        Appendix A  (split out of the worksheet)
  speakers_meta.csv             Appendix A  (the Phase 1.6 matcher input)
  speakers_meta_sex_review.csv  only if any speaker's sex is unresolved / disputed

Rules (spec Phase 1.5):
  - sex: model and name-lookup must agree; blanks fall back to the model, disputes
    keep the researched answer -- either way the speaker is listed for Sheetal.
  - age_final: age_birthdate only when confidence == high, else age_model
    (the worksheet's 'med' recording years are back-computed from age_model).
  - age_lo/age_hi: age_final +/- tol,  tol = 8 birthdate-based / 12 model-only.
  - age_model_vs_birthdate_gap carried as a QC column; prints the model-vs-lookup MAE.

Run:  .venv/Scripts/python.exe scripts/speech_05_speakers_meta.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from _speech_common import DATA

TOL_BIRTHDATE = 8.0
TOL_MODEL = 12.0
# Only 'high' trusts the researched age. The worksheet's 'med' recording years
# were back-computed as birth_year + round(age_model) (123/125 exact), so they
# carry no independent age signal -- treat them as model-derived. (Deviates from
# the spec's {high, med}; see DATASHEET age-QC note.)
BIRTHDATE_CONF = {"high"}

BIRTHDATES_COLS = [
    "speaker_id", "display_name", "birth_year", "est_recording_year",
    "age_birthdate", "confidence", "note",
]
SEX_LOOKUP_COLS = ["speaker_id", "sex_name", "ambiguous"]
META_COLS = [
    "speaker_id", "label", "sex", "age_final", "age_lo", "age_hi",
    "age_model", "age_birthdate", "age_source", "age_model_vs_birthdate_gap",
    "orig_split",
]


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("", np.nan), errors="coerce")


def main() -> int:
    ws = pd.read_csv(DATA / "_speaker_research_worksheet_filled.csv", dtype=str, keep_default_na=False)
    model = pd.read_csv(DATA / "speech_model_agesex.csv", dtype=str, keep_default_na=False)
    clips = pd.read_csv(DATA / "speech_clips.csv", dtype=str, keep_default_na=False)

    ws = ws.sort_values("speaker_id").reset_index(drop=True)
    if set(ws.speaker_id) != set(model.speaker_id) or set(ws.speaker_id) != set(clips.speaker_id):
        print("ERROR: speaker_id sets differ across worksheet / model / clips", file=sys.stderr)
        return 1

    ws["birth_year_n"] = _num(ws["birth_year"])
    ws["rec_year_n"] = _num(ws["est_recording_year"])
    ws["age_birthdate"] = ws["rec_year_n"] - ws["birth_year_n"]

    # ---- split file 1: speaker_birthdates.csv -------------------------------
    birthdates = ws.assign(
        birth_year=ws["birth_year_n"].astype("Int64"),
        est_recording_year=ws["rec_year_n"].astype("Int64"),
        age_birthdate=ws["age_birthdate"].round(1),
    )[BIRTHDATES_COLS]
    birthdates.to_csv(DATA / "speaker_birthdates.csv", index=False)

    # ---- split file 2: speaker_sex_lookup.csv -----------------------------
    sex_lookup = ws.rename(columns={"sex_confirmed": "sex_name"})[SEX_LOOKUP_COLS]
    sex_lookup.to_csv(DATA / "speaker_sex_lookup.csv", index=False)

    # ---- reconcile ------------------------------------------------------
    # the worksheet carries pre-filled age_model/sex_model copies; drop them and
    # take the authoritative values straight from speech_model_agesex.csv.
    ws = ws.drop(columns=["age_model", "sex_model", "n_clips"], errors="ignore")
    df = ws.merge(model[["speaker_id", "age_model", "sex_model"]], on="speaker_id", how="left")
    df["age_model"] = _num(df["age_model"])

    split = (
        clips.groupby("speaker_id")["orig_split"]
        .agg(lambda s: next((v for v in s if v), ""))
        .rename("orig_split")
    )
    df = df.merge(split, on="speaker_id", how="left")

    # sex: researched answer wins; blank -> model fallback. Flag both cases.
    researched = df["sex_confirmed"].str.strip().str.upper()
    model_sex = df["sex_model"].str.strip().str.upper()
    df["sex"] = researched.where(researched.isin(["M", "F"]), model_sex)
    df["_sex_blank"] = ~researched.isin(["M", "F"])
    df["_sex_dispute"] = researched.isin(["M", "F"]) & (researched != model_sex)

    # age
    has_birthdate = df["age_birthdate"].notna()
    use_birthdate = has_birthdate & df["confidence"].str.strip().str.lower().isin(BIRTHDATE_CONF)
    df["age_source"] = np.where(use_birthdate, "birthdate", "model")
    df["age_final"] = np.where(use_birthdate, df["age_birthdate"], df["age_model"])
    tol = np.where(use_birthdate, TOL_BIRTHDATE, TOL_MODEL)
    df["age_lo"] = df["age_final"] - tol
    df["age_hi"] = df["age_final"] + tol
    df["age_model_vs_birthdate_gap"] = (df["age_model"] - df["age_birthdate"]).round(1)

    if df["age_final"].isna().any():
        bad = df.loc[df["age_final"].isna(), "speaker_id"].tolist()
        print(f"ERROR: no age_final for {bad}", file=sys.stderr)
        return 1

    meta = df.assign(
        label=df["label"].astype(int),
        age_final=df["age_final"].round(1),
        age_lo=df["age_lo"].round(1),
        age_hi=df["age_hi"].round(1),
        age_model=df["age_model"].round(1),
        age_birthdate=df["age_birthdate"].round(1),
    ).sort_values("speaker_id")[META_COLS]
    meta.to_csv(DATA / "speakers_meta.csv", index=False)

    # ---- sex review list ------------------------------------------------
    review = df[df["_sex_blank"] | df["_sex_dispute"]].copy()
    review_path = DATA / "speakers_meta_sex_review.csv"
    if len(review):
        review["reason"] = np.where(review["_sex_dispute"], "model_disagrees", "not_researched_used_model")
        review[["speaker_id", "display_name", "sex_model", "sex_confirmed", "sex", "ambiguous", "confidence", "reason", "note"]].to_csv(review_path, index=False)
    elif review_path.exists():
        review_path.unlink()

    # ---- summary ------------------------------------------------------
    gap = df["age_model_vs_birthdate_gap"].abs()
    print(f"speakers_meta.csv: {len(meta)} speakers  ({(meta.label == 1).sum()} label-1 / {(meta.label == 0).sum()} label-0)")
    print(f"  sex:   {(meta.sex == 'F').sum()} F / {(meta.sex == 'M').sum()} M")
    print(f"  age_source: {(df.age_source == 'birthdate').sum()} birthdate / {(df.age_source == 'model').sum()} model")
    print(f"  model-vs-lookup age MAE (n={gap.notna().sum()}): {gap.mean():.2f}  (max {gap.max():.1f})")
    print(f"  age_lo/hi span: {(meta.age_hi - meta.age_lo).min():.0f}-{(meta.age_hi - meta.age_lo).max():.0f} yr")
    by = df.groupby(["label", "sex"]).size().rename("n").reset_index()
    print("  (label, sex) speaker counts:")
    for _, r in by.iterrows():
        print(f"    label {r.label}  {r.sex}:  {r.n}")
    if len(review):
        print(f"\n  {len(review)} speaker(s) need Sheetal's sex sign-off -> {review_path.name}:")
        for _, r in review.iterrows():
            print(f"    {r.speaker_id:<20} model={r.sex_model}  researched={r.sex_confirmed or '(blank)'}  used={r.sex}  [{r.confidence}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
