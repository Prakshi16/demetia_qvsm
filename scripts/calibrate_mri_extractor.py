"""Fit the v2 MRI extractor's calibration: scan depth-profile -> true nWBV / eTIV.

Why v2 (2026-11-04)
-------------------
v1 was fit on 39 processed, skull-stripped OASIS-1 scans and was only reliable
on that kind of input. The live app must cope with whatever MRI it is handed,
including raw scans with the skull on, other orientations, DICOM series and
single 2-D pictures of a slice. So v2 trains on every kind of labelled scan
we have on disk:

* OASIS-1 **raw** native-space acquisitions (skull on, ``RAW/*mpr-N_anon``)
* OASIS-1 **atlas-space** with skull (``T88_111/*_t88_gfc``)
* OASIS-1 **atlas-space skull-stripped** (``T88_111/*_t88_masked_gfc``)
* OASIS-2 **raw** native-space acquisitions (``OAS2_RAW_PART1/*/RAW/mpr-N``)
* **2-D slices** cut from all of the above along all three axes (8-bit,
  randomly resized, the way an exported PNG/JPG arrives), plus OASIS's own
  GIF snapshots

Targets are OASIS's own per-session nWBV and eTIV. Features come from
``app.services.mri_features.compute_features``, the exact function the live
upload path runs, so fitting and serving can't drift apart.

Validation is ``GroupKFold(5)`` **by subject**: no person's scans are ever in
both the fitting and the scoring fold. That matters because one subject has
several near-identical acquisitions. Errors are reported per input type
because they differ a lot by type, and the pickle stores them so the app can show the
expected error next to each extraction.

Run from the repository root (feature extraction is cached in
``data/mri_calibration_features.csv``; pass ``--rebuild`` to recompute):

    python scripts/calibrate_mri_extractor.py [--rebuild] [--workers 8]
"""
from __future__ import annotations

import argparse
import io
import pickle
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.services.mri_features import (  # noqa: E402
    CALIBRATION_VERSION,
    ETIV_FALLBACK_ML,
    AveragedRegressor,
    GatedRegressor,
    compute_features,
    feature_names,
)
from app.services.mri_io import LoadedScan, load_scan  # noqa: E402

DATA = REPO_ROOT / "data"
OAS1_SCANS = DATA / "oasis1_MRI_scans"
OAS2_SCANS = DATA / "OAS2_RAW_PART1"
OAS1_XLSX = DATA / "oasis1_clinical data.xlsx"
OAS2_XLSX = DATA / "oasis2_mri_clinical.xlsx"
CACHE = DATA / "mri_calibration_features.csv"
OUTPUT = REPO_ROOT / "results" / "mri_extractor_calibration.pkl"

SLICE_OFFSETS = (-0.12, 0.0, 0.12)  # fraction of the axis length around the middle
RNG_SEED = 7


# --------------------------------------------------------------------------- #
# inventory
# --------------------------------------------------------------------------- #
def _labels() -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    oas1 = pd.read_excel(OAS1_XLSX)
    for _, row in oas1.iterrows():
        if pd.notna(row["nWBV"]) and pd.notna(row["eTIV"]):
            out[str(row["ID"])] = (float(row["nWBV"]), float(row["eTIV"]))
    oas2 = pd.read_excel(OAS2_XLSX)
    for _, row in oas2.iterrows():
        if pd.notna(row["nWBV"]) and pd.notna(row["eTIV"]):
            out[str(row["MRI ID"])] = (float(row["nWBV"]), float(row["eTIV"]))
    return out


def _inventory(labels: dict[str, tuple[float, float]]) -> list[dict]:
    jobs: list[dict] = []
    for session in sorted(OAS1_SCANS.glob("OAS1_*_MR*")):
        if session.name not in labels:
            continue
        subject = re.sub(r"_MR\d+$", "", session.name)
        for hdr in sorted((session / "RAW").glob("*mpr-*_anon.hdr")):
            jobs.append({"session": session.name, "subject": subject, "input_type": "oas1_raw", "path": str(hdr)})
        t88 = session / "PROCESSED" / "MPRAGE" / "T88_111"
        for hdr in sorted(t88.glob("*_t88_gfc.hdr")):
            jobs.append({"session": session.name, "subject": subject, "input_type": "oas1_atlas", "path": str(hdr)})
        for hdr in sorted(t88.glob("*_t88_masked_gfc.hdr")):
            jobs.append({"session": session.name, "subject": subject, "input_type": "oas1_stripped", "path": str(hdr)})
        for gif in sorted(t88.glob("*.gif")):
            jobs.append({"session": session.name, "subject": subject, "input_type": "gif", "path": str(gif)})
    for session in sorted(OAS2_SCANS.glob("OAS2_*_MR*")):
        if session.name not in labels:
            continue
        subject = re.sub(r"_MR\d+$", "", session.name)
        for hdr in sorted((session / "RAW").glob("mpr-*.nifti.hdr")):
            jobs.append({"session": session.name, "subject": subject, "input_type": "oas2_raw", "path": str(hdr)})
    return jobs


# --------------------------------------------------------------------------- #
# per-scan work (runs in worker processes)
# --------------------------------------------------------------------------- #
def _read(path: Path) -> list[tuple[str, bytes]]:
    if path.suffix == ".hdr":
        img = path.with_suffix(".img")
        return [(path.name, path.read_bytes()), (img.name, img.read_bytes())]
    return [(path.name, path.read_bytes())]


def _to_png_bytes(slice2d: np.ndarray, rng: np.random.Generator) -> bytes:
    """Mimic a real exported slice: 8-bit, upright, resized, sometimes JPEG."""
    from PIL import Image

    lo, hi = np.percentile(slice2d, [0.5, 99.5])
    scaled = np.clip((slice2d - lo) / max(hi - lo, 1e-6), 0, 1)
    img = Image.fromarray((np.rot90(scaled) * 255).astype(np.uint8))
    side = int(rng.choice([128, 176, 208, 256]))
    img = img.resize((side, int(side * img.height / img.width)))
    buf = io.BytesIO()
    if rng.random() < 0.5:
        img.save(buf, format="JPEG", quality=int(rng.integers(70, 95)))
    else:
        img.save(buf, format="PNG")
    return buf.getvalue()


def _work(job: dict) -> list[dict]:
    warnings.filterwarnings("ignore")
    import logging

    logging.getLogger("nibabel").setLevel(logging.ERROR)
    rows: list[dict] = []
    path = Path(job["path"])
    rng = np.random.default_rng(abs(hash(job["path"])) % (2**32))
    try:
        scan = load_scan(_read(path))
    except Exception as exc:  # noqa: BLE001
        return [{**job, "error": f"load: {exc}"}]

    def add(sc: LoadedScan, input_type: str, extra: str = "") -> None:
        try:
            kind, feats, _ = compute_features(sc)
        except Exception as exc:  # noqa: BLE001
            rows.append({**job, "input_type": input_type, "variant": extra, "error": f"features: {exc}"})
            return
        rows.append({**job, "input_type": input_type, "variant": extra, "kind": kind, **feats})

    if scan.is_volume:
        add(scan, job["input_type"])
        # Two-D views of this volume: each axis, a few levels around the middle.
        # (Every atlas-space volume, but only the first raw acquisition per
        # session — the repeats are near-duplicates.)
        if job["input_type"] in ("oas1_atlas", "oas1_stripped") or "mpr-1" in path.name:
            for axis in range(3):
                n = scan.data.shape[axis]
                for off in SLICE_OFFSETS:
                    idx = int(np.clip(n // 2 + off * n, 0, n - 1))
                    sl = np.take(scan.data, idx, axis=axis)
                    raw = _to_png_bytes(sl, rng)
                    try:
                        add(load_scan([("slice.png", raw)]), "slice_from_" + job["input_type"], f"ax{axis}{off:+.2f}")
                    except Exception as exc:  # noqa: BLE001
                        rows.append({**job, "input_type": "slice", "error": f"slice: {exc}"})
    else:
        add(scan, "gif")
    return rows


def build_table(workers: int) -> pd.DataFrame:
    labels = _labels()
    jobs = _inventory(labels)
    print(f"  {len(jobs)} files from {len({j['session'] for j in jobs})} labelled sessions")
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_work, job) for job in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            rows.extend(fut.result())
            if i % 50 == 0 or i == len(futures):
                print(f"  processed {i}/{len(futures)} files", flush=True)
    df = pd.DataFrame(rows)
    df["true_nwbv"] = df["session"].map(lambda s: labels[s][0])
    df["true_etiv"] = df["session"].map(lambda s: labels[s][1])
    return df


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #
def _base_models():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))),
        "hgb": lambda: HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=12,
            l2_regularization=1.0, random_state=RNG_SEED,
        ),
    }


def _candidates(names: list[str]):
    """name -> fit(X, y, stripped) -> fitted model with .predict.

    (A gated variant whose stripped branch used only a compact feature subset
    was also tried, 2026-11-04; it lost to the ensemble below on every type.)
    """
    base = _base_models()
    out = {}
    for name, make in base.items():
        out[name] = lambda X, y, s, make=make: make().fit(X, y)
        out[f"gated-{name}"] = lambda X, y, s, make=make: GatedRegressor(make).fit(X, y, s)
    out["ensemble"] = lambda X, y, s: AveragedRegressor(
        [GatedRegressor(base["ridge"]).fit(X, y, s), base["hgb"]().fit(X, y)]
    )
    return out


def _cv_predict(fit, X, y, stripped, groups) -> np.ndarray:
    from sklearn.model_selection import GroupKFold

    preds = np.empty_like(y)
    for train, test in GroupKFold(n_splits=5).split(X, y, groups):
        model = fit(X[train], y[train], stripped[train])
        preds[test] = model.predict(X[test])
    return preds


def _is_stripped(sub: pd.DataFrame) -> np.ndarray:
    return (
        sub["input_type"].str.contains("stripped")
        | ((sub["input_type"] == "gif") & sub["path"].str.contains("masked"))
    ).to_numpy()


def _score(y, p) -> dict:
    err = np.abs(p - y)
    ss_res = float(((p - y) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"mae": float(err.mean()), "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"), "n": int(len(y))}


def fit_kind(df: pd.DataFrame, kind: str, targets: list[str]) -> dict:
    sub = df[df["kind"] == kind].reset_index(drop=True)
    names = feature_names(kind)
    X = sub[names].to_numpy(dtype=float)
    groups = sub["subject"].to_numpy()
    stripped = _is_stripped(sub)
    block: dict = {"feature_names": names, "stats": {}}
    print(f"\n[{kind}] {len(sub)} samples, {sub['subject'].nunique()} subjects, {len(names)} features")

    # How well does the gate itself recognise skull-stripped input?
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    gate_pred = cross_val_predict(
        make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
        X, stripped, groups=groups, cv=GroupKFold(5),
    )
    gate_acc = float((gate_pred == stripped).mean())
    print(f"  skull-stripped gate: GroupKFold accuracy {gate_acc:.3f} ({int(stripped.sum())} stripped samples)")
    block["gate_accuracy"] = gate_acc

    for target in targets:
        y = sub[f"true_{target}"].to_numpy(dtype=float)
        best = None
        types = sub["input_type"].to_numpy()
        for name, fit in _candidates(names).items():
            preds = _cv_predict(fit, X, y, stripped, groups)
            overall = _score(y, preds)
            # Select on the MEAN of per-input-type errors, so the 771 OASIS-2 raw
            # scans can't outvote the rarer input types.
            macro = float(np.mean([np.abs(preds[types == t] - y[types == t]).mean() for t in np.unique(types)]))
            overall["macro_mae"] = macro
            print(f"  {target:5} {name:18} MAE={overall['mae']:.4f}  per-type mean MAE={macro:.4f}  R2={overall['r2']:+.3f}")
            if best is None or macro < best[1]["macro_mae"]:
                best = (name, overall, preds, fit)
        name, overall, preds, fit = best
        per_type = {
            t: _score(y[m], preds[m])
            for t in sorted(sub["input_type"].unique())
            if (m := (sub["input_type"] == t).to_numpy()).sum() >= 5
        }
        baseline = float(np.abs(y - np.median(y)).mean())
        print(f"  -> {target}: {name} (predict-the-median baseline MAE={baseline:.4f})")
        for t, s in per_type.items():
            print(f"       {t:26} MAE={s['mae']:.4f}  R2={s['r2']:+.3f}  n={s['n']}")
        block[f"{target}_model"] = fit(X, y, stripped)
        block["stats"][target] = {
            "model": name, "cv": "GroupKFold(5) by subject", **overall,
            "median_baseline_mae": baseline, "per_input_type": per_type,
        }
        sub[f"oof_{target}"] = preds
    block["_oof"] = sub
    return block


def downstream_check(volume_block: dict) -> None:
    """Does swapping true MRI values for extracted ones change SVM predictions?

    Uses out-of-fold extractions only, on subjects that are in the model's
    training set. Reported, not gated on.
    """
    provenance = DATA / "multimodal_real_provenance.csv"
    dataset = DATA / "multimodal_dementia_dataset.csv"
    svm_path = REPO_ROOT / "results" / "svm_model.pkl"
    if not (provenance.exists() and dataset.exists() and svm_path.exists()):
        return
    prov = pd.read_csv(provenance)
    data = pd.read_csv(dataset)
    if "oasis_subject_id" not in prov.columns or len(prov) != len(data):
        return
    with svm_path.open("rb") as f:
        svm = pickle.load(f)
    oof = volume_block["_oof"]
    cols = [c for c in data.columns if c not in ("Subject_ID", "Label")]
    flips, total = 0, 0
    for i, row in data.iterrows():
        sid = str(prov.loc[i, "oasis_subject_id"])
        # The exact session this training row's MRI values came from.
        cand = oof[
            oof["session"].str.startswith(sid + "_")
            & oof["input_type"].str.endswith("raw")
            & ((oof["true_nwbv"] - float(row["nWBV"])).abs() < 1e-3)
        ]
        if cand.empty:
            continue
        base = row[cols].to_numpy(dtype=float)
        ref = np.sign(svm.decision_function([base])[0])
        for _, scan_row in cand.iterrows():
            alt = row[cols].copy()
            alt["nWBV"], alt["eTIV"] = scan_row["oof_nwbv"], scan_row["oof_etiv"]
            alt["ASF"] = 1755.0 / alt["eTIV"]
            total += 1
            flips += int(np.sign(svm.decision_function([alt.to_numpy(dtype=float)])[0]) != ref)
    if total:
        print(f"\nDownstream: {flips}/{total} raw-scan extractions flip the SVM label vs the true MRI values "
              f"({100 * (1 - flips / total):.1f}% label agreement)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if CACHE.exists() and not args.rebuild:
        print(f"1. Using cached features {CACHE.relative_to(REPO_ROOT)}")
        df = pd.read_csv(CACHE)
    else:
        print("1. Extracting features from every labelled scan")
        df = build_table(args.workers)
        df.to_csv(CACHE, index=False)
    if "error" in df.columns:
        bad = df[df["error"].notna()]
        if len(bad):
            print(f"  {len(bad)} files failed; first: {bad.iloc[0]['path']} -> {bad.iloc[0]['error']}")
        df = df[df["error"].isna()]

    print("2. Fitting")
    volume = fit_kind(df, "volume", ["nwbv", "etiv"])
    slice_block = fit_kind(df, "slice", ["nwbv"])
    slice_block["etiv_fallback"] = float(df[df["kind"] == "volume"]["true_etiv"].median()) or ETIV_FALLBACK_ML
    downstream_check(volume)

    for block in (volume, slice_block):
        block.pop("_oof", None)
    payload = {"version": CALIBRATION_VERSION, "volume": volume, "slice": slice_block}
    OUTPUT.write_bytes(pickle.dumps(payload))
    print(f"\nWrote {OUTPUT.relative_to(REPO_ROOT)} ({OUTPUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
