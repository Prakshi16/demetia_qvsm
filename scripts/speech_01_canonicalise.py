"""Phase 1.2 step 1 — canonicalise the celebrity-interview speech corpus.

Output:
  data/speech_clips.csv      speaker_id, speaker_folder, clip_path, label, orig_split
  data/_speaker_folders.txt   plain list of display names (handoff to Sheetal, Phase 1.4)

Label comes from the top folder (dementia-audio = 1, nodementia-audio = 0).
orig_split is only a hint from data/{train,valid}_dm.csv — those files' paths are
broken and labels unreliable, so we use them for nothing else (spec §1).

Run:  .venv/Scripts/python.exe scripts/speech_01_canonicalise.py
"""

from __future__ import annotations

import pandas as pd

from _speech_common import DATA, REPO, iter_speaker_folders, rel, slugify_speaker

OUT_CSV = DATA / "speech_clips.csv"
OUT_FOLDERS = DATA / "_speaker_folders.txt"


def load_split_hint() -> dict[str, str]:
    """slug(speaker) -> 'train' | 'valid', parsed from the tab-separated split files."""
    hint: dict[str, str] = {}
    for name, split in (("train_dm.csv", "train"), ("valid_dm.csv", "valid")):
        path = DATA / name
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t")
        for raw_path in df["path"].astype(str):
            # .../type3/data/dementia/Dan Ingram/daningram_15.wav  -> 'Dan Ingram'
            parts = raw_path.replace("\\", "/").split("/")
            if len(parts) >= 2:
                hint.setdefault(slugify_speaker(parts[-2]), split)
    return hint


def main() -> int:
    split_hint = load_split_hint()

    rows: list[dict] = []
    collisions: dict[str, set[str]] = {}

    for folder, label in iter_speaker_folders():
        speaker_id = slugify_speaker(folder.name)
        collisions.setdefault(speaker_id, set()).add(folder.name)
        clips = sorted(folder.glob("*.wav"))
        for clip in clips:
            rows.append(
                {
                    "speaker_id": speaker_id,
                    "speaker_folder": folder.name,
                    "clip_path": rel(clip),
                    "label": label,
                    "orig_split": split_hint.get(speaker_id, ""),
                }
            )

    df = pd.DataFrame(rows).sort_values(["label", "speaker_id", "clip_path"])
    df.to_csv(OUT_CSV, index=False)

    display_names = sorted({r["speaker_folder"] for r in rows})
    OUT_FOLDERS.write_text("\n".join(display_names) + "\n", encoding="utf-8")

    n_speakers = df["speaker_id"].nunique()
    print(f"wrote {rel(OUT_CSV)}")
    print(f"  clips:           {len(df)}")
    print(f"  distinct speakers:{n_speakers}")
    print(f"  label 1 / 0:      {(df['label'] == 1).sum()} clips "
          f"({df[df.label == 1].speaker_id.nunique()} spk) / "
          f"{(df['label'] == 0).sum()} clips "
          f"({df[df.label == 0].speaker_id.nunique()} spk)")
    print(f"  split hint hit:   {(df['orig_split'] != '').sum()} clips "
          f"(train={ (df.orig_split=='train').sum() }, valid={ (df.orig_split=='valid').sum() })")
    print(f"  wrote folder list: {rel(OUT_FOLDERS)} ({len(display_names)} names)")

    bad = {sid: names for sid, names in collisions.items() if len(names) > 1}
    if bad:
        print("\n  SLUG COLLISIONS (two folders -> one speaker_id):")
        for sid, names in bad.items():
            print(f"    {sid}: {sorted(names)}")
    else:
        print("  no slug collisions")

    if n_speakers != 131 or len(df) != 230:
        print(f"\n  NOTE: expected ~131 speakers / ~230 clips, got {n_speakers} / {len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
