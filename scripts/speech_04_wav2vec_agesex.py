"""Phase 1.2 step 5 — wav2vec2 age/sex estimate per speaker.

Model: audeering/wav2vec2-large-robust-24-ft-age-gender  (custom head; gender
classes = [female, male, child]). Run per clip on the cleaned dominant-speaker
wav (fall back to raw), then aggregate to speaker.

Output:
  data/speech_model_agesex.csv   speaker_id, age_model, sex_model, n_clips

Run:  .venv/Scripts/python.exe scripts/speech_04_wav2vec_agesex.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

import librosa

from _speech_common import CLEAN_DIR, DATA, REPO, TARGET_SR, rel

MODEL_ID = "audeering/wav2vec2-large-robust-24-ft-age-gender"
CHUNK_S = 15.0
KNOWN_FEMALE = {  # spec §1 — sanity target
    "aileenhernandez", "annettemichelson", "charmiancarr", "estellegetty", "evelynkeyes",
    "irismurdoch", "jeannelittle", "maureenforrester", "patpariseau", "paulinephillips",
    "stellastevens", "teresagorman", "unitablackwell", "vivnicholson",
    "angelalansbury", "barbrastreisand", "dionnewarwick", "judidench", "linerenaud",
    "mireilledarc", "raquelwelch", "yokoono",
}


class ModelHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(self, input_values):
        hidden = self.wav2vec2(input_values)[0]
        pooled = torch.mean(hidden, dim=1)
        return self.age(pooled), torch.softmax(self.gender(pooled), dim=1)


def clean_or_raw(clip_path: str, speaker_id: str) -> str:
    from pathlib import Path

    cand = CLEAN_DIR / speaker_id / (Path(clip_path).stem + ".wav")
    return str(cand if cand.exists() else REPO / clip_path)


@torch.no_grad()
def predict_clip(path: str, processor, model, device) -> tuple[float, np.ndarray]:
    y, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    step = int(CHUNK_S * TARGET_SR)
    chunks = [y[i : i + step] for i in range(0, max(len(y), 1), step)] or [y]
    ages, genders = [], []
    for ch in chunks:
        if len(ch) < TARGET_SR // 2:
            continue
        inp = processor(ch, sampling_rate=TARGET_SR, return_tensors="pt").input_values.to(device)
        age, gender = model(inp)
        ages.append(float(age.squeeze().cpu()))
        genders.append(gender.squeeze().cpu().numpy())
    if not ages:
        return float("nan"), np.array([np.nan, np.nan, np.nan])
    return float(np.mean(ages)) * 100.0, np.mean(genders, axis=0)


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = AgeGenderModel.from_pretrained(MODEL_ID).to(device).eval()

    clips = pd.read_csv(DATA / "speech_clips.csv")
    per_clip: list[dict] = []
    for i, c in clips.iterrows():
        age, gender = predict_clip(clean_or_raw(c["clip_path"], c["speaker_id"]), processor, model, device)
        per_clip.append(
            {
                "speaker_id": c["speaker_id"],
                "age": age,
                # gender probs are [female, male, child]; decide F/M on the first two
                "sex": "F" if (np.nan_to_num(gender[0]) >= np.nan_to_num(gender[1])) else "M",
            }
        )
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(clips)} clips")

    pc = pd.DataFrame(per_clip)
    agg = (
        pc.groupby("speaker_id")
        .agg(
            age_model=("age", "median"),
            sex_model=("sex", lambda s: s.mode().iat[0]),
            n_clips=("age", "size"),
        )
        .reset_index()
    )
    agg["age_model"] = agg["age_model"].round(1)
    agg.to_csv(DATA / "speech_model_agesex.csv", index=False)

    print(f"\nwrote {rel(DATA / 'speech_model_agesex.csv')}  ({len(agg)} speakers)")
    print(f"  age_model range:  {agg.age_model.min()} .. {agg.age_model.max()} "
          f"(median {agg.age_model.median()})")
    print(f"  sex_model F / M:  {(agg.sex_model == 'F').sum()} / {(agg.sex_model == 'M').sum()}")

    flagged = agg[agg.speaker_id.isin(KNOWN_FEMALE) & (agg.sex_model != "F")]
    print(f"  known-female speakers predicted M: {len(flagged)} / {len(KNOWN_FEMALE)}")
    for sid in flagged.speaker_id:
        print(f"    {sid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
