"""Real parser for EMO-DB (Berlin Database of Emotional Speech, Burkhardt
et al. 2005) - one of the two primary datasets for the emotion2vec+-based
production model (docs/emotion.md). Does not download anything itself -
see ml/datasets/README.md for the acquisition path.

**Why via a HuggingFace mirror, not the original site**: EMO-DB's official
host (emodb.bilderbar.info) actively refuses connections (`curl` gets
"Recv failure: Connection reset by peer" - verified directly, not
assumed) - a dead academic server, not a download-restriction. `renumics/
emodb` on HuggingFace is used instead; the underlying database and its
usage terms are unchanged, only the hosting is different (the same
practice already used for RAVDESS/CREMA-D mirrors elsewhere in this
project).

**Speaker-ID reconstruction**: the HuggingFace mirror's parquet only
carries `age`, `gender`, `emotion` (int), `audio` (raw WAV bytes) - no
speaker-id column, even though the *original* EMO-DB filenames encode a
real 2-digit speaker id (e.g. `03a01Wa.wav` = speaker 03). This module
recovers the real speaker identity via the dataset's own documented
per-speaker demographics (Burkhardt et al. 2005, Table 1) - verified
empirically in this environment (not assumed): every one of the 10
speakers has a genuinely unique (age, gender) pair in the real downloaded
data (confirmed by direct inspection - no two speakers share both age and
gender), so this mapping is a legitimate, real speaker-ID recovery, not a
guess.

**Label-to-integer mapping**: verified empirically, not trusted from any
single source - the real per-integer clip counts in the downloaded parquet
(127, 81, 46, 69, 71, 79, 62) were cross-checked against Burkhardt et al.'s
own documented per-emotion totals (anger=127, boredom=81, disgust=46,
fear=69, happiness=71, neutral=79, sadness=62) and match exactly.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import soundfile as sf

from ml.datasets.schema import EmotionSample

DATASET_SOURCE = "emodb"

# Verified empirically against real per-integer clip counts - see module docstring.
EMOTION_INT_TO_LABEL: dict[int, str] = {
    0: "anger",
    1: "boredom",  # excluded - no home in the 6-class target vocabulary
    2: "disgust",
    3: "fear",
    4: "happiness",
    5: "neutral",
    6: "sadness",
}

# Burkhardt et al. 2005, "A Database of German Emotional Speech" (Table 1) -
# the original EMO-DB speaker demographics, reproduced here as the join key
# used to recover real speaker ids from (age, gender) - see module docstring.
# gender: 0=female, 1=male (the HF mirror's own encoding, verified against
# the real downloaded data's per-(age,gender) clip counts).
SPEAKER_DEMOGRAPHICS: dict[tuple[float, int], str] = {
    (31.0, 1): "03",
    (34.0, 0): "08",
    (21.0, 0): "09",
    (32.0, 1): "10",
    (26.0, 1): "11",
    (30.0, 1): "12",
    (32.0, 0): "13",
    (35.0, 0): "14",
    (25.0, 1): "15",
    (31.0, 0): "16",
}


def build_manifest_from_parquet(parquet_path: str | Path, *, audio_out_dir: str | Path) -> list[EmotionSample]:
    """Reads the real downloaded EMO-DB parquet, writes each clip's real
    embedded WAV bytes out to `audio_out_dir` (this project's existing
    dataset-preparation and training code all expects real file paths, not
    embedded bytes - matching RAVDESS/CREMA-D/TESS's own on-disk shape),
    and returns one EmotionSample per non-excluded clip. `boredom` clips
    (81 real clips) are excluded, never remapped, matching this project's
    explicit "do not silently merge labels" requirement."""
    audio_out_dir = Path(audio_out_dir)
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet_path)
    samples: list[EmotionSample] = []
    unmatched_demographics = 0

    for i, row in df.iterrows():
        emotion_label = EMOTION_INT_TO_LABEL.get(int(row["emotion"]))
        if emotion_label is None or emotion_label == "boredom":
            continue

        key = (float(row["age"]), int(row["gender"]))
        speaker_id = SPEAKER_DEMOGRAPHICS.get(key)
        if speaker_id is None:
            unmatched_demographics += 1
            continue

        audio_bytes = row["audio"]["bytes"] if isinstance(row["audio"], dict) else row["audio"]
        waveform, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        sample_id = f"emodb-{i:04d}-spk{speaker_id}"
        out_path = audio_out_dir / f"{sample_id}.wav"
        if not out_path.exists():
            sf.write(out_path, waveform, sample_rate)

        samples.append(
            EmotionSample(
                sample_id=sample_id,
                audio_path=str(out_path.resolve()),
                label=emotion_label,
                dataset_source=DATASET_SOURCE,
                speaker_id=speaker_id,
            )
        )

    if unmatched_demographics:
        raise AssertionError(
            f"{unmatched_demographics} EMO-DB clip(s) had an (age, gender) combination not in the "
            "documented 10-speaker demographics table - real speaker-id recovery failed for them. "
            "This should not happen against the real dataset; investigate before proceeding."
        )
    if not samples:
        raise FileNotFoundError(f"No usable EMO-DB samples produced from {parquet_path}.")
    return samples
