"""Real parser for TESS (Toronto Emotional Speech Set) filenames - the
second primary dataset for the emotion2vec+-based production model
(docs/emotion.md). Does not download anything itself - see
ml/datasets/README.md.

TESS filenames encode metadata directly:

    {speaker}_{word}_{emotion}.wav

e.g. `OAF_back_angry.wav` = Older Actress Female, target word "back",
angry. Only two speakers exist in the entire corpus - OAF (older actress)
and YAF (younger actress) - a real, hard limit of this dataset, not a
sampling choice made here. **Any speaker-independent split of TESS alone
can only ever be a 1-speaker-train / 1-speaker-test split** - there is no
way to have a validation speaker distinct from both. This is documented
explicitly (docs/emotion.md) rather than silently glossed over; TESS's
contribution to VoxMind's training set is real and useful for word/lexical
diversity (200 distinct target words x 2 speakers x 7 emotions = 2800
clips) but its OWN speaker-independent generalization claim is
necessarily weak.

"ps" (pleasant surprise) is TESS's 7th class and has no home in this
project's 6-class target vocabulary (angry, disgust, fear, happy, neutral,
sad) - excluded here, never remapped to another class.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ml.datasets.schema import EmotionSample

DATASET_SOURCE = "tess"

VALID_EMOTIONS = {"angry", "disgust", "fear", "happy", "neutral", "sad"}  # "ps" deliberately excluded
VALID_SPEAKERS = {"OAF", "YAF"}


@dataclass
class TessFilenameFields:
    speaker: str
    word: str
    emotion: str


def parse_tess_filename(filename: str) -> TessFilenameFields:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) != 3:
        raise ValueError(f"Not a valid TESS filename (expected 3 '_'-separated fields): {filename}")
    speaker, word, emotion = parts
    if speaker not in VALID_SPEAKERS:
        raise ValueError(f"Unknown TESS speaker {speaker!r} in filename: {filename}")
    if emotion not in VALID_EMOTIONS and emotion != "ps":
        raise ValueError(f"Unknown TESS emotion {emotion!r} in filename: {filename}")
    return TessFilenameFields(speaker=speaker, word=word, emotion=emotion)


def build_manifest_from_directory(root_dir: str | Path) -> list[EmotionSample]:
    """Scans `root_dir` for TESS-named `.wav` files. "ps" (pleasant
    surprise) clips are skipped (not an error, not remapped) - see module
    docstring."""
    root_dir = Path(root_dir)
    samples: list[EmotionSample] = []
    skipped = 0

    for wav_path in sorted(root_dir.rglob("*.wav")):
        try:
            fields = parse_tess_filename(wav_path.name)
        except ValueError:
            skipped += 1
            continue
        if fields.emotion == "ps":
            continue

        samples.append(
            EmotionSample(
                sample_id=wav_path.stem,
                audio_path=str(wav_path.resolve()),
                label=fields.emotion,
                dataset_source=DATASET_SOURCE,
                speaker_id=fields.speaker,
            )
        )

    if not samples:
        raise FileNotFoundError(
            f"No TESS-named .wav files found under {root_dir}. See ml/datasets/README.md."
        )
    return samples
