"""Real parser for the RAVDESS (Ryerson Audio-Visual Database of Emotional
Speech and Song) filename convention - the documented target dataset for
VoxMind's emotion classifier. See ml/datasets/README.md for licensing and
download instructions; this module does not download anything itself.

RAVDESS filenames encode metadata directly, with no separate label file:

    {modality}-{vocal_channel}-{emotion}-{intensity}-{statement}-{repetition}-{actor}.wav

e.g. `03-01-06-01-02-01-12.wav` = audio-only, speech, fearful, normal
intensity, statement 2, 1st repetition, actor 12.

Actor IDs (01-24, odd=male/even=female) are used as `speaker_id` for
speaker-independent splitting (splitting.py) - RAVDESS's own actor numbering
is exactly the speaker-identity metadata that requirement needs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ml.datasets.schema import EmotionSample

DATASET_SOURCE = "ravdess"

# Official RAVDESS emotion code -> label mapping (from the dataset's own
# documentation, not invented). Only the 8 speech emotions - "song" clips
# use vocal_channel=02 and are excluded (see build_manifest_from_directory).
EMOTION_CODES = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

SPEECH_VOCAL_CHANNEL = "01"


@dataclass
class RavdessFilenameFields:
    modality: str
    vocal_channel: str
    emotion_code: str
    intensity: str
    statement: str
    repetition: str
    actor: str

    @property
    def emotion_label(self) -> str:
        return EMOTION_CODES[self.emotion_code]


def parse_ravdess_filename(filename: str) -> RavdessFilenameFields:
    stem = Path(filename).stem
    parts = stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"Not a valid RAVDESS filename (expected 7 '-'-separated fields): {filename}")
    modality, vocal_channel, emotion_code, intensity, statement, repetition, actor = parts
    if emotion_code not in EMOTION_CODES:
        raise ValueError(f"Unknown RAVDESS emotion code {emotion_code!r} in filename: {filename}")
    return RavdessFilenameFields(
        modality=modality,
        vocal_channel=vocal_channel,
        emotion_code=emotion_code,
        intensity=intensity,
        statement=statement,
        repetition=repetition,
        actor=actor,
    )


def build_manifest_from_directory(root_dir: str | Path) -> list[EmotionSample]:
    """Scans `root_dir` recursively for `.wav` files following the RAVDESS
    naming convention. Non-matching files are skipped (not an error) - a
    RAVDESS download typically also contains a "song" subset (vocal_channel
    02) and/or video files, which are intentionally excluded here since
    VoxMind analyzes speech, not song."""
    root_dir = Path(root_dir)
    samples: list[EmotionSample] = []
    skipped = 0

    for wav_path in sorted(root_dir.rglob("*.wav")):
        try:
            fields = parse_ravdess_filename(wav_path.name)
        except ValueError:
            skipped += 1
            continue
        if fields.vocal_channel != SPEECH_VOCAL_CHANNEL:
            continue

        samples.append(
            EmotionSample(
                sample_id=wav_path.stem,
                audio_path=str(wav_path.resolve()),
                label=fields.emotion_label,
                dataset_source=DATASET_SOURCE,
                speaker_id=fields.actor,
            )
        )

    if not samples:
        raise FileNotFoundError(
            f"No RAVDESS-named .wav files found under {root_dir}. "
            "See ml/datasets/README.md for the expected download/directory structure."
        )
    return samples
