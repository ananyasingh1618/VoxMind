"""Real parser for CREMA-D (Crowd-sourced Emotional Multimodal Actors
Dataset) filenames - the second dataset used by Emotion Model v2 (see
docs/emotion.md's "Emotion Model v2" section). Does not download anything
itself - see ml/datasets/README.md for where to get it.

CREMA-D filenames encode metadata directly, with no separate label file
needed for the *intended* (acted) emotion:

    {ActorID}_{SentenceCode}_{Emotion}_{Level}.wav

e.g. `1001_DFA_ANG_XX.wav` = actor 1001, sentence "DFA", intended emotion
Anger, emotion level unspecified ("XX" - CREMA-D only records a level for
four of the six emotions; Neutral and some sentence types are always "XX").

This module uses the **intended/acted** emotion encoded in the filename
(the label the actor was asked to portray) as the training label - exactly
the same choice `ravdess.py` already makes for RAVDESS (parsed from its
filename, not from any separate perceptual-rating file), so both datasets'
labels are the same *kind* of ground truth. CREMA-D also publishes
crowd-sourced human perceptual ratings of each clip (`finishedResponses.csv`,
`processedResults/summaryTable.csv` - majority-vote human perception of
displayed emotion, which sometimes disagrees with the actor's intended
emotion). Those files are NOT used here: mixing "intended" labels for
RAVDESS with "perceived" labels for CREMA-D would make the six-class label
space inconsistent across datasets in a way that isn't documented or
requested. Using perceptual-consensus labels instead is a legitimate
alternative for future work, not attempted in this phase.

Verified against the actual downloaded corpus in this environment (not
assumed): all 7,442 real filenames match the pattern below with zero
malformed names; exactly six emotion codes appear (ANG, DIS, FEA, HAP, NEU,
SAD); 91 distinct 4-digit actor IDs (1001-1091) appear - see
tests/ml/test_real_crema_d_dataset.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ml.datasets.schema import EmotionSample

DATASET_SOURCE = "crema_d"

# Official CREMA-D emotion code -> label mapping (from the dataset's own
# filename convention / README, not invented). All six of CREMA-D's classes
# map directly onto Emotion Model v2's six-class vocabulary - no exclusions
# needed here, unlike RAVDESS (which has two extra classes, "calm" and
# "surprised", excluded from the v2 task - see emotion_v2_labels.py).
EMOTION_CODES = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

VALID_LEVEL_CODES = {"LO", "MD", "HI", "XX"}


@dataclass
class CremaDFilenameFields:
    actor_id: str
    sentence_code: str
    emotion_code: str
    level_code: str

    @property
    def emotion_label(self) -> str:
        return EMOTION_CODES[self.emotion_code]


def parse_crema_d_filename(filename: str) -> CremaDFilenameFields:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) != 4:
        raise ValueError(f"Not a valid CREMA-D filename (expected 4 '_'-separated fields): {filename}")
    actor_id, sentence_code, emotion_code, level_code = parts
    if not (actor_id.isdigit() and len(actor_id) == 4):
        raise ValueError(f"Unexpected CREMA-D actor id (expected 4 digits): {filename}")
    if emotion_code not in EMOTION_CODES:
        raise ValueError(f"Unknown CREMA-D emotion code {emotion_code!r} in filename: {filename}")
    if level_code not in VALID_LEVEL_CODES:
        raise ValueError(f"Unknown CREMA-D level code {level_code!r} in filename: {filename}")
    return CremaDFilenameFields(
        actor_id=actor_id, sentence_code=sentence_code, emotion_code=emotion_code, level_code=level_code
    )


def build_manifest_from_directory(root_dir: str | Path) -> list[EmotionSample]:
    """Scans `root_dir` recursively for `.wav` files following the CREMA-D
    naming convention. Non-matching files are skipped (not an error) -
    mirrors ravdess.py's approach exactly, including raising
    FileNotFoundError if nothing at all was found (a real, actionable
    failure rather than silently producing an empty dataset)."""
    root_dir = Path(root_dir)
    samples: list[EmotionSample] = []
    skipped = 0

    for wav_path in sorted(root_dir.rglob("*.wav")):
        try:
            fields = parse_crema_d_filename(wav_path.name)
        except ValueError:
            skipped += 1
            continue

        samples.append(
            EmotionSample(
                sample_id=wav_path.stem,
                audio_path=str(wav_path.resolve()),
                label=fields.emotion_label,
                dataset_source=DATASET_SOURCE,
                speaker_id=fields.actor_id,
            )
        )

    if not samples:
        raise FileNotFoundError(
            f"No CREMA-D-named .wav files found under {root_dir}. "
            "See ml/datasets/README.md for the expected download/directory structure."
        )
    return samples
