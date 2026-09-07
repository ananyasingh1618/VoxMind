"""Six-class label vocabulary and per-dataset mappings for the
emotion2vec+-based production model (docs/emotion.md). See
ml/datasets/emotion_v2_labels.py for the precedent this deliberately
mirrors (same style of explicit, auditable per-dataset dict, same
"exclude, never silently remap" policy for classes with no home in the
target vocabulary).
"""
from __future__ import annotations

V2_LABELS: list[str] = ["angry", "disgust", "fear", "happy", "neutral", "sad"]

# TESS's own filenames already use exactly these words - identity mapping.
# "ps" (pleasant surprise) has no entry: excluded upstream in tess.py's
# parser itself (never reaches this mapping at all).
TESS_TO_V2: dict[str, str | None] = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
}

# EMO-DB's raw labels (ml/datasets/emodb.py's EMOTION_INT_TO_LABEL values) -
# "anger"/"happiness"/"sadness" -> "angry"/"happy"/"sad" is a naming
# normalization only, matching the identical "fearful" -> "fear" precedent
# in emotion_v2_labels.py. "boredom" has no entry: excluded upstream in
# emodb.py's own parser (never reaches this mapping).
EMODB_TO_V2: dict[str, str | None] = {
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happiness": "happy",
    "neutral": "neutral",
    "sadness": "sad",
}

# RAVDESS is used ONLY as an external robustness benchmark here (never
# trained on, per docs/emotion.md) - reusing the exact same mapping
# ml/datasets/emotion_v2_labels.py::RAVDESS_TO_V2 already established for
# the identical purpose (six-class projection, "calm"/"surprised" excluded,
# never remapped), rather than defining a second, potentially-divergent copy.
from ml.datasets.emotion_v2_labels import RAVDESS_TO_V2  # noqa: E402

DATASET_MAPPINGS: dict[str, dict[str, str | None]] = {
    "tess": TESS_TO_V2,
    "emodb": EMODB_TO_V2,
    "ravdess": RAVDESS_TO_V2,
}
