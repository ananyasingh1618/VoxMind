"""Emotion Model v2's unified six-class label vocabulary and the explicit,
documented per-dataset label mappings that project RAVDESS's eight classes
and CREMA-D's six classes onto it. See docs/emotion.md's "Emotion Model v2"
section for the full rationale.

Every mapping here is a plain, auditable dict - nothing is inferred or
reinterpreted at training time. A raw label that maps to `None` is
*excluded* from the v2 training task entirely (RAVDESS's "calm" and
"surprised"), never silently folded into another class.
"""
from __future__ import annotations

# Sorted alphabetically - the same convention train_emotion_model.py already
# uses for v1's label_names (`sorted(set(...))`), so v2's label order is
# derived the same deterministic way, not hand-ordered.
V2_LABELS: list[str] = ["angry", "disgust", "fear", "happy", "neutral", "sad"]

# RAVDESS's raw parsed label (ml/datasets/ravdess.py's EMOTION_CODES values)
# -> v2's six-class label, or None to exclude that clip from v2 training.
# "fearful" -> "fear" is a naming normalization only, not a reinterpretation
# of the emotion itself. "calm" and "surprised" are excluded per the v2
# spec (docs/emotion.md) - they have no CREMA-D counterpart and are not
# reinterpreted as any of the six retained classes.
RAVDESS_TO_V2: dict[str, str | None] = {
    "neutral": "neutral",
    "calm": None,
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fearful": "fear",
    "disgust": "disgust",
    "surprised": None,
}

# CREMA-D's raw parsed label (ml/datasets/crema_d.py's EMOTION_CODES values)
# already *is* the v2 six-class vocabulary - included here (as an identity
# mapping) so both datasets go through the exact same
# "look up in a mapping, drop if None" code path in
# prepare_emotion_dataset_v2.py, rather than one dataset being special-cased.
CREMA_D_TO_V2: dict[str, str | None] = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
}

DATASET_MAPPINGS: dict[str, dict[str, str | None]] = {
    "ravdess": RAVDESS_TO_V2,
    "crema_d": CREMA_D_TO_V2,
}
