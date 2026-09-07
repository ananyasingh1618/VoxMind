"""Emotion Model v2: a Wav2Vec2-backbone-fine-tuning architecture, additive
to (never replacing) the frozen v1 (`voxmind/services/emotion/model.py`,
`classifier_provider.py`). See docs/emotion.md's "Emotion Model v2" section
for the full rationale, dataset, training, and evaluation story.

Nothing in this subpackage is imported by v1 code, and v1's
`ravdess-v1` checkpoint/ModelVersion/behavior is never touched by anything
here.
"""
from __future__ import annotations
