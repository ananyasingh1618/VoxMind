"""Emotion2Vec+ production candidate: a frozen `emotion2vec/
emotion2vec_plus_base` encoder (via FunASR - see encoder.py) feeding a
small trained classification head (model.py) for VoxMind's six-class
target vocabulary (angry, disgust, fear, happy, neutral, sad). See
docs/emotion.md for the full dataset/training/evaluation account.

Additive to (never replacing) v1 (`../model.py`) and v2 (`../v2/`) - none
of those are imported here, and nothing here is imported by them.
"""
from __future__ import annotations
