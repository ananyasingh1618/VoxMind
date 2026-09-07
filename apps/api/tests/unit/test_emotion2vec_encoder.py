"""Tests `Emotion2VecEncoder`'s error handling for malformed/empty audio -
these paths are reached and raise *before* the real emotion2vec+ model is
ever loaded (see encoder.py: `sf.read` and the empty-audio check both
happen ahead of `_load()`), so this stays a fast unit test with no real
model download - mirrors v1/v2's identical bad-input test pattern.

The real, loaded-model integration test lives in
tests/integration/test_emotion2vec_real_model.py (`real_model`-marked,
downloads the actual checkpoint) - not duplicated here.
"""
from __future__ import annotations

import pytest

from voxmind.core.config import get_settings
from voxmind.core.exceptions import AudioProcessingError
from voxmind.services.emotion.emotion2vec.encoder import Emotion2VecEncoder


@pytest.mark.asyncio
async def test_rejects_empty_audio():
    encoder = Emotion2VecEncoder(get_settings())

    with pytest.raises(AudioProcessingError):
        await encoder.embed(b"", sample_rate=16000)


@pytest.mark.asyncio
async def test_rejects_unreadable_audio():
    encoder = Emotion2VecEncoder(get_settings())

    with pytest.raises(AudioProcessingError):
        await encoder.embed(b"this is not a real wav file at all", sample_rate=16000)


def test_model_is_lazily_constructed_not_at_init_time():
    encoder = Emotion2VecEncoder(get_settings())

    # Constructing the adapter must never trigger a real model download -
    # matching v1's HuggingFaceWav2Vec2Provider and v2's identical
    # lazy-load convention.
    assert encoder._model is None
