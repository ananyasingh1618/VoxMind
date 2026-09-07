"""Regression coverage for a real, serious bug found via live testing
(test_real_worker_runs_tts_synthesis, Phase 9.1): `BaseModel.model_dump(
mode="json")` raises `UnicodeDecodeError` on a real `bytes` field
containing genuine binary data (raw WAV audio, not valid UTF-8) - and that
call used to sit outside `_execute_pipeline_stage_async`'s try/except
entirely, so the failure escaped uncaught and left a `pipeline_runs` row
stuck at "running" forever. These are deterministic fixtures, not real
audio - the real end-to-end proof is the live test named above.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

import pytest
from pydantic import BaseModel

from voxmind.workers.json_safe import dump_json_safe, load_json_safe


class _Color(str, Enum):
    RED = "red"


class _Nested(BaseModel):
    blob: bytes


class _Sample(BaseModel):
    text: str
    raw_audio: bytes
    nested: _Nested
    created_at: datetime
    id: uuid.UUID
    color: _Color


def test_real_binary_data_that_is_not_valid_utf8_round_trips():
    """The exact failure mode that broke TTS: raw bytes containing invalid
    UTF-8 sequences (a real WAV header does this) must survive a full
    dump/load round trip unchanged."""
    not_valid_utf8 = b"RIFF\x24\x8c\x00\x00WAVE\xff\xfe\xff\xff\x00\x01\x80>"
    sample = _Sample(
        text="hello",
        raw_audio=not_valid_utf8,
        nested=_Nested(blob=b"\xff\xfe more raw bytes \x00\x01"),
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        id=uuid.uuid4(),
        color=_Color.RED,
    )

    dumped = dump_json_safe(sample)

    # The dumped form is genuinely JSON-safe - no raw bytes anywhere.
    import json

    json.dumps(dumped)  # must not raise

    restored = load_json_safe(_Sample, dumped)
    assert restored.raw_audio == not_valid_utf8
    assert restored.nested.blob == sample.nested.blob
    assert restored.text == "hello"
    assert restored.id == sample.id
    assert restored.color == _Color.RED
    assert restored.created_at == sample.created_at


def test_the_naive_json_mode_dump_actually_fails_on_this_input():
    """Documents *why* dump_json_safe exists - proves the bug is real, not
    hypothetical, by showing the naive approach genuinely raises."""
    sample = _Sample(
        text="hello",
        raw_audio=b"\xff\xfe\x00\x01 not valid utf-8 \x80",
        nested=_Nested(blob=b"\x00"),
        created_at=datetime.now(timezone.utc),
        id=uuid.uuid4(),
        color=_Color.RED,
    )
    with pytest.raises(UnicodeDecodeError):
        sample.model_dump(mode="json")


def test_dump_json_safe_handles_plain_json_native_values_normally():
    """Every field type these stages actually use elsewhere (str, list,
    float, None) must still work exactly as `mode="json"` would - the fix
    must not regress the common case."""

    class Plain(BaseModel):
        chunks: list[str]
        score: float | None
        count: int

    original = Plain(chunks=["a", "b"], score=None, count=3)
    dumped = dump_json_safe(original)
    assert dumped == {"chunks": ["a", "b"], "score": None, "count": 3}
    assert load_json_safe(Plain, dumped) == original
