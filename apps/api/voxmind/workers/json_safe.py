"""Phase 9.1: a real bug, found via live testing
(test_real_worker_runs_tts_synthesis), needed this module to exist.

`BaseModel.model_dump(mode="json")` assumes every `bytes` field is valid
UTF-8 text - it tries to decode it as a string. That's true for none of
this codebase's actual text fields, but `TtsSynthesisOutput.result.audio_bytes`
(the real synthesized WAV audio - see services/tts/interfaces.py) is raw
binary, and `model_dump(mode="json")` raised a real, uncaught
`UnicodeDecodeError` on it in the worker. The exception escaped
`_execute_pipeline_stage_async` entirely (that call sat outside its
try/except - also fixed, see tasks.py), leaving the `pipeline_runs` row
stuck at "running" forever.

The fix is deliberately local to the Celery persistence boundary, not to
`TtsResult`/`TtsSynthesisStage` themselves - those are real, working Phase 5
code, used as-is (in memory, never JSON-serialized) by `InProcessTaskRunner`,
and changing their field types (e.g. to Pydantic's `Base64Bytes`, which
turned out to mean "the wire value is already base64-encoded text", the
opposite of what's needed here) would have changed what the field means for
that already-working, unrelated path too.

`model_dump(mode="python")` never attempts to decode `bytes` fields at all
(unlike `mode="json"`), but it also doesn't stringify `datetime`/`UUID`/
`Enum` values the way `mode="json"` does - so this round-trips through
`json.dumps(..., default=_json_default)` once, which handles all of those
generically, then `json.loads()` back into a plain, JSONB-safe dict.
"""
from __future__ import annotations

import base64
import json
import uuid
from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel

_BYTES_MARKER_KEY = "__voxmind_b64_bytes__"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, bytes):
        return {_BYTES_MARKER_KEY: base64.b64encode(obj).decode("ascii")}
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def dump_json_safe(model: BaseModel) -> dict:
    """The write side: a `PipelineStage` output -> a plain dict safe to
    store in a Postgres JSONB column, with any `bytes` field (raw audio,
    currently only `TtsResult.audio_bytes`) base64-wrapped rather than
    naively (and incorrectly) treated as UTF-8 text."""
    raw = model.model_dump(mode="python")
    return json.loads(json.dumps(raw, default=_json_default))


def _restore_bytes(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value.keys()) == {_BYTES_MARKER_KEY} and isinstance(value[_BYTES_MARKER_KEY], str):
            return base64.b64decode(value[_BYTES_MARKER_KEY])
        return {k: _restore_bytes(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore_bytes(v) for v in value]
    return value


def load_json_safe(output_model: type[BaseModel], data: dict) -> BaseModel:
    """The read side: reverses `dump_json_safe`'s base64 wrapping before
    handing the dict to Pydantic for real validation into the stage's
    actual output type."""
    return output_model.model_validate(_restore_bytes(data))
