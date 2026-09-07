"""TTS service boundary.

`TtsResult` carries raw synthesized audio bytes (not a storage key) - the
same separation-of-concerns every other real provider in this codebase
follows (Wav2Vec2Provider/HuggingFaceEmbeddingProvider return plain data,
never touch StorageBackend themselves): providers do pure computation,
`VoiceTurnService` handles storage I/O. `content_type` varies genuinely by
provider (`audio/wav` for the local model, which emits raw PCM samples that
must be WAV-encoded; `audio/mpeg` for OpenAI's TTS API, which already
returns MP3 bytes) - never assumed to be one fixed format.
"""
from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


class TtsResult(BaseModel):
    audio_bytes: bytes
    content_type: str
    provider: str
    duration_ms: int


class TextToSpeechProvider(Protocol):
    async def synthesize(self, text: str, voice_profile: str | None = None) -> TtsResult: ...
