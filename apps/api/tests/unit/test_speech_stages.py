"""Unit tests for the pipeline stages' own wiring logic (storage I/O,
input/output typing) - using test-double storage and providers, never the
real ffmpeg/faster-whisper/pyannote code paths (those are exercised for real
in tests/unit/test_preprocessing.py and the real-model integration tests).

A test double returning a fixed result here is standard unit-testing
practice for isolating one unit of code (the stage) from its collaborators
- it is not, and must never become, part of any production code path. The
production `AudioService` always constructs stages with the real
`FasterWhisperProvider`/`PyannoteDiarizationProvider`/`preprocess_audio`.
"""
from __future__ import annotations

import pytest

from voxmind.services.speech.interfaces import (
    DiarizationResult,
    PreprocessedAudio,
    SpeakerSegment,
    TranscriptionResult,
    TranscriptSegment,
)
from voxmind.services.speech.stages import (
    AlignmentInput,
    AudioPreprocessingStage,
    DiarizationInput,
    PreprocessInput,
    SpeakerDiarizationStage,
    TranscriptAlignmentStage,
    TranscriptionInput,
    WhisperTranscriptionStage,
)


class InMemoryStorage:
    """Minimal StorageBackend test double - no filesystem, no network."""

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}

    async def upload(self, key: str, data: bytes, *, content_type: str | None = None) -> None:
        self._data[key] = data

    async def download(self, key: str) -> bytes:
        return self._data[key]

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def exists(self, key: str) -> bool:
        return key in self._data


class StubWhisperProvider:
    """Test double satisfying SpeechToTextProvider - NOT used anywhere in
    the production AudioService, only to verify WhisperTranscriptionStage
    correctly downloads audio and forwards provider arguments/results."""

    def __init__(self) -> None:
        self.received_sample_rate: int | None = None
        self.received_language: str | None = None

    async def transcribe(self, audio_bytes: bytes, *, sample_rate: int, language=None):
        self.received_sample_rate = sample_rate
        self.received_language = language
        return TranscriptionResult(
            text="stub transcript",
            segments=[TranscriptSegment(start_ms=0, end_ms=1000, text="stub transcript", confidence=0.5)],
            language="en",
            language_probability=0.99,
            model_version="stub-whisper-test-double",
        )


class StubDiarizationProvider:
    async def diarize(self, audio_bytes: bytes, *, sample_rate: int):
        return DiarizationResult(
            speaker_segments=[SpeakerSegment(speaker_label="speaker_0", start_ms=0, end_ms=1000)],
            num_speakers=1,
            model_version="stub-diarization-test-double",
        )


@pytest.mark.asyncio
async def test_preprocessing_stage_uploads_processed_audio_under_its_own_key(monkeypatch):
    storage = InMemoryStorage()
    await storage.upload("original.wav", b"fake-raw-bytes")

    async def fake_preprocess_audio(raw_bytes, *, settings):
        assert raw_bytes == b"fake-raw-bytes"
        return PreprocessedAudio(storage_key="", duration_seconds=2.5, sample_rate=16000, channels=1), b"wav-bytes"

    import voxmind.services.speech.stages as stages_module

    monkeypatch.setattr(stages_module, "preprocess_audio", fake_preprocess_audio)

    from voxmind.core.config import get_settings

    stage = AudioPreprocessingStage(storage, get_settings())
    output = await stage.run(
        PreprocessInput(original_storage_key="original.wav", processed_storage_key="processed.wav")
    )

    assert output.audio.storage_key == "processed.wav"
    assert output.audio.duration_seconds == 2.5
    assert await storage.download("processed.wav") == b"wav-bytes"


@pytest.mark.asyncio
async def test_whisper_stage_downloads_audio_and_forwards_to_provider():
    storage = InMemoryStorage()
    await storage.upload("processed.wav", b"canonical-wav-bytes")
    provider = StubWhisperProvider()

    stage = WhisperTranscriptionStage(storage, provider)
    output = await stage.run(
        TranscriptionInput(processed_storage_key="processed.wav", sample_rate=16000, language="en")
    )

    assert provider.received_sample_rate == 16000
    assert provider.received_language == "en"
    assert output.transcription.text == "stub transcript"
    assert output.transcription.model_version == "stub-whisper-test-double"


@pytest.mark.asyncio
async def test_diarization_stage_downloads_audio_and_forwards_to_provider():
    storage = InMemoryStorage()
    await storage.upload("processed.wav", b"canonical-wav-bytes")
    provider = StubDiarizationProvider()

    stage = SpeakerDiarizationStage(storage, provider)
    output = await stage.run(DiarizationInput(processed_storage_key="processed.wav", sample_rate=16000))

    assert output.diarization.num_speakers == 1
    assert output.diarization.speaker_segments[0].speaker_label == "speaker_0"


@pytest.mark.asyncio
async def test_alignment_stage_runs_the_pure_algorithm():
    stage = TranscriptAlignmentStage()
    output = await stage.run(
        AlignmentInput(
            transcript_segments=[TranscriptSegment(start_ms=0, end_ms=1000, text="hi", confidence=0.9)],
            speaker_segments=[SpeakerSegment(speaker_label="speaker_0", start_ms=0, end_ms=1000)],
        )
    )
    assert len(output.turns) == 1
    assert output.turns[0].speaker_label == "speaker_0"


def test_stage_names_are_explicit_and_distinct():
    names = {
        AudioPreprocessingStage.name,
        WhisperTranscriptionStage.name,
        SpeakerDiarizationStage.name,
        TranscriptAlignmentStage.name,
    }
    assert names == {
        "audio_preprocessing",
        "whisper_transcription",
        "speaker_diarization",
        "transcript_alignment",
    }
