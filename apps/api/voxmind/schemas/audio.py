from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel


class AudioAssetOut(BaseModel):
    id: uuid.UUID
    original_filename: str | None
    content_type: str | None
    size_bytes: int | None
    duration_ms: int | None
    sample_rate: int | None
    channels: int | None
    format: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ProcessAudioRequest(BaseModel):
    language: str | None = None


class AudioProcessingJobOut(BaseModel):
    id: uuid.UUID
    audio_asset_id: uuid.UUID
    message_id: uuid.UUID | None
    status: str
    error_code: str | None
    error_message: str | None
    diarization_status: str | None
    diarization_error: str | None
    model_versions: dict
    stage_durations_ms: dict
    created_at: datetime
    completed_at: datetime | None

    model_config = {"from_attributes": True}


class TranscriptSegmentOut(BaseModel):
    start_ms: int
    end_ms: int
    text: str
    confidence: float | None

    model_config = {"from_attributes": True}


class SpeakerSegmentOut(BaseModel):
    speaker_label: str
    start_ms: int
    end_ms: int
    confidence: float | None

    model_config = {"from_attributes": True}


class AlignedTurnOut(BaseModel):
    id: uuid.UUID
    speaker_label: str | None
    start_ms: int
    end_ms: int
    text: str

    model_config = {"from_attributes": True}


class TranscriptOut(BaseModel):
    message_id: uuid.UUID
    transcript_segments: list[TranscriptSegmentOut]
    speaker_segments: list[SpeakerSegmentOut]
    aligned_turns: list[AlignedTurnOut]
