"""Persists and retrieves the structured speech artifacts (raw transcript
segments, raw speaker segments, and the derived aligned turns) attached to a
Message. Named `*Row` locally to avoid confusion with the identically-named
Pydantic pipeline models in services/speech/interfaces.py.
"""
from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.models.aligned_turn import AlignedTurn as AlignedTurnRow
from voxmind.models.speaker_segment import SpeakerSegment as SpeakerSegmentRow
from voxmind.models.transcript_segment import TranscriptSegment as TranscriptSegmentRow
from voxmind.services.speech.interfaces import (
    AlignedTurn,
    SpeakerSegment,
    TranscriptSegment,
)


class TranscriptRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add_transcript_segments(
        self, *, message_id: uuid.UUID, segments: list[TranscriptSegment], model_version: str
    ) -> None:
        for segment in segments:
            self._session.add(
                TranscriptSegmentRow(
                    message_id=message_id,
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                    text=segment.text,
                    confidence=segment.confidence,
                    stt_model_version=model_version,
                )
            )
        await self._session.flush()

    async def add_speaker_segments(
        self, *, message_id: uuid.UUID, segments: list[SpeakerSegment], model_version: str
    ) -> None:
        for segment in segments:
            self._session.add(
                SpeakerSegmentRow(
                    message_id=message_id,
                    speaker_label=segment.speaker_label,
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                    confidence=segment.confidence,
                    diarization_model_version=model_version,
                )
            )
        await self._session.flush()

    async def add_aligned_turns(self, *, message_id: uuid.UUID, turns: list[AlignedTurn]) -> None:
        for turn in turns:
            self._session.add(
                AlignedTurnRow(
                    message_id=message_id,
                    speaker_label=turn.speaker_label,
                    start_ms=turn.start_ms,
                    end_ms=turn.end_ms,
                    text=turn.text,
                )
            )
        await self._session.flush()

    async def get_transcript_segments(self, message_id: uuid.UUID) -> list[TranscriptSegmentRow]:
        stmt = (
            select(TranscriptSegmentRow)
            .where(TranscriptSegmentRow.message_id == message_id)
            .order_by(TranscriptSegmentRow.start_ms.asc())
        )
        return list((await self._session.execute(stmt)).scalars().all())

    async def get_speaker_segments(self, message_id: uuid.UUID) -> list[SpeakerSegmentRow]:
        stmt = (
            select(SpeakerSegmentRow)
            .where(SpeakerSegmentRow.message_id == message_id)
            .order_by(SpeakerSegmentRow.start_ms.asc())
        )
        return list((await self._session.execute(stmt)).scalars().all())

    async def get_aligned_turns(self, message_id: uuid.UUID) -> list[AlignedTurnRow]:
        stmt = (
            select(AlignedTurnRow)
            .where(AlignedTurnRow.message_id == message_id)
            .order_by(AlignedTurnRow.start_ms.asc())
        )
        return list((await self._session.execute(stmt)).scalars().all())
