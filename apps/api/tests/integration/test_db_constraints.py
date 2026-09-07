"""Exercises actual Postgres-enforced constraints - unique, check, and
foreign key/cascade behavior - directly against the ORM layer, independent
of API-level validation. These prove the schema itself is correct even if an
application-layer check were ever bypassed or buggy.
"""
from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from voxmind.models.aligned_turn import AlignedTurn
from voxmind.models.audio_asset import AudioAsset
from voxmind.models.audio_processing_job import AudioProcessingJob
from voxmind.models.conversation import Conversation
from voxmind.models.emotion_prediction import EmotionPrediction
from voxmind.models.emotion_processing_job import EmotionProcessingJob
from voxmind.models.message import Message
from voxmind.models.model_version import ModelVersion
from voxmind.models.speaker_segment import SpeakerSegment
from voxmind.models.transcript_segment import TranscriptSegment
from voxmind.models.user import User


@pytest.mark.asyncio
async def test_duplicate_email_violates_unique_constraint(db_session):
    db_session.add(User(email="dup@voxmind.dev", hashed_password="x"))
    await db_session.flush()

    db_session.add(User(email="dup@voxmind.dev", hashed_password="y"))
    with pytest.raises(IntegrityError):
        await db_session.flush()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_invalid_conversation_status_violates_check_constraint(db_session):
    user = User(email="checkc@voxmind.dev", hashed_password="x")
    db_session.add(user)
    await db_session.flush()

    db_session.add(Conversation(user_id=user.id, status="not-a-real-status"))
    with pytest.raises(IntegrityError):
        await db_session.flush()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_invalid_message_role_violates_check_constraint(db_session):
    user = User(email="checkr@voxmind.dev", hashed_password="x")
    db_session.add(user)
    await db_session.flush()
    conversation = Conversation(user_id=user.id)
    db_session.add(conversation)
    await db_session.flush()

    db_session.add(Message(session_id=conversation.id, role="not-a-real-role", content="hi"))
    with pytest.raises(IntegrityError):
        await db_session.flush()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_message_requires_existing_conversation(db_session):
    db_session.add(Message(session_id=uuid.uuid4(), role="user", content="orphaned"))
    with pytest.raises(IntegrityError):
        await db_session.flush()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_invalid_audio_asset_kind_violates_check_constraint(db_session):
    user = User(email="checkkind@voxmind.dev", hashed_password="x")
    db_session.add(user)
    await db_session.flush()
    conversation = Conversation(user_id=user.id)
    db_session.add(conversation)
    await db_session.flush()

    db_session.add(
        AudioAsset(session_id=conversation.id, storage_key="x", kind="not-a-real-kind")
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_invalid_processing_job_status_violates_check_constraint(db_session):
    user = User(email="checkjob@voxmind.dev", hashed_password="x")
    db_session.add(user)
    await db_session.flush()
    conversation = Conversation(user_id=user.id)
    db_session.add(conversation)
    await db_session.flush()
    asset = AudioAsset(session_id=conversation.id, storage_key="x")
    db_session.add(asset)
    await db_session.flush()

    db_session.add(
        AudioProcessingJob(session_id=conversation.id, audio_asset_id=asset.id, status="bogus-status")
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_deleting_message_cascades_to_speech_artifacts(db_session):
    """Deleting the Message a transcript is attached to must remove its
    transcript_segments/speaker_segments/aligned_turns too - verified
    against the real database, not asserted from the model definition."""
    user = User(email="cascade@voxmind.dev", hashed_password="x")
    db_session.add(user)
    await db_session.flush()
    conversation = Conversation(user_id=user.id)
    db_session.add(conversation)
    await db_session.flush()
    message = Message(session_id=conversation.id, role="user", content="hello")
    db_session.add(message)
    await db_session.flush()

    db_session.add(
        TranscriptSegment(
            message_id=message.id, start_ms=0, end_ms=1000, text="hello", stt_model_version="test"
        )
    )
    db_session.add(
        SpeakerSegment(
            message_id=message.id,
            speaker_label="speaker_0",
            start_ms=0,
            end_ms=1000,
            diarization_model_version="test",
        )
    )
    db_session.add(
        AlignedTurn(message_id=message.id, speaker_label="speaker_0", start_ms=0, end_ms=1000, text="hello")
    )
    await db_session.flush()

    await db_session.delete(message)
    await db_session.flush()

    remaining_transcript = (
        await db_session.execute(select(TranscriptSegment).where(TranscriptSegment.message_id == message.id))
    ).scalars().all()
    remaining_speaker = (
        await db_session.execute(select(SpeakerSegment).where(SpeakerSegment.message_id == message.id))
    ).scalars().all()
    remaining_turns = (
        await db_session.execute(select(AlignedTurn).where(AlignedTurn.message_id == message.id))
    ).scalars().all()

    assert remaining_transcript == []
    assert remaining_speaker == []
    assert remaining_turns == []
    await db_session.rollback()


@pytest.mark.asyncio
async def test_deleting_original_audio_asset_cascades_to_processed_asset(db_session):
    user = User(email="cascadeaudio@voxmind.dev", hashed_password="x")
    db_session.add(user)
    await db_session.flush()
    conversation = Conversation(user_id=user.id)
    db_session.add(conversation)
    await db_session.flush()
    original = AudioAsset(session_id=conversation.id, storage_key="original.wav", kind="original")
    db_session.add(original)
    await db_session.flush()
    processed = AudioAsset(
        session_id=conversation.id,
        storage_key="processed.wav",
        kind="processed",
        source_asset_id=original.id,
    )
    db_session.add(processed)
    await db_session.flush()
    processed_id = processed.id

    await db_session.delete(original)
    await db_session.flush()

    # A fresh SELECT, not session.get(): the `processed` object is still in
    # this session's identity map, and .get() would return that cached
    # (now stale) instance without re-querying the database.
    remaining = (
        await db_session.execute(select(AudioAsset).where(AudioAsset.id == processed_id))
    ).scalar_one_or_none()
    assert remaining is None
    await db_session.rollback()


async def _make_conversation_with_aligned_turn(db_session):
    user = User(email=f"emotion-fk-{uuid.uuid4().hex[:8]}@voxmind.dev", hashed_password="x")
    db_session.add(user)
    await db_session.flush()
    conversation = Conversation(user_id=user.id)
    db_session.add(conversation)
    await db_session.flush()
    message = Message(session_id=conversation.id, role="user", content="hello")
    db_session.add(message)
    await db_session.flush()
    turn = AlignedTurn(message_id=message.id, speaker_label="speaker_0", start_ms=0, end_ms=1000, text="hi")
    db_session.add(turn)
    await db_session.flush()
    model_version = ModelVersion(component="emotion_classifier", version_tag="test", trained=True)
    db_session.add(model_version)
    await db_session.flush()
    return conversation, message, turn, model_version


@pytest.mark.asyncio
async def test_invalid_emotion_processing_job_status_violates_check_constraint(db_session):
    conversation, message, _turn, _model_version = await _make_conversation_with_aligned_turn(db_session)

    db_session.add(
        EmotionProcessingJob(session_id=conversation.id, message_id=message.id, status="not-a-real-status")
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_deleting_aligned_turn_cascades_to_emotion_predictions(db_session):
    conversation, _message, turn, model_version = await _make_conversation_with_aligned_turn(db_session)
    del conversation

    prediction = EmotionPrediction(
        aligned_turn_id=turn.id,
        model_version_id=model_version.id,
        predicted_label="happy",
        confidence=0.9,
        probabilities={"happy": 0.9, "sad": 0.1},
    )
    db_session.add(prediction)
    await db_session.flush()
    prediction_id = prediction.id

    await db_session.delete(turn)
    await db_session.flush()

    remaining = (
        await db_session.execute(select(EmotionPrediction).where(EmotionPrediction.id == prediction_id))
    ).scalar_one_or_none()
    assert remaining is None
    await db_session.rollback()


@pytest.mark.asyncio
async def test_cannot_delete_model_version_referenced_by_a_prediction(db_session):
    """ON DELETE RESTRICT: a model version that produced real predictions
    can never be deleted out from under its audit trail."""
    conversation, _message, turn, model_version = await _make_conversation_with_aligned_turn(db_session)
    del conversation

    db_session.add(
        EmotionPrediction(
            aligned_turn_id=turn.id,
            model_version_id=model_version.id,
            predicted_label="happy",
            confidence=0.9,
            probabilities={"happy": 0.9, "sad": 0.1},
        )
    )
    await db_session.flush()

    await db_session.delete(model_version)
    with pytest.raises(IntegrityError):
        await db_session.flush()
    await db_session.rollback()
