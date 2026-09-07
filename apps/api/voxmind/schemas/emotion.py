from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel


class EmotionProcessingJobOut(BaseModel):
    id: uuid.UUID
    message_id: uuid.UUID
    model_version_id: uuid.UUID | None
    status: str
    error_code: str | None
    error_message: str | None
    turns_processed: int
    stage_durations_ms: dict
    created_at: datetime
    completed_at: datetime | None

    model_config = {"from_attributes": True}


class EmotionPredictionOut(BaseModel):
    id: uuid.UUID
    aligned_turn_id: uuid.UUID
    model_version_id: uuid.UUID
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]
    created_at: datetime

    model_config = {"from_attributes": True}


class ModelVersionOut(BaseModel):
    """Deliberately excludes `artifact_storage_key` (an internal storage
    location, not something a client should see or be able to use to
    request arbitrary paths) and `training_config`/`dataset_info` unless
    explicitly requested via the detail endpoint - the list view stays
    lightweight."""

    id: uuid.UUID
    component: str
    version_tag: str
    task: str | None
    base_model: str | None
    label_mapping: dict | None
    trained: bool
    trained_at: datetime | None
    metrics: dict | None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}
