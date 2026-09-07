"""Model/experiment registry.

Extended in Phase 3 to carry everything needed to identify, reload, and
audit a specific trained model: architecture/label mapping/training config
alongside the metrics genuinely computed for it. A new training run always
INSERTs a new row (a new `version_tag`) - existing rows are never
overwritten, so a prediction's `model_version_id` always resolves to the
exact configuration that produced it, even after later models are trained.

Empty until a real training run exists (see ml/training/train_emotion_model.py)
- no row here claims `trained=True` unless `ml/evaluation/evaluate_emotion_model.py`
has genuinely measured it.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from voxmind.db.base import Base


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    component: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    version_tag: Mapped[str] = mapped_column(String(100), nullable=False)
    task: Mapped[str | None] = mapped_column(String(100), nullable=True)
    base_model: Mapped[str | None] = mapped_column(
        String(200), nullable=True, doc="e.g. the Wav2Vec2 checkpoint id used for embeddings."
    )
    label_mapping: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True, doc='e.g. {"0": "neutral", "1": "happy", ...}'
    )
    training_config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    dataset_info: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True, doc="source, split sizes, class distribution, speaker-split strategy."
    )
    artifact_storage_key: Mapped[str | None] = mapped_column(
        String(500), nullable=True, doc="StorageBackend key for the checkpoint - never a raw filesystem path."
    )
    trained: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    trained_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
