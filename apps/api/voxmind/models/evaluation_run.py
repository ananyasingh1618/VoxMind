"""Phase 7: the single, genuine persistence mechanism behind every
evaluation VoxMind ever runs (STT, emotion, retrieval, grounding, system).

Not user/conversation-scoped, deliberately - same reasoning as
`ModelVersion`: an evaluation run measures a component or pipeline against
a versioned dataset/configuration, not a private user's data. A new
evaluation always INSERTs a new row; existing rows are never overwritten or
deleted, so historical evaluation runs remain inspectable even after a
newer run of the same type exists (see docs/evaluation.md).

`status` distinguishes four real, distinct outcomes that must never be
blurred (see the Phase 7 spec):
  - "completed": the evaluation ran and every requested metric was computed.
  - "partial": the evaluation ran but some samples/metrics were skipped
    (e.g. a sample's audio file was missing) - `errors` records why.
  - "failed": the evaluation attempted to run but could not complete at all
    (e.g. the model checkpoint failed to load).
  - "unavailable": the evaluation could not even be attempted because a
    required prerequisite (ground-truth labels, a dataset, credentials)
    does not exist yet - `notes` records exactly what's missing. This is
    the honest alternative to fabricating a metric.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import CheckConstraint, DateTime, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from voxmind.db.base import Base

VALID_EVALUATION_TYPES = ("stt", "emotion", "retrieval", "grounding", "system")
VALID_STATUSES = ("completed", "partial", "failed", "unavailable")


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    __table_args__ = (
        CheckConstraint(f"evaluation_type IN {VALID_EVALUATION_TYPES}", name="valid_evaluation_type"),
        CheckConstraint(f"status IN {VALID_STATUSES}", name="valid_status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    evaluation_type: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    dataset_version: Mapped[str | None] = mapped_column(
        String(200), nullable=True, doc="e.g. 'stt-eval-v1', 'ravdess-v1/test', or null when not dataset-based."
    )
    model_version: Mapped[str | None] = mapped_column(
        String(200), nullable=True, doc="e.g. 'faster-whisper-small', 'ravdess-v1', 'claude-sonnet-5', or a provider id."
    )
    configuration: Mapped[dict] = mapped_column(
        JSONB, nullable=False, default=dict, doc="Exact parameters used - split, thresholds, provider config, etc."
    )
    code_version: Mapped[str | None] = mapped_column(
        String(100), nullable=True, doc="Short git commit hash of the code that produced this run, where available."
    )
    random_seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    metrics: Mapped[dict] = mapped_column(
        JSONB, nullable=False, default=dict, doc="Real computed metrics only - empty when status is 'unavailable'."
    )
    errors: Mapped[list] = mapped_column(
        JSONB, nullable=False, default=list, doc="Per-sample or per-stage skip/failure records, if any."
    )
    notes: Mapped[str | None] = mapped_column(
        String(2000), nullable=True, doc="Free-text explanation - required when status is 'unavailable'."
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=lambda: datetime.now(timezone.utc)
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        default=lambda: datetime.now(timezone.utc),
        index=True,
        doc="A Python-side (not server-side) default: two runs created in the same transaction/commit "
        "would otherwise share Postgres's transaction-start `now()` value, making 'latest run per "
        "type' ambiguous - see the regression test in tests/integration/test_evaluation_dashboard.py.",
    )
