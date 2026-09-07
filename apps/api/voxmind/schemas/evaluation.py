"""Evaluation dashboard response shapes (Phase 7). Every field reflects a
real, persisted `EvaluationRun` row - produced by actually executing one of
the ml/evaluation/evaluate_*.py scripts against real data, never fabricated
or estimated here. A type that has never been evaluated returns
`status="never_run"` with `latest=None`, not a placeholder metric - see
docs/evaluation.md.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel

EVALUATION_TYPES = ("stt", "emotion", "retrieval", "grounding", "system")


class EvaluationRunSummary(BaseModel):
    id: uuid.UUID
    evaluation_type: str
    dataset_version: str | None
    model_version: str | None
    configuration: dict
    code_version: str | None
    random_seed: int | None
    sample_count: int
    status: str
    metrics: dict
    errors: list
    notes: str | None
    started_at: datetime
    completed_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


class EvaluationTypeSummary(BaseModel):
    evaluation_type: str
    status: str
    """'evaluated' if at least one run exists for this type, 'never_run' otherwise."""
    latest: EvaluationRunSummary | None


class EvaluationDashboard(BaseModel):
    types: list[EvaluationTypeSummary]
