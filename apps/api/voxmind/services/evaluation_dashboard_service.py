"""Read-only application service behind the Phase 7 evaluation dashboard.
Every field returned here is read directly from a persisted `EvaluationRun`
row created by a real, already-executed evaluation run (see
ml/evaluation/evaluate_*.py) - this service computes nothing itself, it
only reports genuinely evaluated results. Not user-scoped, same reasoning
as `GET /models/{component}` and the analytics dashboard's model-version
section: an evaluation run measures a shared pipeline/model, not private
per-user data.
"""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.repositories.evaluation_run_repository import EvaluationRunRepository
from voxmind.schemas.evaluation import (
    EVALUATION_TYPES,
    EvaluationDashboard,
    EvaluationRunSummary,
    EvaluationTypeSummary,
)


class EvaluationDashboardService:
    def __init__(self, session: AsyncSession) -> None:
        self._runs = EvaluationRunRepository(session)

    async def get_dashboard(self) -> EvaluationDashboard:
        latest_by_type = await self._runs.list_latest_per_type()
        types = [
            EvaluationTypeSummary(
                evaluation_type=evaluation_type,
                status="evaluated" if evaluation_type in latest_by_type else "never_run",
                latest=(
                    EvaluationRunSummary.model_validate(latest_by_type[evaluation_type])
                    if evaluation_type in latest_by_type
                    else None
                ),
            )
            for evaluation_type in EVALUATION_TYPES
        ]
        return EvaluationDashboard(types=types)

    async def list_runs(self, evaluation_type: str, *, limit: int = 20) -> list[EvaluationRunSummary]:
        runs = await self._runs.list_by_type(evaluation_type, limit=limit)
        return [EvaluationRunSummary.model_validate(run) for run in runs]
