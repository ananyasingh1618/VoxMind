from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.api.deps import get_current_user
from voxmind.db.session import get_db
from voxmind.models.user import User
from voxmind.schemas.evaluation import EvaluationDashboard, EvaluationRunSummary
from voxmind.services.evaluation_dashboard_service import EvaluationDashboardService

EvaluationTypeParam = Literal["stt", "emotion", "retrieval", "grounding", "system"]

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


def get_evaluation_dashboard_service(session: AsyncSession = Depends(get_db)) -> EvaluationDashboardService:
    return EvaluationDashboardService(session)


@router.get("/dashboard", response_model=EvaluationDashboard)
async def get_evaluation_dashboard(
    _current_user: User = Depends(get_current_user),
    service: EvaluationDashboardService = Depends(get_evaluation_dashboard_service),
) -> EvaluationDashboard:
    """The most recent real evaluation run for each of stt/emotion/retrieval/
    grounding/system - `status="never_run"` and `latest=null` for any type
    that has never actually been evaluated in this environment, never a
    placeholder metric. See docs/evaluation.md and ml/evaluation/evaluate_*.py.
    """
    return await service.get_dashboard()


@router.get("/runs", response_model=list[EvaluationRunSummary])
async def list_evaluation_runs(
    evaluation_type: EvaluationTypeParam = Query(...),
    limit: int = Query(default=20, ge=1, le=100),
    _current_user: User = Depends(get_current_user),
    service: EvaluationDashboardService = Depends(get_evaluation_dashboard_service),
) -> list[EvaluationRunSummary]:
    """Historical runs of one evaluation type, newest first - lets a caller
    compare how a metric has changed across real evaluation runs over time.
    Every returned row is a real, previously persisted, never-overwritten
    `EvaluationRun`."""
    return await service.list_runs(evaluation_type, limit=limit)
