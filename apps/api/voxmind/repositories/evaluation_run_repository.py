from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.models.evaluation_run import EvaluationRun


class EvaluationRunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, **fields: object) -> EvaluationRun:
        row = EvaluationRun(**fields)
        self._session.add(row)
        await self._session.flush()
        return row

    async def mark_completed(
        self, run: EvaluationRun, *, status: str, metrics: dict, errors: list | None = None
    ) -> EvaluationRun:
        run.status = status
        run.metrics = metrics
        if errors is not None:
            run.errors = errors
        run.completed_at = datetime.utcnow()
        await self._session.flush()
        return run

    async def get(self, run_id: uuid.UUID) -> EvaluationRun | None:
        return await self._session.get(EvaluationRun, run_id)

    async def list_by_type(self, evaluation_type: str, *, limit: int = 50) -> list[EvaluationRun]:
        stmt = (
            select(EvaluationRun)
            .where(EvaluationRun.evaluation_type == evaluation_type)
            .order_by(EvaluationRun.created_at.desc())
            .limit(limit)
        )
        return list((await self._session.execute(stmt)).scalars().all())

    async def list_latest_per_type(self) -> dict[str, EvaluationRun]:
        """The most recent run of each evaluation type, for the dashboard summary."""
        from voxmind.models.evaluation_run import VALID_EVALUATION_TYPES

        latest: dict[str, EvaluationRun] = {}
        for evaluation_type in VALID_EVALUATION_TYPES:
            stmt = (
                select(EvaluationRun)
                .where(EvaluationRun.evaluation_type == evaluation_type)
                .order_by(EvaluationRun.created_at.desc())
                .limit(1)
            )
            row = (await self._session.execute(stmt)).scalars().first()
            if row is not None:
                latest[evaluation_type] = row
        return latest

    async def list_all(self, *, limit: int = 100) -> list[EvaluationRun]:
        stmt = select(EvaluationRun).order_by(EvaluationRun.created_at.desc()).limit(limit)
        return list((await self._session.execute(stmt)).scalars().all())
