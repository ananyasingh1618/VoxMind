"""First real consumer of the `model_versions` table (empty since Phase 1,
which created it only so this FK target would exist before Phase 3 needed
it - see models/model_version.py).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.models.model_version import ModelVersion


class ModelVersionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        component: str,
        version_tag: str,
        task: str | None = None,
        base_model: str | None = None,
        label_mapping: dict | None = None,
        training_config: dict | None = None,
        dataset_info: dict | None = None,
        artifact_storage_key: str | None = None,
        trained: bool = False,
        metrics: dict | None = None,
        mlflow_run_id: str | None = None,
        is_active: bool = False,
    ) -> ModelVersion:
        """Always INSERTs a new row - callers never update an existing
        version's identity fields, so a prediction's `model_version_id`
        permanently resolves to the exact config that produced it."""
        version = ModelVersion(
            component=component,
            version_tag=version_tag,
            task=task,
            base_model=base_model,
            label_mapping=label_mapping,
            training_config=training_config,
            dataset_info=dataset_info,
            artifact_storage_key=artifact_storage_key,
            trained=trained,
            trained_at=datetime.now(timezone.utc) if trained else None,
            metrics=metrics,
            mlflow_run_id=mlflow_run_id,
            is_active=is_active,
        )
        self._session.add(version)
        await self._session.flush()
        return version

    async def get_by_id(self, model_version_id: uuid.UUID) -> ModelVersion | None:
        return await self._session.get(ModelVersion, model_version_id)

    async def get_active_for_component(self, component: str) -> ModelVersion | None:
        stmt = select(ModelVersion).where(
            ModelVersion.component == component,
            ModelVersion.is_active.is_(True),
            ModelVersion.trained.is_(True),
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def list_for_component(self, component: str) -> list[ModelVersion]:
        stmt = (
            select(ModelVersion)
            .where(ModelVersion.component == component)
            .order_by(ModelVersion.created_at.desc())
        )
        return list((await self._session.execute(stmt)).scalars().all())

    async def activate(self, model_version_id: uuid.UUID) -> ModelVersion:
        """Marks one version active for its component and deactivates any
        previously-active version of the same component - at most one
        active version per component at a time, never a silent overwrite of
        the version rows themselves."""
        version = await self._session.get(ModelVersion, model_version_id)
        if version is None:
            raise ValueError(f"Unknown model_version_id: {model_version_id}")

        stmt = select(ModelVersion).where(
            ModelVersion.component == version.component, ModelVersion.is_active.is_(True)
        )
        for currently_active in (await self._session.execute(stmt)).scalars().all():
            currently_active.is_active = False

        version.is_active = True
        await self._session.flush()
        return version
