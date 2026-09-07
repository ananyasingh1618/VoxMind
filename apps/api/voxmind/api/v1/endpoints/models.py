"""Read-only visibility into the model/experiment registry
(`model_versions`). Not conversation-scoped - model metadata isn't private
user data, it's the same for every authenticated user. Never exposes
`artifact_storage_key` (an internal storage location) or full
`training_config`/`dataset_info` - just enough to know what a prediction
came from.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.api.deps import get_current_user
from voxmind.db.session import get_db
from voxmind.models.user import User
from voxmind.repositories.model_version_repository import ModelVersionRepository
from voxmind.schemas.emotion import ModelVersionOut

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/{component}", response_model=list[ModelVersionOut])
async def list_model_versions(
    component: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> list[ModelVersionOut]:
    versions = await ModelVersionRepository(session).list_for_component(component)
    return [ModelVersionOut.model_validate(v) for v in versions]
