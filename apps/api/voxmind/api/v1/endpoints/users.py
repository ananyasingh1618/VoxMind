from __future__ import annotations

from fastapi import APIRouter, Depends

from voxmind.api.deps import get_current_user
from voxmind.models.user import User
from voxmind.schemas.auth import UserOut

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me", response_model=UserOut)
async def get_me(current_user: User = Depends(get_current_user)) -> UserOut:
    return UserOut.model_validate(current_user)
