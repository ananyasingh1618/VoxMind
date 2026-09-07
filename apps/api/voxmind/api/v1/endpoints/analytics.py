from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.api.deps import get_current_user, get_settings
from voxmind.core.config import Settings
from voxmind.db.session import get_db
from voxmind.models.user import User
from voxmind.schemas.analytics import AnalyticsDashboard
from voxmind.services.analytics_service import AnalyticsService

router = APIRouter(prefix="/analytics", tags=["analytics"])


def get_analytics_service(
    session: AsyncSession = Depends(get_db), settings: Settings = Depends(get_settings)
) -> AnalyticsService:
    return AnalyticsService(session, settings=settings)


@router.get("/dashboard", response_model=AnalyticsDashboard)
async def get_dashboard(
    current_user: User = Depends(get_current_user),
    service: AnalyticsService = Depends(get_analytics_service),
) -> AnalyticsDashboard:
    """Every field is computed from a real aggregate query over the calling
    user's own data - never fabricated. A metric with no underlying data
    yet returns a genuine zero/empty result (`sample_count=0`, `avg_ms=null`,
    an empty `by_label` list) rather than an invented value - see
    docs/analytics.md.
    """
    return await service.get_dashboard(user_id=current_user.id)
