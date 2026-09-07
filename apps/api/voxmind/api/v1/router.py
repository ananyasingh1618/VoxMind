from fastapi import APIRouter, Depends

from voxmind.api.v1.endpoints import (
    analytics,
    audio,
    auth,
    conversations,
    emotion,
    evaluation,
    health,
    incongruence,
    insights,
    knowledge,
    memory,
    models,
    nlp,
    rag,
    users,
    voice,
)
from voxmind.core.rate_limit import rate_limit_by_user

# Category D (ordinary authenticated API traffic - see core/rate_limit.py)
# applied once, here, to every router except:
#  - health.router: liveness/readiness must never be rate-limited - an
#    orchestrator's healthcheck polling this frequently is exactly the
#    traffic this endpoint exists to always answer.
#  - auth.router: register/login already get their own, stricter, IP-based
#    "auth" category (applied per-route in auth.py) - applying a
#    user-based dependency router-wide here would break those two routes
#    outright, since they run before any user exists to key on.
# Routers whose specific endpoints already carry a stricter "expensive" or
# "knowledge" category (audio, voice, emotion, knowledge, rag) still get
# this default too - harmless, redundant layering, since the stricter
# category's lower limit always binds first in practice.
_default_rate_limit = Depends(rate_limit_by_user("default", "RATE_LIMIT_DEFAULT_PER_WINDOW"))

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(users.router, dependencies=[_default_rate_limit])
api_router.include_router(conversations.router, dependencies=[_default_rate_limit])
api_router.include_router(audio.router, dependencies=[_default_rate_limit])
api_router.include_router(emotion.router, dependencies=[_default_rate_limit])
api_router.include_router(models.router, dependencies=[_default_rate_limit])
api_router.include_router(nlp.router, dependencies=[_default_rate_limit])
api_router.include_router(incongruence.router, dependencies=[_default_rate_limit])
api_router.include_router(knowledge.router, dependencies=[_default_rate_limit])
api_router.include_router(memory.router, dependencies=[_default_rate_limit])
api_router.include_router(rag.router, dependencies=[_default_rate_limit])
api_router.include_router(voice.router, dependencies=[_default_rate_limit])
api_router.include_router(analytics.router, dependencies=[_default_rate_limit])
api_router.include_router(insights.router, dependencies=[_default_rate_limit])
api_router.include_router(evaluation.router, dependencies=[_default_rate_limit])
