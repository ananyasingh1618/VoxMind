from __future__ import annotations

from fastapi import APIRouter, Cookie, Depends, Header, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.api.deps import get_current_user
from voxmind.core.config import Settings, get_settings
from voxmind.core.cookies import clear_auth_cookies, set_auth_cookies, verify_csrf
from voxmind.core.exceptions import CsrfValidationError, InvalidCredentialsError
from voxmind.core.rate_limit import rate_limit_by_ip, rate_limit_by_user
from voxmind.db.session import get_db
from voxmind.models.user import User
from voxmind.schemas.auth import AccessTokenResponse, LoginRequest, RegisterRequest, UserOut
from voxmind.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])

# Category A: unauthenticated by nature (no user exists yet, or the caller
# is presenting a cookie rather than a bearer token), so keyed by IP - the
# classic brute-force/credential-stuffing target. See core/rate_limit.py.
_auth_rate_limit = Depends(rate_limit_by_ip("auth", "RATE_LIMIT_AUTH_PER_WINDOW"))


def _token_response(user: User, tokens, settings: Settings) -> AccessTokenResponse:
    return AccessTokenResponse(
        access_token=tokens.access_token,
        expires_in_seconds=settings.ACCESS_TOKEN_TTL_MINUTES * 60,
        user=UserOut.model_validate(user),
    )


@router.post(
    "/register", response_model=UserOut, status_code=status.HTTP_201_CREATED, dependencies=[_auth_rate_limit]
)
async def register(
    body: RegisterRequest,
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> UserOut:
    user = await AuthService(session, settings).register(email=body.email, password=body.password)
    return UserOut.model_validate(user)


@router.post("/login", response_model=AccessTokenResponse, dependencies=[_auth_rate_limit])
async def login(
    body: LoginRequest,
    response: Response,
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> AccessTokenResponse:
    user, tokens = await AuthService(session, settings).login(email=body.email, password=body.password)
    set_auth_cookies(response, tokens, settings)
    return _token_response(user, tokens, settings)


@router.post("/refresh", response_model=AccessTokenResponse, dependencies=[_auth_rate_limit])
async def refresh(
    response: Response,
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
    refresh_cookie: str | None = Cookie(default=None, alias="voxmind_refresh"),
    csrf_cookie: str | None = Cookie(default=None, alias="voxmind_csrf"),
    csrf_header: str | None = Header(default=None, alias="X-CSRF-Token"),
) -> AccessTokenResponse:
    if not verify_csrf(csrf_cookie=csrf_cookie, csrf_header=csrf_header):
        raise CsrfValidationError("Missing or mismatched CSRF token.")
    if not refresh_cookie:
        raise InvalidCredentialsError("No refresh token present.")

    user, tokens = await AuthService(session, settings).refresh(raw_refresh_token=refresh_cookie)
    set_auth_cookies(response, tokens, settings)
    return _token_response(user, tokens, settings)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def logout(
    response: Response,
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
    refresh_cookie: str | None = Cookie(default=None, alias="voxmind_refresh"),
    csrf_cookie: str | None = Cookie(default=None, alias="voxmind_csrf"),
    csrf_header: str | None = Header(default=None, alias="X-CSRF-Token"),
) -> None:
    if not verify_csrf(csrf_cookie=csrf_cookie, csrf_header=csrf_header):
        raise CsrfValidationError("Missing or mismatched CSRF token.")
    if refresh_cookie:
        await AuthService(session, settings).logout(raw_refresh_token=refresh_cookie)
    clear_auth_cookies(response, settings)


@router.post(
    "/logout-everywhere",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    dependencies=[Depends(rate_limit_by_user("default", "RATE_LIMIT_DEFAULT_PER_WINDOW"))],
)
async def logout_everywhere(
    response: Response,
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
    current_user: User = Depends(get_current_user),
    csrf_cookie: str | None = Cookie(default=None, alias="voxmind_csrf"),
    csrf_header: str | None = Header(default=None, alias="X-CSRF-Token"),
) -> None:
    """Revokes every active refresh-token session for this user, across all
    devices/families - stronger than /logout, which only revokes the current
    session's family."""
    if not verify_csrf(csrf_cookie=csrf_cookie, csrf_header=csrf_header):
        raise CsrfValidationError("Missing or mismatched CSRF token.")
    await AuthService(session, settings).logout_everywhere(user_id=current_user.id)
    clear_auth_cookies(response, settings)
