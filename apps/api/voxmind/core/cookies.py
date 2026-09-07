"""Cookie helpers for the refresh-token/CSRF pair. Centralized so every
endpoint that touches auth cookies uses identical flags — a mismatch between
set and clear (e.g. different Path) would leave stale cookies behind.
"""
from __future__ import annotations

from fastapi import Response

from voxmind.core.config import Settings
from voxmind.services.auth_service import AuthTokens

# The refresh cookie is httpOnly and only ever needs to be sent to the auth
# endpoints themselves (/auth/refresh, /auth/logout) - narrowly scoping its
# Path is a deliberate, real security reduction of its exposure, not an
# oversight.
REFRESH_COOKIE_PATH = "/api/v1/auth"
# The CSRF cookie is deliberately NOT httpOnly - the frontend's own
# `useBootstrapSession` reads it via `document.cookie` on every page load, as
# the only readable signal that a session might exist worth attempting a
# silent /auth/refresh for (the httpOnly refresh cookie itself is invisible
# to JS by design). A browser's `document.cookie` only ever exposes cookies
# whose Path matches the *current page's* URL - real bug found via a real
# browser walkthrough (Playwright, not just source inspection): this cookie
# used to share the auth endpoints' narrow /api/v1/auth Path, so it was
# genuinely invisible to `document.cookie` on every real frontend route
# (/app/conversations, /app/analytics, ...), silently defeating the reload
# flow it was created for - the httpOnly refresh cookie and the real backend
# session were always fine, but the frontend could never see a reason to
# even attempt using them. Must stay "/" - it needs to be visible from every
# route in the SPA, not just the auth ones.
CSRF_COOKIE_PATH = "/"


def set_auth_cookies(response: Response, tokens: AuthTokens, settings: Settings) -> None:
    response.set_cookie(
        key=settings.REFRESH_COOKIE_NAME,
        value=tokens.refresh_token,
        max_age=settings.REFRESH_TOKEN_TTL_DAYS * 24 * 3600,
        path=REFRESH_COOKIE_PATH,
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite=settings.COOKIE_SAMESITE,
    )
    response.set_cookie(
        key=settings.CSRF_COOKIE_NAME,
        value=tokens.csrf_token,
        max_age=settings.REFRESH_TOKEN_TTL_DAYS * 24 * 3600,
        path=CSRF_COOKIE_PATH,
        httponly=False,
        secure=settings.COOKIE_SECURE,
        samesite=settings.COOKIE_SAMESITE,
    )


def clear_auth_cookies(response: Response, settings: Settings) -> None:
    response.delete_cookie(key=settings.REFRESH_COOKIE_NAME, path=REFRESH_COOKIE_PATH)
    response.delete_cookie(key=settings.CSRF_COOKIE_NAME, path=CSRF_COOKIE_PATH)


def verify_csrf(*, csrf_cookie: str | None, csrf_header: str | None) -> bool:
    return bool(csrf_cookie) and bool(csrf_header) and csrf_cookie == csrf_header
