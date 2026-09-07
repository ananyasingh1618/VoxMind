"""Auth application service — the only place register/login/refresh/logout
business logic lives. Endpoints stay thin: parse request, call this, shape
the response.
"""
from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.core.config import Settings
from voxmind.core.exceptions import ConflictError, InvalidCredentialsError, ReuseDetectedError
from voxmind.core.security import (
    create_access_token,
    generate_csrf_token,
    hash_password,
    verify_password,
)
from voxmind.models.user import User
from voxmind.repositories.refresh_token_repository import RefreshTokenRepository
from voxmind.repositories.user_repository import UserRepository


class AuthTokens:
    def __init__(self, *, access_token: str, refresh_token: str, csrf_token: str) -> None:
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.csrf_token = csrf_token


class AuthService:
    def __init__(self, session: AsyncSession, settings: Settings) -> None:
        self._session = session
        self._settings = settings
        self._users = UserRepository(session)
        self._tokens = RefreshTokenRepository(session)

    async def register(self, *, email: str, password: str) -> User:
        if await self._users.get_by_email(email) is not None:
            raise ConflictError("An account with this email already exists.")
        user = await self._users.create(email=email, hashed_password=hash_password(password))
        await self._session.commit()
        return user

    async def login(self, *, email: str, password: str) -> tuple[User, AuthTokens]:
        user = await self._users.get_by_email(email)
        if user is None or not verify_password(password, user.hashed_password):
            raise InvalidCredentialsError("Incorrect email or password.")
        tokens = await self._issue_session(user)
        await self._session.commit()
        return user, tokens

    async def refresh(self, *, raw_refresh_token: str) -> tuple[User, AuthTokens]:
        try:
            new_raw, row = await self._tokens.rotate(raw_token=raw_refresh_token, settings=self._settings)
        except ReuseDetectedError:
            # The family revocation performed inside rotate() must survive
            # even though we're about to raise - it's the security response.
            await self._session.commit()
            raise
        except Exception:
            await self._session.rollback()
            raise

        user = await self._users.get_by_id(row.user_id)
        if user is None:
            await self._session.rollback()
            raise InvalidCredentialsError("User for this refresh token no longer exists.")

        access_token = create_access_token(user_id=user.id, settings=self._settings)
        tokens = AuthTokens(
            access_token=access_token, refresh_token=new_raw, csrf_token=generate_csrf_token()
        )
        await self._session.commit()
        return user, tokens

    async def logout(self, *, raw_refresh_token: str) -> None:
        """Revokes only the current session's token family."""
        await self._tokens.revoke_by_raw_token(raw_refresh_token)
        await self._session.commit()

    async def logout_everywhere(self, *, user_id: uuid.UUID) -> None:
        """Revokes every active refresh token for the user, across every
        family/device — a stronger action than plain logout()."""
        await self._tokens.revoke_all_for_user(user_id)
        await self._session.commit()

    async def _issue_session(self, user: User) -> AuthTokens:
        access_token = create_access_token(user_id=user.id, settings=self._settings)
        raw_refresh, _ = await self._tokens.issue(user_id=user.id, settings=self._settings)
        return AuthTokens(
            access_token=access_token, refresh_token=raw_refresh, csrf_token=generate_csrf_token()
        )
