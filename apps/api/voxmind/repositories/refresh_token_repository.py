"""Refresh-token issuance, rotation, and revocation.

Rotation is the one place in the app where naive logic would be a real
security/correctness bug, so the reasoning is documented inline rather than
left implicit:

Concurrency safety: `rotate()` takes a `SELECT ... FOR UPDATE` row lock on the
presented token before deciding anything. Two concurrent requests racing to
rotate the *same still-valid* token are therefore serialized by Postgres, not
by application code — the second request's SELECT blocks until the first
request's transaction commits or rolls back.

Reuse vs. benign race: once the first request commits, the second request's
(now unblocked) SELECT sees a row that is already revoked. Naively treating
*any* revoked-token presentation as theft would incorrectly nuke the user's
whole session family on ordinary races (a duplicated network retry, a second
browser tab firing refresh at the same moment). Instead, a short grace window
(`REFRESH_REUSE_GRACE_SECONDS`) after a *normal rotation* (`replaced_by_id` is
set) is treated as a benign race: the caller gets a plain 401 telling it a
concurrent request already rotated this token, without any revocation. Only
reuse outside that window, or reuse of a token that was revoked for a reason
other than normal rotation (explicit logout, or a prior reuse-detection
event, both of which leave `replaced_by_id` unset), is treated as theft and
revokes the entire token family.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.core.config import Settings
from voxmind.core.exceptions import InvalidCredentialsError, ReuseDetectedError, SessionExpiredError
from voxmind.core.security import generate_opaque_token, hash_token
from voxmind.models.refresh_token import RefreshToken


class RefreshTokenRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def issue(self, *, user_id: uuid.UUID, settings: Settings) -> tuple[str, RefreshToken]:
        """Issues a brand-new token family (used at login)."""
        raw_token = generate_opaque_token()
        row = RefreshToken(
            user_id=user_id,
            token_hash=hash_token(raw_token),
            family_id=uuid.uuid4(),
            expires_at=datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_TTL_DAYS),
        )
        self._session.add(row)
        await self._session.flush()
        return raw_token, row

    async def rotate(self, *, raw_token: str, settings: Settings) -> tuple[str, RefreshToken]:
        token_hash = hash_token(raw_token)
        stmt = select(RefreshToken).where(RefreshToken.token_hash == token_hash).with_for_update()
        row = (await self._session.execute(stmt)).scalar_one_or_none()

        if row is None:
            raise InvalidCredentialsError("Refresh token not recognized.")

        now = datetime.now(timezone.utc)

        if row.revoked_at is not None:
            was_normal_rotation = row.replaced_by_id is not None
            within_grace = (now - row.revoked_at) <= timedelta(
                seconds=settings.REFRESH_REUSE_GRACE_SECONDS
            )
            if was_normal_rotation and within_grace:
                raise InvalidCredentialsError(
                    "Refresh token was already rotated by a concurrent request; "
                    "retry with the token returned from that request."
                )
            await self.revoke_family(row.family_id)
            raise ReuseDetectedError("Refresh token reuse detected; session family revoked.")

        if row.expires_at < now:
            raise SessionExpiredError("Refresh token has expired.")

        new_raw, new_row = await self.issue(user_id=row.user_id, settings=settings)
        new_row.family_id = row.family_id  # descend from the same family, not a new one
        row.revoked_at = now
        row.replaced_by_id = new_row.id
        await self._session.flush()
        return new_raw, new_row

    async def revoke_family(self, family_id: uuid.UUID) -> None:
        stmt = (
            select(RefreshToken)
            .where(RefreshToken.family_id == family_id, RefreshToken.revoked_at.is_(None))
            .with_for_update()
        )
        rows = (await self._session.execute(stmt)).scalars().all()
        now = datetime.now(timezone.utc)
        for row in rows:
            row.revoked_at = now
        await self._session.flush()

    async def revoke_all_for_user(self, user_id: uuid.UUID) -> None:
        """'Log out everywhere': revokes every active token for the user across
        *all* families, not just the one presented at the call site."""
        stmt = (
            select(RefreshToken)
            .where(RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None))
            .with_for_update()
        )
        rows = (await self._session.execute(stmt)).scalars().all()
        now = datetime.now(timezone.utc)
        for row in rows:
            row.revoked_at = now
        await self._session.flush()

    async def revoke_by_raw_token(self, raw_token: str) -> None:
        """Normal logout: revokes only the current session's token family."""
        token_hash = hash_token(raw_token)
        stmt = select(RefreshToken).where(RefreshToken.token_hash == token_hash).with_for_update()
        row = (await self._session.execute(stmt)).scalar_one_or_none()
        if row is None:
            return
        await self.revoke_family(row.family_id)
