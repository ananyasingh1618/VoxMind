# ADR 0001: Auth token strategy

## Decision

Access tokens are short-lived JWTs (15 min) held in memory on the client only. Refresh tokens are opaque random values, stored server-side as a SHA-256 hash (never the raw value), delivered via an `HttpOnly` cookie, and rotated on every use.

## Why not a single long-lived JWT?

A single long-lived token can't be revoked before it expires without maintaining a blocklist, which reintroduces server-side state anyway - at that point, a short-lived access token plus a revocable refresh token gives the same revocability with a much smaller blast radius per leaked token (15 minutes vs. days).

## Why rotate refresh tokens on every use?

Rotation turns refresh-token theft into a detectable event: if a stolen token is used after the legitimate client has already rotated it, the reused (already-revoked) token is presented, which the server can distinguish from normal use and respond to by revoking the entire session family.

## Concurrency correctness

Naive rotation has a race: two legitimate concurrent requests presenting the same still-valid token could both see it as valid and both attempt to rotate it, corrupting the rotation chain - or worse, a naive "any reuse of a revoked token is theft" rule would treat an ordinary network retry as an attack and destroy the user's session.

The implementation (`apps/api/voxmind/repositories/refresh_token_repository.py`) fixes both problems:

1. `SELECT ... FOR UPDATE` on the presented token row serializes concurrent rotation attempts at the database level - only one request can rotate a given token, ever.
2. A short grace window (`REFRESH_REUSE_GRACE_SECONDS`, default 10s) distinguishes a benign race (a second request arriving just after the first legitimately rotated the same token) from genuine reuse (a token replayed well after rotation, or one revoked for an unrelated reason like logout). Only the latter revokes the entire token family.

This is covered by `apps/api/tests/integration/test_auth_endpoints.py::test_concurrent_refresh_race_does_not_revoke_family` and `test_refresh_reuse_outside_grace_window_revokes_entire_family`, both run against a real Postgres database.

## Logout scope

`/auth/logout` revokes only the current session's token family (one device/browser). `/auth/logout-everywhere` revokes every active refresh token for the user across every family - a deliberately stronger, separate action, not the default behavior of logout.
