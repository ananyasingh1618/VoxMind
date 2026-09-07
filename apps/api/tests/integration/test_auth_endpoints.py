from __future__ import annotations

import pytest

from voxmind.core.config import get_settings

REGISTER_PAYLOAD = {"email": "demo@voxmind.dev", "password": "correct-horse-battery-staple"}


def _csrf_headers(client) -> dict:
    token = client.cookies.get("voxmind_csrf")
    assert token, "CSRF cookie must be set after login"
    return {"X-CSRF-Token": token}


@pytest.mark.asyncio
async def test_register_creates_user_without_storing_plaintext_password(client, db_session):
    response = await client.post("/api/v1/auth/register", json=REGISTER_PAYLOAD)
    assert response.status_code == 201
    body = response.json()
    assert body["email"] == REGISTER_PAYLOAD["email"]
    assert "password" not in body
    assert "hashed_password" not in body


@pytest.mark.asyncio
async def test_register_duplicate_email_is_rejected(client):
    await client.post("/api/v1/auth/register", json=REGISTER_PAYLOAD)
    response = await client.post("/api/v1/auth/register", json=REGISTER_PAYLOAD)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "conflict"


@pytest.mark.asyncio
async def test_login_success_returns_access_token_and_sets_cookies(client):
    await client.post("/api/v1/auth/register", json=REGISTER_PAYLOAD)
    response = await client.post("/api/v1/auth/login", json=REGISTER_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["access_token"]
    assert body["user"]["email"] == REGISTER_PAYLOAD["email"]
    assert client.cookies.get("voxmind_refresh")
    assert client.cookies.get("voxmind_csrf")


@pytest.mark.asyncio
async def test_csrf_cookie_path_is_root_not_the_narrow_auth_path(client):
    """Real regression test for a real bug found via a real browser
    walkthrough (Playwright, not just source inspection): the CSRF cookie
    used to share the httpOnly refresh cookie's narrow `Path=/api/v1/auth`,
    which made it genuinely invisible to `document.cookie` on every real
    frontend route (e.g. /app/conversations) - a browser's `document.cookie`
    only ever exposes cookies whose Path matches the current page's URL.
    Since `useBootstrapSession` on the frontend relies on reading this exact
    cookie as its only signal to attempt a silent session-restoring
    `/auth/refresh` call on page load, this silently defeated that entire
    mechanism: a real page reload always logged the user out, even though
    the real backend session/refresh-token was completely intact. The
    refresh cookie itself is correctly, deliberately kept narrow (see
    core/cookies.py) - only the CSRF cookie must be root-scoped."""
    await client.post("/api/v1/auth/register", json=REGISTER_PAYLOAD)
    response = await client.post("/api/v1/auth/login", json=REGISTER_PAYLOAD)
    set_cookie_headers = response.headers.get_list("set-cookie")

    refresh_header = next(h for h in set_cookie_headers if h.startswith("voxmind_refresh="))
    csrf_header = next(h for h in set_cookie_headers if h.startswith("voxmind_csrf="))

    assert "Path=/api/v1/auth" in refresh_header
    assert "Path=/;" in csrf_header or csrf_header.rstrip().endswith("Path=/")


@pytest.mark.asyncio
async def test_login_with_wrong_password_is_rejected(client):
    await client.post("/api/v1/auth/register", json=REGISTER_PAYLOAD)
    response = await client.post(
        "/api/v1/auth/login", json={"email": REGISTER_PAYLOAD["email"], "password": "wrong-password"}
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_credentials"


@pytest.mark.asyncio
async def test_login_with_unknown_email_is_rejected_identically(client):
    response = await client.post(
        "/api/v1/auth/login", json={"email": "nobody@voxmind.dev", "password": "whatever123"}
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_credentials"


@pytest.mark.asyncio
async def test_protected_endpoint_rejects_missing_token(client):
    response = await client.get("/api/v1/users/me")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_protected_endpoint_rejects_garbage_token(client):
    response = await client.get(
        "/api/v1/users/me", headers={"Authorization": "Bearer not-a-real-token"}
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_protected_endpoint_accepts_valid_token(client):
    await client.post("/api/v1/auth/register", json=REGISTER_PAYLOAD)
    login = await client.post("/api/v1/auth/login", json=REGISTER_PAYLOAD)
    access_token = login.json()["access_token"]

    response = await client.get(
        "/api/v1/users/me", headers={"Authorization": f"Bearer {access_token}"}
    )
    assert response.status_code == 200
    assert response.json()["email"] == REGISTER_PAYLOAD["email"]


@pytest.mark.asyncio
async def test_refresh_without_csrf_header_is_rejected(client):
    await client.post("/api/v1/auth/register", json=REGISTER_PAYLOAD)
    await client.post("/api/v1/auth/login", json=REGISTER_PAYLOAD)

    response = await client.post("/api/v1/auth/refresh")
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "csrf_invalid"


@pytest.mark.asyncio
async def test_refresh_rotates_token_and_issues_new_access_token(client):
    await client.post("/api/v1/auth/register", json=REGISTER_PAYLOAD)
    await client.post("/api/v1/auth/login", json=REGISTER_PAYLOAD)
    old_refresh = client.cookies.get("voxmind_refresh")

    response = await client.post("/api/v1/auth/refresh", headers=_csrf_headers(client))
    assert response.status_code == 200
    new_refresh = client.cookies.get("voxmind_refresh")
    assert new_refresh != old_refresh


@pytest.mark.asyncio
async def test_concurrent_refresh_race_does_not_revoke_family(client):
    """Two requests racing to rotate the *same* still-valid token: the first
    wins and rotates; the second must get a plain 401 (retry-with-new-token),
    and - critically - must NOT revoke the session family. The first
    request's new token must keep working afterward."""
    await client.post("/api/v1/auth/register", json=REGISTER_PAYLOAD)
    await client.post("/api/v1/auth/login", json=REGISTER_PAYLOAD)
    original_refresh = client.cookies.get("voxmind_refresh")

    # CSRF is double-submit against whatever csrf cookie is *currently* set,
    # which rotates alongside the refresh token - so it must be re-read
    # before each call, exactly as a real frontend would read it fresh from
    # document.cookie rather than caching a stale value.
    first = await client.post(
        "/api/v1/auth/refresh",
        cookies={"voxmind_refresh": original_refresh},
        headers=_csrf_headers(client),
    )
    assert first.status_code == 200
    rotated_refresh = client.cookies.get("voxmind_refresh")

    # Simulate a racing duplicate request that presents the now-already-rotated
    # original token, within the grace window. It must be rejected using the
    # CSRF token that is valid *now* (post-rotation) - an attacker replaying
    # the old refresh token doesn't have a stale CSRF problem in real life,
    # so the test isolates the refresh-token-reuse behavior from CSRF.
    race = await client.post(
        "/api/v1/auth/refresh",
        cookies={"voxmind_refresh": original_refresh},
        headers=_csrf_headers(client),
    )
    assert race.status_code == 401
    assert race.json()["error"]["code"] == "invalid_credentials"

    # The legitimate rotated token must still work - the family was not revoked.
    still_valid = await client.post(
        "/api/v1/auth/refresh",
        cookies={"voxmind_refresh": rotated_refresh},
        headers=_csrf_headers(client),
    )
    assert still_valid.status_code == 200


@pytest.mark.asyncio
async def test_refresh_reuse_outside_grace_window_revokes_entire_family(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "REFRESH_REUSE_GRACE_SECONDS", 0)

    await client.post("/api/v1/auth/register", json=REGISTER_PAYLOAD)
    await client.post("/api/v1/auth/login", json=REGISTER_PAYLOAD)
    original_refresh = client.cookies.get("voxmind_refresh")

    first = await client.post(
        "/api/v1/auth/refresh",
        cookies={"voxmind_refresh": original_refresh},
        headers=_csrf_headers(client),
    )
    assert first.status_code == 200
    rotated_refresh = client.cookies.get("voxmind_refresh")

    reuse = await client.post(
        "/api/v1/auth/refresh",
        cookies={"voxmind_refresh": original_refresh},
        headers=_csrf_headers(client),
    )
    assert reuse.status_code == 401
    assert reuse.json()["error"]["code"] == "refresh_reuse_detected"

    # The entire family - including the token that was legitimately rotated
    # to - must now be revoked as a theft response.
    now_dead = await client.post(
        "/api/v1/auth/refresh",
        cookies={"voxmind_refresh": rotated_refresh},
        headers=_csrf_headers(client),
    )
    assert now_dead.status_code == 401


@pytest.mark.asyncio
async def test_logout_revokes_current_session_only(client):
    await client.post("/api/v1/auth/register", json=REGISTER_PAYLOAD)
    await client.post("/api/v1/auth/login", json=REGISTER_PAYLOAD)
    session1_refresh = client.cookies.get("voxmind_refresh")
    csrf1 = client.cookies.get("voxmind_csrf")

    # A second, independent login (second "device") for the same user.
    client.cookies.clear()
    await client.post("/api/v1/auth/login", json=REGISTER_PAYLOAD)
    session2_refresh = client.cookies.get("voxmind_refresh")
    csrf2 = client.cookies.get("voxmind_csrf")

    logout_response = await client.post(
        "/api/v1/auth/logout",
        cookies={"voxmind_refresh": session1_refresh, "voxmind_csrf": csrf1},
        headers={"X-CSRF-Token": csrf1},
    )
    assert logout_response.status_code == 204

    dead = await client.post(
        "/api/v1/auth/refresh",
        cookies={"voxmind_refresh": session1_refresh, "voxmind_csrf": csrf1},
        headers={"X-CSRF-Token": csrf1},
    )
    assert dead.status_code == 401

    still_alive = await client.post(
        "/api/v1/auth/refresh",
        cookies={"voxmind_refresh": session2_refresh, "voxmind_csrf": csrf2},
        headers={"X-CSRF-Token": csrf2},
    )
    assert still_alive.status_code == 200


@pytest.mark.asyncio
async def test_logout_everywhere_revokes_every_session(client):
    await client.post("/api/v1/auth/register", json=REGISTER_PAYLOAD)
    login1 = await client.post("/api/v1/auth/login", json=REGISTER_PAYLOAD)
    access_token = login1.json()["access_token"]
    session1_refresh = client.cookies.get("voxmind_refresh")
    csrf1 = client.cookies.get("voxmind_csrf")

    client.cookies.clear()
    await client.post("/api/v1/auth/login", json=REGISTER_PAYLOAD)
    session2_refresh = client.cookies.get("voxmind_refresh")
    csrf2 = client.cookies.get("voxmind_csrf")

    response = await client.post(
        "/api/v1/auth/logout-everywhere",
        cookies={"voxmind_csrf": csrf1},
        headers={"Authorization": f"Bearer {access_token}", "X-CSRF-Token": csrf1},
    )
    assert response.status_code == 204

    for refresh_token, csrf in ((session1_refresh, csrf1), (session2_refresh, csrf2)):
        dead = await client.post(
            "/api/v1/auth/refresh",
            cookies={"voxmind_refresh": refresh_token, "voxmind_csrf": csrf},
            headers={"X-CSRF-Token": csrf},
        )
        assert dead.status_code == 401
