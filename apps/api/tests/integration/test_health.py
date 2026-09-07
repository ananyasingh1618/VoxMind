from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_health_endpoint_reports_ok_and_db_connectivity(client):
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"


@pytest.mark.asyncio
async def test_health_endpoint_echoes_request_id(client):
    response = await client.get("/api/v1/health", headers={"X-Request-ID": "test-request-123"})
    assert response.headers["X-Request-ID"] == "test-request-123"


@pytest.mark.asyncio
async def test_health_reports_celery_not_configured_in_default_in_process_mode(client):
    """The default TASK_RUNNER=in_process deployment doesn't need Redis or a
    worker at all - that must be a real, honest "not_configured" state, not
    a fabricated "ok"."""
    response = await client.get("/api/v1/health")
    body = response.json()
    assert body["celery_workers"] == "not_configured"


@pytest.mark.asyncio
async def test_readiness_is_true_in_default_in_process_mode(client):
    response = await client.get("/api/v1/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["checks"]["database"] == "ok"
    assert body["checks"]["celery_workers"] == "not_configured"


@pytest.mark.asyncio
async def test_readiness_reports_not_ready_when_celery_mode_has_no_worker(client, monkeypatch):
    """In celery mode, readiness must genuinely depend on a live worker
    actually being reachable - not just the broker port being open."""
    from voxmind.api.v1.endpoints import health as health_module
    from voxmind.core.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "TASK_RUNNER", "celery")
    monkeypatch.setattr(health_module, "get_settings", lambda: settings)

    response = await client.get("/api/v1/health/ready")
    body = response.json()
    # No real worker is running in this test process, so this must be
    # honestly reported as not ready, never a fabricated "ok".
    assert body["ready"] is False
    assert body["checks"]["celery_workers"] in ("no_workers", "unreachable")
