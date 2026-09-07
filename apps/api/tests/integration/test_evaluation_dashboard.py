"""Real Postgres-backed tests for the Phase 7 evaluation dashboard and the
`EvaluationRun` persistence mechanism: a type that has never been evaluated
must report `status="never_run"` (never a fabricated metric), a real
persisted run must show up as `evaluated`, and historical runs must never
be overwritten by a newer run of the same type.
"""
from __future__ import annotations

import pytest

from voxmind.repositories.evaluation_run_repository import EvaluationRunRepository

CREDENTIALS = {"email": "eval-dashboard-test@voxmind.dev", "password": "correct-horse-battery-staple"}


async def _authed_headers(client) -> dict:
    await client.post("/api/v1/auth/register", json=CREDENTIALS)
    login = await client.post("/api/v1/auth/login", json=CREDENTIALS)
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


@pytest.mark.asyncio
async def test_never_evaluated_types_report_never_run_not_fabricated(client):
    headers = await _authed_headers(client)

    response = await client.get("/api/v1/evaluations/dashboard", headers=headers)
    assert response.status_code == 200
    body = response.json()

    types_by_name = {t["evaluation_type"]: t for t in body["types"]}
    assert set(types_by_name) == {"stt", "emotion", "retrieval", "grounding", "system"}
    for entry in types_by_name.values():
        assert entry["status"] == "never_run"
        assert entry["latest"] is None


@pytest.mark.asyncio
async def test_a_real_persisted_run_shows_up_as_evaluated(client, db_session):
    headers = await _authed_headers(client)

    await EvaluationRunRepository(db_session).create(
        evaluation_type="stt",
        dataset_version="stt-eval-v1",
        model_version="faster-whisper:tiny",
        configuration={"whisper_device": "cpu"},
        sample_count=5,
        status="completed",
        metrics={"n_samples": 5, "avg_wer": 0.05, "avg_cer": 0.04},
        errors=[],
        notes=None,
    )
    await db_session.commit()

    response = await client.get("/api/v1/evaluations/dashboard", headers=headers)
    body = response.json()
    stt_entry = next(t for t in body["types"] if t["evaluation_type"] == "stt")

    assert stt_entry["status"] == "evaluated"
    assert stt_entry["latest"]["dataset_version"] == "stt-eval-v1"
    assert stt_entry["latest"]["metrics"]["avg_wer"] == 0.05
    assert stt_entry["latest"]["sample_count"] == 5

    other_entry = next(t for t in body["types"] if t["evaluation_type"] == "emotion")
    assert other_entry["status"] == "never_run"


@pytest.mark.asyncio
async def test_historical_runs_are_never_overwritten(client, db_session):
    headers = await _authed_headers(client)
    repo = EvaluationRunRepository(db_session)

    first = await repo.create(
        evaluation_type="retrieval", dataset_version="retrieval-eval-v1", model_version="v1",
        configuration={}, sample_count=6, status="completed",
        metrics={"mrr": 0.8}, errors=[], notes=None,
    )
    second = await repo.create(
        evaluation_type="retrieval", dataset_version="retrieval-eval-v1", model_version="v2",
        configuration={}, sample_count=6, status="completed",
        metrics={"mrr": 0.9}, errors=[], notes=None,
    )
    await db_session.commit()

    response = await client.get(
        "/api/v1/evaluations/runs", params={"evaluation_type": "retrieval"}, headers=headers
    )
    assert response.status_code == 200
    runs = response.json()
    run_ids = {r["id"] for r in runs}
    assert str(first.id) in run_ids
    assert str(second.id) in run_ids
    # Newest first.
    assert runs[0]["id"] == str(second.id)

    dashboard = await client.get("/api/v1/evaluations/dashboard", headers=headers)
    retrieval_entry = next(t for t in dashboard.json()["types"] if t["evaluation_type"] == "retrieval")
    assert retrieval_entry["latest"]["id"] == str(second.id)


@pytest.mark.asyncio
async def test_unknown_evaluation_type_is_rejected(client):
    headers = await _authed_headers(client)
    response = await client.get(
        "/api/v1/evaluations/runs", params={"evaluation_type": "not-a-real-type"}, headers=headers
    )
    assert response.status_code == 422
