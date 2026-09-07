from __future__ import annotations

import pytest
from pydantic import BaseModel

from voxmind.workers.task_runner import CeleryTaskRunner, InProcessTaskRunner, JobStatus, build_task_runner


class _EchoInput(BaseModel):
    value: int


class _EchoOutput(BaseModel):
    doubled: int


class _EchoStage:
    name = "echo"

    async def run(self, input: _EchoInput) -> _EchoOutput:
        return _EchoOutput(doubled=input.value * 2)


class _FailingStage:
    name = "failing"

    async def run(self, input: BaseModel) -> BaseModel:
        raise RuntimeError("stage exploded")


@pytest.mark.asyncio
async def test_in_process_runner_completes_and_returns_result():
    runner = InProcessTaskRunner()
    handle = await runner.dispatch(_EchoStage(), _EchoInput(value=21))

    assert handle.status == JobStatus.COMPLETED
    status = await runner.get_status(handle.job_id)
    assert status.status == JobStatus.COMPLETED

    result = await runner.get_result(handle.job_id)
    assert result.doubled == 42


@pytest.mark.asyncio
async def test_in_process_runner_reports_failure_without_raising():
    runner = InProcessTaskRunner()
    handle = await runner.dispatch(_FailingStage(), _EchoInput(value=1))

    assert handle.status == JobStatus.FAILED
    assert "stage exploded" in handle.error

    with pytest.raises(ValueError):
        await runner.get_result(handle.job_id)


@pytest.mark.asyncio
async def test_in_process_runner_unknown_job_id_raises():
    runner = InProcessTaskRunner()
    with pytest.raises(KeyError):
        await runner.get_status(__import__("uuid").uuid4())


def test_build_task_runner_factory():
    """Phase 9: `CeleryTaskRunner` is now a real implementation (it was a
    documented `NotImplementedError` seam through Phase 8), so the factory
    builds one - but only when given the session + settings it genuinely
    needs to read/write `pipeline_runs`. Asking for one without them must
    fail loudly rather than hand back a half-constructed runner."""
    from voxmind.core.config import get_settings

    assert isinstance(build_task_runner("in_process"), InProcessTaskRunner)

    with pytest.raises(ValueError):
        build_task_runner("celery")

    runner = build_task_runner("celery", session=object(), settings=get_settings())  # type: ignore[arg-type]
    assert isinstance(runner, CeleryTaskRunner)

    with pytest.raises(ValueError):
        build_task_runner("something_else")
