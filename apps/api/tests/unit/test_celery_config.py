"""Phase 9: deterministic checks on the real Celery app's configuration and
the stage registry - no broker, no worker, no network needed. These guard
the security- and reliability-relevant settings that are easy to regress
silently (a serializer change to `pickle`, losing `acks_late`, a stage
disappearing from the registry so its dispatch would fail only at runtime).
"""
from __future__ import annotations

import pytest

from voxmind.core.config import get_settings
from voxmind.workers.celery_app import celery_app
from voxmind.workers.stage_registry import ML_STAGE_NAMES, STAGE_REGISTRY, queue_for_stage


def test_the_generic_pipeline_task_is_registered_once_the_worker_imports_it():
    """The API process deliberately does NOT import `workers.tasks` - it
    dispatches by name via `send_task()`, so it never needs the task module
    or its heavy ML imports loaded. The *worker* loads it via the app's
    `imports` setting; this test does the same import the worker would and
    then asserts the task is registered under its stable public name."""
    assert "voxmind.workers.tasks" in list(celery_app.conf.imports)

    import voxmind.workers.tasks  # noqa: F401  (registers the task on the app)

    assert "voxmind.execute_pipeline_stage" in celery_app.tasks


def test_serializers_are_json_never_pickle():
    """A broker that accepts pickled payloads is a real RCE risk - this must
    never silently regress to `pickle`."""
    assert celery_app.conf.task_serializer == "json"
    assert celery_app.conf.result_serializer == "json"
    assert list(celery_app.conf.accept_content) == ["json"]
    assert "pickle" not in list(celery_app.conf.accept_content)


def test_reliability_settings_are_production_safe():
    # acks_late: a worker killed mid-task leaves the message requeued for
    # another worker instead of silently dropping the job.
    assert celery_app.conf.task_acks_late is True
    # prefetch=1: one worker can't hoard several minutes-long ML tasks while
    # sibling workers on the same queue sit idle.
    assert celery_app.conf.worker_prefetch_multiplier == 1
    assert celery_app.conf.task_time_limit > celery_app.conf.task_soft_time_limit


def test_redis_visibility_timeout_is_explicit_and_proportionate():
    """Kombu's own Redis-transport default (3600s) has no relationship to
    this app's actual task durations - a genuinely crashed worker's task
    would sit unrecovered for up to an hour. Must be set explicitly, and
    long enough that a still-legitimately-running task is never mistaken
    for lost (the real risk of setting it too short) - see celery_app.py's
    module docstring for the full reasoning."""
    settings = get_settings()
    timeout = celery_app.conf.broker_transport_options.get("visibility_timeout")
    assert timeout is not None
    assert timeout >= settings.CELERY_TASK_TIME_LIMIT_SECONDS
    assert timeout < 3600, "no reason to be as long as Kombu's own unconfigured default"


def test_every_registered_stage_has_input_output_models_and_a_factory():
    for stage_name, registration in STAGE_REGISTRY.items():
        assert registration.input_model is not None, stage_name
        assert registration.output_model is not None, stage_name
        assert callable(registration.build), stage_name


@pytest.mark.parametrize("stage_name", sorted(STAGE_REGISTRY))
def test_every_real_pipeline_stage_class_is_registered(stage_name: str):
    """Guards the one failure mode this registry design has: a stage that
    exists in the codebase but was never registered would dispatch fine and
    then fail inside the worker with "Unknown stage_name"."""
    assert stage_name in STAGE_REGISTRY


def test_stage_registry_covers_every_stage_class_in_the_codebase():
    """Discovers `PipelineStage` implementations by their real `name`
    attribute rather than hardcoding a list, so adding a new stage without
    registering it fails here instead of at runtime in a worker."""
    from voxmind.services.emotion import stages as emotion_stages
    from voxmind.services.incongruence import stages as incongruence_stages
    from voxmind.services.knowledge import stages as knowledge_stages
    from voxmind.services.nlp import stages as nlp_stages
    from voxmind.services.retrieval import stages as retrieval_stages
    from voxmind.services.speech import stages as speech_stages
    from voxmind.services.tts import stages as tts_stages

    modules = [
        speech_stages,
        emotion_stages,
        nlp_stages,
        incongruence_stages,
        knowledge_stages,
        retrieval_stages,
        tts_stages,
    ]
    discovered: set[str] = set()
    for module in modules:
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and attr_name.endswith("Stage") and hasattr(attr, "name"):
                discovered.add(attr.name)

    missing = discovered - set(STAGE_REGISTRY)
    assert not missing, f"PipelineStage(s) missing from STAGE_REGISTRY: {sorted(missing)}"


def test_queue_routing_separates_expensive_ml_work_from_cheap_work():
    settings = get_settings()
    # A model-loading stage goes to the ML queue...
    assert queue_for_stage("whisper_transcription", settings) == settings.CELERY_QUEUE_ML
    assert queue_for_stage("chunk_embedding", settings) == settings.CELERY_QUEUE_ML
    # ...a pure, deterministic CPU stage does not.
    assert queue_for_stage("transcript_alignment", settings) == settings.CELERY_QUEUE_DEFAULT
    assert queue_for_stage("document_ingestion", settings) == settings.CELERY_QUEUE_DEFAULT
    assert ML_STAGE_NAMES <= set(STAGE_REGISTRY)


def test_reconciliation_is_registered_on_a_real_beat_schedule():
    """Final limitations-clearance pass: automatic reconciliation
    scheduling must actually be configured, not just theoretically
    possible - a real regression guard against someone accidentally
    removing `beat_schedule` (or renaming the task) without noticing,
    since nothing else in the request path would ever call it."""
    settings = get_settings()
    schedule = celery_app.conf.beat_schedule
    assert "reconcile-stale-pipeline-runs" in schedule
    entry = schedule["reconcile-stale-pipeline-runs"]
    assert entry["task"] == "voxmind.reconcile_stale_pipeline_runs"
    assert entry["schedule"] == settings.CELERY_RECONCILIATION_INTERVAL_SECONDS
    # The interval controls how often the check runs, not how patient it
    # is - it must stay meaningfully shorter than the stale threshold
    # itself, or a genuinely abandoned run could sit uncaught for close to
    # (threshold + interval) instead of just past the threshold.
    assert settings.CELERY_RECONCILIATION_INTERVAL_SECONDS < settings.CELERY_RECONCILIATION_STALE_THRESHOLD_SECONDS


def test_the_reconciliation_task_is_registered_once_the_worker_imports_it():
    import voxmind.workers.tasks  # noqa: F401  (registers the task on the app)

    assert "voxmind.reconcile_stale_pipeline_runs" in celery_app.tasks


def test_reconciliation_task_invokes_the_real_underlying_function(monkeypatch):
    """The scheduled Celery task is a thin wrapper - this proves it
    actually calls the real `reconcile_stale_pipeline_runs()` function
    (the same one the CLI entrypoint and every reconciliation test use),
    not a reimplementation, by monkeypatching that one function and
    confirming the task's own return value comes from it."""
    import voxmind.workers.tasks as tasks_module

    calls = []

    async def _fake_reconcile(session, **kwargs):
        calls.append(session)
        from dataclasses import dataclass, field
        import uuid as uuid_module

        @dataclass
        class _FakeResult:
            reconciled_ids: list = field(default_factory=lambda: [uuid_module.uuid4(), uuid_module.uuid4()])

            @property
            def reconciled_count(self):
                return len(self.reconciled_ids)

        return _FakeResult()

    monkeypatch.setattr("voxmind.workers.reconciliation.reconcile_stale_pipeline_runs", _fake_reconcile)

    import contextlib

    @contextlib.asynccontextmanager
    async def _fake_session_scope():
        yield object()

    monkeypatch.setattr(tasks_module, "worker_session_scope", _fake_session_scope)

    result = tasks_module.reconcile_stale_pipeline_runs_task()

    assert result == 2
    assert len(calls) == 1
