"""Regression tests for two real bugs found during the Phase 9-readiness
audit: the API Docker image was missing `ffmpeg` (every real audio upload
would fail inside a container despite passing CI, since preprocessing.py
shells out to the real ffmpeg binary), and the CI workflow's `tests/ml`
step referenced `apps/api/.venv/bin/pytest`, a path that only exists on a
developer's machine - `actions/setup-python` never creates that directory,
so the step would fail in real GitHub Actions execution. These are plain
text/config assertions (no Docker daemon or CI runner needed here) so the
bugs can never silently regress.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_api_dockerfile_installs_ffmpeg():
    dockerfile = (REPO_ROOT / "docker" / "api.Dockerfile").read_text()
    apt_install_lines = [line for line in dockerfile.splitlines() if "apt-get install" in line]
    assert apt_install_lines, "api.Dockerfile should have an apt-get install step"
    combined = "\n".join(dockerfile.splitlines())
    assert "ffmpeg" in combined, (
        "api.Dockerfile must install ffmpeg - services/speech/preprocessing.py "
        "shells out to the real ffmpeg binary for every audio upload."
    )


def test_ci_workflow_never_references_a_dev_only_venv_path():
    ci_yaml = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    run_lines = [line for line in ci_yaml.splitlines() if line.strip().startswith("- run:")]
    assert run_lines, "ci.yml should have at least one '- run:' step"
    for line in run_lines:
        assert ".venv/bin/" not in line, (
            f"CI step references a dev-only venv path that actions/setup-python never "
            f"creates - it would fail in real GitHub Actions execution: {line.strip()!r}"
        )
