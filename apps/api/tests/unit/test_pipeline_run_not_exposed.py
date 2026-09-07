"""Phase 9.1 security audit (Target 9): `pipeline_runs`/`PipelineRun` is an
internal implementation detail of the `TaskRunner` abstraction - it must
never be directly reachable over HTTP. User-facing job tracking already has
its own, separately ownership-checked tables (`AudioProcessingJob`,
`EmotionProcessingJob`, `VoiceTurn`, ...); a client can never query
`pipeline_runs` by id at all, so there is no ID-guessing/cross-user-
exposure surface to defend in the first place - this test locks that
property in against a future regression (e.g. a debug/admin endpoint added
later that exposes it without an ownership check).
"""
from __future__ import annotations

import ast
from pathlib import Path

ENDPOINTS_DIR = Path(__file__).resolve().parents[2] / "voxmind" / "api" / "v1" / "endpoints"


def test_no_endpoint_module_imports_pipeline_run():
    offending: list[str] = []
    for path in ENDPOINTS_DIR.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "pipeline_run" in node.module:
                offending.append(f"{path.name}: {node.module}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "pipeline_run" in alias.name:
                        offending.append(f"{path.name}: {alias.name}")
    assert not offending, f"pipeline_runs must stay internal-only, but found: {offending}"


def test_no_route_path_references_pipeline_runs():
    from voxmind.main import app

    offending = [route.path for route in app.routes if "pipeline" in getattr(route, "path", "").lower()]
    assert not offending, f"No HTTP route should expose pipeline_runs directly: {offending}"
