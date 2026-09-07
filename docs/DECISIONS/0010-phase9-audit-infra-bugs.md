# ADR 0010: Two infrastructure bugs found by a pre-Phase-9 readiness audit

Neither bug was found by any automated test passing or failing - both were found by inspecting configuration files line-by-line for exactly what the Phase 9-readiness audit asked for ("any known bugs or technical debt that would materially affect production readiness," "identify any stale commands"). Both are fixed here; neither has been runtime-verified, for the same reason documented since Phase 1: no Docker daemon and no `gh`/GitHub Actions execution access exist in this development environment.

## Bug 1: `docker/api.Dockerfile` never installed `ffmpeg`

`voxmind/services/speech/preprocessing.py` shells out to the real `ffmpeg` binary (not a Python binding) for every real audio upload - this has been true since Phase 2. `docker/api.Dockerfile`'s `apt-get install` line only ever installed `libpq-dev gcc`. This means a real Docker deployment of the API would boot successfully, pass its health check, and then fail every single audio upload with `AudioProcessingError("ffmpeg not found")` - a production-breaking bug for the product's core feature.

This was never caught by CI because `.github/workflows/ci.yml`'s backend job installs `ffmpeg` as its own explicit `apt-get install` step, entirely separate from `api.Dockerfile` - the two environments diverged silently. It was also never caught by local development, since every contributor's documented setup path (`docs/development.md`'s Modes A/C) installs ffmpeg on the host directly (`brew install ffmpeg` / `apt-get install ffmpeg`), never through this Dockerfile.

**Fix**: added `ffmpeg` to `api.Dockerfile`'s `apt-get install` line.

**Verification status**: the corrected Dockerfile has NOT been built or run - no Docker daemon is available in this environment (consistent with every prior Docker-related note in this repository since Phase 1). The fix is a one-line, low-risk addition of a package that's already a hard runtime dependency; regression coverage is a plain-text assertion (`tests/unit/test_infra_config.py::test_api_dockerfile_installs_ffmpeg`) that the Dockerfile's apt-get line contains `ffmpeg`, so this specific regression can never silently reappear even without a live Docker build.

## Bug 2: `.github/workflows/ci.yml`'s `tests/ml` step referenced a path that doesn't exist in CI

The final step of the backend CI job was `cd ../.. && apps/api/.venv/bin/pytest tests/ml`. This works on a developer's machine because the documented local setup (`python3 -m venv .venv && source .venv/bin/activate`, per the README) creates a real `apps/api/.venv` directory with its own `bin/pytest`. It does not work in GitHub Actions: the job's earlier steps use `actions/setup-python` plus a plain `pip install -e ".[dev]"`, which installs `pytest` into that action's own managed interpreter, on `PATH` - it never creates anything at `apps/api/.venv`. The literal path `apps/api/.venv/bin/pytest` would not exist in a real run of this workflow, so this step should fail with "No such file or directory" every time the workflow actually executes on GitHub.

This looks like a copy-paste of the equivalent local development instruction (`README.md`'s testing section documents exactly this hardcoded-venv-path form for local use, which is correct there) into the CI config, where the assumption doesn't hold.

**Fix**: changed the step to `cd ../.. && pytest tests/ml` - plain `pytest`, resolved via `PATH` (the same interpreter every earlier step in the job already used), no hardcoded venv path.

**Verification status**: actual GitHub Actions execution is UNVERIFIED - this repository has no `gh` CLI or network access to trigger or observe a real workflow run in this environment. The fix was validated by the closest available proxy: activating this repository's own local `apps/api/.venv` (so `pytest` resolves via `PATH`, exactly mirroring how `actions/setup-python`'s installed interpreter would resolve it) and confirming `cd .. && ../.. && pytest tests/ml` run from repo root produces the same 54-passed result as the previously-used hardcoded-path invocation. This proves the corrected command is valid and produces the right result under PATH resolution; it does not prove the GitHub Actions runner environment behaves identically in every respect. A regression test (`test_ci_workflow_never_references_a_dev_only_venv_path`) asserts no `- run:` step in `ci.yml` contains a `.venv/bin/` path, so this specific mistake can't reappear even without a real CI run to catch it.
