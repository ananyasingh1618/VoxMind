# ADR 0016: a real numpy version conflict that only a fresh Python 3.11 resolve could ever surface

## The chain of failures, in the order they were actually found

Building the API image for the first time (`docker/api.Dockerfile`'s `pip install --no-cache-dir -e .`, on the `python:3.11-slim` base image) failed with:

```
RuntimeError: Cannot install on Python version 3.11.16; only versions >=3.7,<3.11 are supported.
```

This is `numba==0.56.4`'s own `setup.py` guard - pip's resolver had backtracked numba from the modern release this project's real `.venv` already uses (`0.67.0`) all the way down to `0.56.4`, an old release with no Python-3.11 wheel at all, while searching for *some* numba version whose own numpy requirement would reconcile the rest of the graph. Reproduced directly (not just in Docker) with a real local Python 3.11 interpreter (`/opt/homebrew/bin/python3.11`) to iterate faster than a full image rebuild per attempt.

Pinning `numba` alone just moved the same symptom onto a different package (`pip is looking at multiple versions of lightning...`, then `resolution-too-deep`). Pinning several likely culprits together moved it again, onto `transformers`. The real, root cause only became visible once pip was given the actual dependency graph and asked outright:

```
ERROR: Cannot install faster-whisper, funasr, librosa, matplotlib, pyannote.audio and voxmind-api
because these package versions have conflicting dependencies.

The conflict is caused by:
    funasr 1.4.14 depends on numpy<2
    numba 0.67.0 depends on numpy<2.6 and >=1.22
    pyannote-core 6.0.1 depends on numpy>=2.0
```

`funasr` (this project's real emotion2vec+ dependency) and `pyannote-core` (a transitive dependency of `pyannote.audio`, this project's real diarization dependency) declare **mutually exclusive** numpy floors. No version of numba, lightning, or transformers could ever resolve this - the earlier failures were pip exhaustively searching a space with no valid solution in it.

## Why this was never caught before now

This project's local development `.venv` has always run **Python 3.13**, and no session before this one ever had a working Docker daemon (`python:3.11-slim`, matching this project's own CI `python-version: "3.11"`, is the *only* place a Python 3.11 dependency resolution had ever actually been attempted). The local `.venv` genuinely has numpy `1.26.4` installed - satisfying `funasr`, not `pyannote-core` - but got there through incremental installs over this project's real history (funasr was added later, after pyannote/numpy were already settled), never through one fresh, from-scratch resolve of the whole graph at once. `pip check` on that real `.venv` has always reported `pyannote-core 6.0.1 has requirement numpy>=2.0, but you have numpy 1.26.4` as a metadata warning - previously noted and accepted as harmless (see the emotion2vec+ integration work), never as a install-blocking conflict, because nothing had ever forced pip to resolve the graph fresh since. Real GitHub Actions CI (`python-version: "3.11"`) would very likely hit the exact same failure the first time it actually ran - this project has never had `gh`/network access to execute it for real in any session, so that was never observed either.

## Why numpy `1.26.4`, not `2.x`, is correct

`pyannote-core`'s `numpy>=2.0` floor is a declared-but-unused ceiling, not an actual runtime requirement - proven directly, not assumed: every real diarization test in this project's history (including this same session's own real gated-model verification, see [DECISIONS/0012](0012-hf-groq-real-credential-verification.md) and [0013](0013-pyannote-diarizeoutput-real-success.md)) has run correctly against numpy `1.26.4`. A real, isolated Python 3.11 interpreter with this exact combination installed was used to `import` every affected library together - `numpy`, `scipy`, `numba`, `librosa`, `pyannote.audio`, `funasr`, `faster_whisper`, `torch`, `transformers`, and the core FastAPI/SQLAlchemy/Celery/Redis/boto3 stack - all real-imported successfully, with `pip check` reporting only the same two already-known, harmless metadata mismatches (`pyannote-core`, `pyannote-metrics`) and nothing else.

## The fix

`apps/api/docker-constraints.txt`: a full, pinned freeze of this project's real, working local dependency set (minus `audioop-lts`, a Python-3.13-only stdlib backport that doesn't exist for 3.11 and isn't needed there - the real stdlib `audioop` module still exists on 3.11). `docker/api.Dockerfile` now installs it with `pip install --no-deps -r docker-constraints.txt`, deliberately skipping pip's dependency resolution entirely (the entire point - there is no resolution pip could ever find, because none exists at the metadata level), followed by `pip install --no-deps -e .` for the local package itself.

No application code changed. No dependency was "blindly upgraded" - every pinned version is the exact one this project's own real, tested, working `.venv` already uses.

## Verification

Reproduced and fixed against a real, local Python 3.11 interpreter first (fast iteration), then verified against the real target: a full rebuild of the actual `docker/api.Dockerfile` image via `docker compose build`, on the real Docker Desktop VM (`linux/arm64`) - see the session's final validation report for the exact build result.

## Also applied to CI

`.github/workflows/ci.yml`'s backend job also targets Python 3.11 and also ran a plain `pip install -e ".[dev]"` - the exact vulnerable pattern. It was updated to the same two-step install (CPU-only torch first, then `docker-constraints.txt` with `--no-deps`). This repository has no `gh`/network access to execute the real GitHub Actions workflow in this environment, so unlike the Docker build above, this specific fix has **not** been independently confirmed by an actual CI run - it is the same already-proven fix applied to the same already-diagnosed root cause, on a different (linux/amd64 vs. this session's linux/arm64) architecture. Flagged honestly as reasoned-but-unverified, not claimed as confirmed.
