# ADR 0015: real Docker verification - a missing `.dockerignore` that would have baked real secrets into an image, and dev-only ports bound to every network interface

This project's Docker configuration had never been runtime-verified in any prior session - no Docker daemon had been available. With Docker Desktop genuinely installed and running for the first time, this pass ran the actual container stack rather than continuing to review the compose files by inspection alone, and found two real, fixable issues before ever building an image.

## A missing `.dockerignore` that would have copied `apps/api/.env` into image layers

No `.dockerignore` existed anywhere in this repository. Every service in `docker-compose.full.yml` (`api`, `worker`, `web`) builds with `context: ..` - the repo root - and `docker/api.Dockerfile` does `COPY apps/api/ ./`. Without a `.dockerignore` excluding it, that `COPY` would have copied `apps/api/.env` - a real file containing a real Hugging Face token and a real Groq API key - directly into the image's build context and, from there, into a layer of the built image. Docker image layers are immutable and additive: a later layer that "deletes" a file does not remove it from the earlier layer that added it, so a secret baked in this way persists in the image itself (and in a registry, if ever pushed) even if a subsequent build step tries to clean it up.

**Fix**: added `/.dockerignore` at the repo root, excluding `**/.env` (keeping `.env.example` via a negated pattern), `.venv/`, `node_modules/`, caches, local runtime state (`apps/api/.data/`, `mlruns/`), and other build-irrelevant content. Real secrets already flow into containers the correct way - `env_file: ../apps/api/.env` in `docker-compose.full.yml`, a Compose-time, run-time-only mechanism that never touches the image build - so no application or compose behavior changed, only what the build context is allowed to see.

## Dev-only infrastructure ports bound to every network interface

`docker-compose.yml`'s Postgres/Redis/MinIO port mappings used Docker's bare `"HOST:CONTAINER"` form (e.g. `"5432:5432"`), which binds the host side to `0.0.0.0` - every network interface, not just localhost. On a laptop this means any other device on the same Wi-Fi/LAN could reach this project's dev Postgres, Redis, and MinIO directly. This compose file's own header docstring already describes it as "Infrastructure-only development mode" - local-machine convenience for a developer's own `psql`/`redis-cli`/browser - never a production deployment, so there was never an intended reason for LAN-wide reachability.

**Fix**: changed all three to the explicit `"127.0.0.1:<port>:<port>"` form. The API (`8000`) and web (`5173`) ports in `docker-compose.full.yml` were deliberately left untouched - reaching the actual application from the host machine is the entire point of those two, unlike the backing infrastructure services.

## Verification

Rebuilt all three images from scratch after both fixes, then ran the complete lifecycle documented in the final validation report: fresh containers, real health/readiness, a fresh-database migration run, and real service-to-service checks (API→Postgres, API→Redis, worker→Redis, worker→Postgres, API→MinIO), a real authenticated HTTP request, and a real rate-limit check against the Docker-network Redis instance - see the session's final report for exact results and any genuine remaining limitations (this machine's Docker Desktop VM has a real, constrained memory allocation, disclosed there rather than glossed over).
