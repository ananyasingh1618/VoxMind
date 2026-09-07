# Docker

Real container images and a real full-stack lifecycle, genuinely built and run (see [DECISIONS/0015](DECISIONS/0015-docker-verification.md) and [0016](DECISIONS/0016-docker-python311-dependency-resolution.md) for what was found and fixed the first time this was actually verified with a working Docker daemon).

**Final verification pass (re-verified fresh)**: a genuine `--no-cache` rebuild of all three images (`api`, `worker`, `web`), a fresh `docker compose down -v` + `up -d` cycle (a truly empty Postgres volume, not a reused one), all five containers reaching `healthy`, all 8 Alembic migrations applying cleanly in order against the empty database, and a real end-to-end pipeline run through the containerized Celery worker - register → login → create conversation → real upload to real MinIO → dispatch → real ffmpeg preprocessing → real faster-whisper transcription (a correct transcript, "the quick brown fox jumps over the lazy dog.") → **real pyannote speaker diarization actually completing** (a real `HUGGINGFACE_TOKEN` was configured this time, unlike the previous pass where it degraded gracefully) → real alignment → `status: "completed"` persisted in the real containerized Postgres. `STORAGE_BACKEND` had to be switched from this environment's local-dev default (`local`) to `s3` for this test, per "Storage mode and the Celery worker" below - reverted back to `local` afterward, since that default is correct for this environment's normal (non-Docker) local dev workflow.

## Prerequisites

- Docker Desktop (or an equivalent daemon) actually running - `docker version` must show both a Client and a Server section, not just the CLI existing.
- Real values in `apps/api/.env` (copy from `.env.example`) - the containers read secrets from this file at *run* time via `env_file`, never baked into an image.

## Service architecture

```
Frontend (web, dev-mode Vite server)
   ↓
FastAPI API (api)
   ↓
PostgreSQL + pgvector (postgres)   Redis (redis)   MinIO (minio)
   ↑
Celery worker (worker) - same image as api, only the command differs
```

Two compose files, layered:

- `docker/docker-compose.yml` - infrastructure only (Postgres, Redis, MinIO). Local development mode: run this, then run the API/frontend directly on the host for the fastest edit/reload loop. All three infra ports are bound to `127.0.0.1` only (not every network interface) - see [DECISIONS/0015](DECISIONS/0015-docker-verification.md).
- `docker/docker-compose.full.yml` - an overlay adding `api`, `worker`, and `web` containers on top. Run both files together for the complete containerized stack.

## Build and start

```bash
# Infra only
docker compose -f docker/docker-compose.yml up -d

# Full stack (infra + api + worker + web)
docker compose -f docker/docker-compose.yml -f docker/docker-compose.full.yml up -d --build
```

`docker/api.Dockerfile` builds the `api` image (the `worker` service reuses the identical image, only its command differs - guaranteeing byte-identical application code between the two). Dependencies install via `apps/api/docker-constraints.txt` with `--no-deps` rather than a normal `pip install -e .` - see [DECISIONS/0016](DECISIONS/0016-docker-python311-dependency-resolution.md) for why a real, previously-undiscovered numpy version conflict makes that necessary, and why the CPU-only PyTorch build is installed from PyTorch's own wheel index first.

## Migrations

Migrations are **not** run automatically on container start (predictability - a schema change should be an explicit, observable step, never hidden inside a container entrypoint that could silently fail or race a sibling replica). Run once, after the containers are healthy:

```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.full.yml exec api alembic upgrade head
```

Verified against a genuinely fresh container Postgres (no prior data, no prior schema) - all revisions applied cleanly.

## Health and readiness

`GET /api/v1/health` and `GET /api/v1/health/ready` are real, not stubbed - live Postgres `SELECT 1`, live Redis ping, and a live Celery control-channel ping to the worker (distinguishing "no workers answering" from "broker unreachable" from "not configured"). The `api` container's own Docker healthcheck calls `/api/v1/health` directly. Verified live: `{"status":"ok","database":"ok","redis":"ok","celery_workers":"ok"}` with the real worker container running, and the worker's own healthcheck (`celery inspect ping`) reporting healthy independently.

## Storage mode and the Celery worker - a real constraint, not a bug

**`STORAGE_BACKEND=local` does not work with a real multi-container Celery worker.** Local filesystem storage writes to the `api` container's own disk; the `worker` container has an entirely separate filesystem, so it can never see a file the API wrote locally - proven live: a real `/process` request failed with `[Errno 2] No such file or directory` inside the worker. This is not a defect to fix - it is the genuine, expected nature of local-filesystem storage in a distributed deployment. **Use `STORAGE_BACKEND=s3` (pointed at the `minio` service) whenever `TASK_RUNNER=celery` spans more than one container.** Verified live with both `api` and `worker` configured for S3/MinIO: a real upload, real STT, real diarization, and real alignment all completed successfully end to end, with the object independently confirmed present in MinIO.

## A real end-to-end run, inside containers

Verified live, once both known issues above were fixed: register → login → create conversation → upload real audio → dispatch processing through the real `worker` container → real faster-whisper transcription → real pyannote diarization → real alignment → `status: "completed"`, persisted in the real containerized Postgres and confirmed by direct query. Total cold-start latency (Whisper + pyannote loading fresh inside the container) was about 78 seconds - real, expected model-loading cost, not a hang. Resource usage stayed comfortable throughout (worker peaked around 1.2GB of the Docker Desktop VM's 3.8GB allocation).

## Rate limiting inside Docker

`REDIS_RATE_LIMIT_URL` must point at the `redis` service name (`redis://redis:6379/2`), not `localhost` - a real bug found during this verification: the compose overlay was missing this override entirely, so the rate limiter resolved `Settings`' bare `localhost` default from inside the container (unreachable there) and, because `RATE_LIMIT_FAIL_OPEN` defaults to `true`, silently ran completely unprotected - honestly logged every time (`rate_limit_backend_unreachable`), never a hard failure, but genuinely not enforcing anything until fixed. Verified live after the fix: exactly 10 login attempts succeeded (the real `RATE_LIMIT_AUTH_PER_WINDOW` default), the 11th and 12th genuinely returned `429` against the real Docker Redis instance.

## Security

- Secrets are provided via `env_file: ../apps/api/.env` at container **run** time - never baked into an image. `.env` is git-ignored and excluded from the Docker build context by `.dockerignore` (see [DECISIONS/0015](DECISIONS/0015-docker-verification.md) - this file did not exist before this verification pass, and without it a real secret would have been copied into an image layer).
- Postgres, Redis, and MinIO are bound to `127.0.0.1` only in `docker-compose.yml` - not reachable from other devices on the same network.
- The `api`/`worker` image runs as a non-root user (`voxmind`, uid 10001).
- Container logs were inspected directly for this verification - no secret values or unexpected errors found (a targeted grep for token/key prefixes and generic "password"/"secret" substrings across all five containers' logs came back empty).

## Stopping the stack

```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.full.yml down
```

Removes the containers and the shared network; named volumes (`voxmind_postgres_data`, `voxmind_minio_data`) persist across restarts by design - add `-v` to also remove them if a genuinely fresh start is needed.

## Troubleshooting

- **`docker: command not found`** - Docker Desktop's CLI isn't always on `PATH` even when the app is running; it lives at `~/.docker/bin/docker` (macOS).
- **Postgres container won't start / "address already in use"** - a local native Postgres (e.g. via Homebrew) is very likely already bound to port 5432 on the host. Stop it first (`brew services stop postgresql@<version>`) or change the host-side port mapping.
- **`ImportError: libcublasLt.so... not found`** - this project's own fix for this (installing the CPU-only PyTorch build explicitly) should already be in place; if it recurs after a dependency change, see [DECISIONS/0016](DECISIONS/0016-docker-python311-dependency-resolution.md).
- **A real audio-processing job fails with a missing-file error inside the worker** - see "Storage mode and the Celery worker" above; switch to `STORAGE_BACKEND=s3`.

## Production deployment notes (final verification pass)

What's actually been verified is a **production-ready local/containerized system** - not a live, publicly deployed one. Concretely, none of the following exist in this repository and would be a real deployer's own responsibility, not something VoxMind ships:

- **TLS termination and a reverse proxy** - every service here talks plain HTTP on `127.0.0.1`-bound ports. A real deployment needs a reverse proxy (nginx, Caddy, a cloud load balancer) terminating TLS in front of the `api`/`web` containers, and `COOKIE_SECURE=true` + `COOKIE_SAMESITE=none` set accordingly (see [docs/security.md](security.md)).
- **A secrets-management service** - `.env`/`env_file` is what this project actually uses; a real deployment would typically use its platform's secret store (AWS Secrets Manager, Vault, Kubernetes Secrets, etc.) instead of a plain file, but that integration doesn't exist here.
- **Horizontal scaling / multi-replica orchestration** - `docker-compose.full.yml` runs exactly one of each service. Scaling the `worker` service to multiple replicas is compatible with the existing Celery architecture (queue-based work distribution, no worker-to-worker coordination needed) but has not been tested with more than one worker container running concurrently.
- **A managed database** - the compose stack runs Postgres in a container with a local named volume, not a managed service with automated backups/failover.
- **Log aggregation / metrics export** - structured JSON logs (structlog) are emitted to stdout, which any container log driver can pick up, but no log shipper or metrics exporter (Prometheus, Datadog, etc.) is wired up here.

If deploying for real: put a reverse proxy with TLS in front, move secrets to a real secret store, point `STORAGE_BACKEND=s3` at a real S3-compatible service (never `local` once more than one container needs the same files - see above), size the worker's memory allocation for real expected load (this pass's real stress testing only measured a single worker at modest concurrency - see [docs/celery.md](celery.md#memory-robustness---real-measured-findings-final-verification-pass)), and run the reconciliation CLI (`python -m voxmind.workers.reconciliation`) on a real schedule (cron, a Kubernetes CronJob, etc.) rather than never invoking it - see [DECISIONS/0017](DECISIONS/0017-pipeline-run-reconciliation.md).

## Resource considerations

This project's own development machine runs Docker Desktop with a real, constrained VM allocation (about 3.8GB of the host's 8GB, in the environment this was verified against). The full real ML pipeline (Whisper + pyannote + emotion2vec+ models loading inside containers) was verified to run comfortably within that allocation for a single real request; a deployment expecting sustained concurrent load should size the worker's memory allocation deliberately rather than assume the default is sufficient at scale.
