# Local Development

## Prerequisites

- Python 3.11+
- Node.js 20+
- Docker (for the infrastructure services), **or** local installs of Postgres 16 + the `pgvector` extension, Redis, and MinIO if you'd rather not use Docker at all
- **ffmpeg** on your PATH (Phase 2+) - real audio preprocessing shells out to it. `brew install ffmpeg` (macOS) / `apt-get install ffmpeg` (Debian/Ubuntu).
- Speaker diarization (Phase 2, optional) needs a Hugging Face account and access token - see [docs/audio.md](audio.md). Everything else works without it; diarization just reports itself unavailable.
- Training an emotion model (Phase 3, optional) needs a real labeled dataset (RAVDESS) supplied separately - see [docs/emotion.md](emotion.md) and [ml/datasets/README.md](../ml/datasets/README.md). The emotion pipeline's code runs without it; only inference against real predictions requires a trained model.
- A real LLM provider (Phase 4, optional) needs `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` - a developer-console API key, not a Claude Pro/Max or ChatGPT Plus chat subscription. See [docs/rag.md](rag.md). Everything else in the RAG pipeline (document ingestion, hybrid retrieval, reranking) runs without one; `/ask` just honestly reports `grounding_status="unavailable"` for the final generation step.

## Three ways to run this stack - pick one

VoxMind supports three genuinely different local setups. Don't mix commands from different modes without understanding what each one does - that's how you end up debugging a "why can't the API reach Postgres" problem that's actually a mode mismatch.

### Mode A - Infrastructure-only (recommended for day-to-day backend/frontend development)

Docker runs **only** Postgres, Redis, and MinIO. You run the API and frontend directly on your machine with `uvicorn --reload` / `npm run dev`, which gives the fastest edit-reload loop and the easiest debugger access.

```bash
docker compose -f docker/docker-compose.yml up -d
```

This does **not** start the API or web containers, and it does **not** run migrations - see [Migrations](#migrations) below, which is always a separate, explicit step regardless of mode.

### Mode B - Full Docker stack

Everything - Postgres, Redis, MinIO, the API, and the web dev server - runs in containers.

```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.full.yml up --build
```

Migrations are still not automatic in this mode either (see below) - run them once the containers report healthy:

```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.full.yml exec api alembic upgrade head
```

**This mode's Dockerfiles and compose wiring have not been executed in this repository's current development environment** (no Docker daemon was available when Phase 1 was built, or during the later Phase 9-readiness audit - see the Phase 1 report for the full explanation). They follow standard patterns and should work, but treat Mode B as unverified until you've run it yourself; Mode A and the local-only path below are fully verified. A real bug (`api.Dockerfile` never installed `ffmpeg`, so every audio upload would have failed inside a container despite passing CI, which installs it as a separate step) was found by inspection during that audit and fixed - see [DECISIONS/0010](DECISIONS/0010-phase9-audit-infra-bugs.md) - but the fix itself has not been runtime-verified against a real Docker build for the same reason.

### Mode C - No Docker at all

Install Postgres 16 + `pgvector`, Redis, and MinIO locally (e.g. via Homebrew on macOS: `brew install postgresql@16 pgvector`), start them yourself, and point `DATABASE_URL`/`REDIS_URL`/`S3_*` at your local instances. This is exactly how Phase 1 was actually built and tested in this repository's development environment, and is a legitimate long-term setup if you prefer not to run Docker at all.

## Backend setup (Modes A and C)

Installs faster-whisper and pyannote.audio (which pulls in PyTorch) as of Phase 2 - a multi-GB download that takes a few minutes on first install. Both have prebuilt wheels for Python 3.11-3.13 on common platforms.

```bash
cd apps/api
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# edit .env: set JWT_SECRET_KEY (openssl rand -hex 32), confirm DATABASE_URL
```

## Migrations

Migrations are **never** run automatically by any compose file or container entrypoint, in any mode - this is a deliberate choice (see `docker/docker-compose.full.yml`'s comments) so that schema changes are always an explicit, observable, one-at-a-time action rather than something that silently happens (or silently races across replicas) on container start.

```bash
cd apps/api
alembic upgrade head
```

Run this once after the database is reachable, and again after pulling any change that includes a new migration file.

## Running the backend

```bash
cd apps/api
source .venv/bin/activate
uvicorn voxmind.main:app --reload
```

Health check: `curl http://localhost:8000/api/v1/health` should return `{"status":"ok","database":"ok"}`.

## Running the frontend

```bash
cd apps/web
npm install
cp .env.example .env   # VITE_API_BASE_URL defaults to /api/v1, proxied to :8000 by vite.config.ts
npm run dev
```

Open http://localhost:5173. Requests to `/api/*` are proxied to `http://localhost:8000` by Vite's dev server (see `apps/web/vite.config.ts`) - you don't need to configure CORS for local development.

## Tests

```bash
# Backend - runs against a real Postgres database (voxmind_test), not mocks or SQLite
cd apps/api
source .venv/bin/activate
createdb -O voxmind voxmind_test   # once
DATABASE_URL=postgresql+asyncpg://voxmind:voxmind@localhost:5432/voxmind_test alembic upgrade head   # once
pytest                              # subsequent runs read the same URL from tests/conftest.py's default

# pytest markers (see docs/audio.md, docs/emotion.md, docs/nlp.md, docs/rag.md):
#   real_model               - genuinely downloads/runs faster-whisper, Wav2Vec2, sentiment/NER/
#                              embedding/reranker/TTS models, no credentials needed
#   requires_hf_token        - genuinely runs pyannote diarization, skipped without a real
#                              HUGGINGFACE_TOKEN
#   requires_llm_credentials - genuinely calls Anthropic/OpenAI, skipped without a real
#                              ANTHROPIC_API_KEY/OPENAI_API_KEY
pytest -m "not real_model and not requires_hf_token and not requires_llm_credentials"   # fast subset only

# ml/ dataset & training pipeline tests - pure logic + real feature/metric
# computation, no Postgres needed. Run from the repo root with apps/api's venv:
cd ../..
apps/api/.venv/bin/pytest tests/ml

# Frontend
cd apps/web
npx vitest run
```

## Linting

```bash
cd apps/api && ruff check voxmind tests
cd apps/web && npx oxlint
```

## Demo walkthrough

A concrete path through the app once the stack is running (any of the three modes above), using no credentials at all - every step below works with `LLM_PROVIDER=local_dev` and no `HUGGINGFACE_TOKEN`:

1. Open the frontend, register a real account, and you land in the conversation list.
2. Click **New conversation**, then upload a short `.wav`/`.mp3` file (or record one) - a real faster-whisper transcript appears once processing finishes; click **Transcribe** if it doesn't start automatically.
3. Open the **Analysis** panel to see the real emotion prediction (`emotion2vec-tess-emodb-v1`), sentiment/NER, and the vocal-emotional incongruence signal for that turn.
4. Open the **Knowledge** panel and upload a `.txt`/`.md`/`.pdf` document, then ask a question about it - retrieval, ranking, and reranking all run for real; without an LLM key configured, the answer step honestly reports `grounding_status="unavailable"` rather than fabricating a response (set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` to see a real generated, cited answer instead).
5. Tap the microphone (**Start speaking**) to try the real-time voice loop - speak, and VoxMind transcribes, analyzes, retrieves, generates (or reports `unavailable` without a key), and speaks the answer back via real local TTS, all streamed live with per-stage latency shown.
6. Visit **Analytics** and **Evaluation** in the sidebar for the real aggregate dashboards - both start empty and fill in as you generate more real activity above.

## Common pitfalls

- **"database ... unreachable" from `/health`**: check `DATABASE_URL` matches whichever mode you're running (container hostname `postgres` in Mode B, `localhost` in Modes A/C).
- **pgvector extension missing**: if you installed Postgres yourself (Mode C) rather than via the `pgvector/pgvector:pg16` Docker image, `pgvector` must be installed separately and matched to your Postgres major version - see the Phase 1 report's "known limitations" for the exact issue this caused during development (a Homebrew `pgvector` bottle built against a different Postgres major version than initially installed).
- **CSRF 403 on refresh/logout**: the frontend must read the current `voxmind_csrf` cookie value fresh before each request, not cache it - it rotates alongside the refresh token.
