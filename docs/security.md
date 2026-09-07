# Security

A consolidated reference for VoxMind's security posture - each control already exists and is documented in its own area of the codebase; this page exists so the full picture doesn't require reading a dozen files. See the linked doc for the full detail behind each item.

## Authentication and session security

- Passwords hashed with bcrypt; never logged, never returned in any response.
- JWT access tokens (short TTL, 15 minutes by default) plus rotating, reuse-detecting refresh tokens - reusing an already-rotated refresh token is treated as a real compromise signal and revokes the session. See [DECISIONS/0001](DECISIONS/0001-auth-token-strategy.md).
- The refresh token lives in an httpOnly cookie, scoped narrowly to the auth endpoints (`REFRESH_COOKIE_PATH`) - never readable by JavaScript, never sent to unrelated routes. A separate, deliberately non-httpOnly CSRF cookie is scoped to `/` (the whole app), since the frontend's own bootstrap logic needs to read it - see `apps/api/voxmind/core/cookies.py` for the real bug this exact split fixed during this pass's browser QA (a session-loss bug from an earlier, overly-narrow CSRF cookie path).
- `COOKIE_SECURE`/`COOKIE_SAMESITE` are `false`/`lax` for local HTTP development; a real deployment over HTTPS with the frontend and API on different subdomains **must** set `COOKIE_SECURE=true` and `COOKIE_SAMESITE=none` - documented directly in `.env.example`.
- Access token is held only in memory (a Zustand store), never in `localStorage`/`sessionStorage` - lost on every page reload by design, silently re-established via the refresh cookie.

## Secrets handling

- No secret is ever placed directly in a committed file - `.env` is gitignored (`**/.env` in `.gitignore` and `.dockerignore`), and Docker images read secrets from `env_file` at container *run* time, never baked into a build layer (`.dockerignore`'s `**/.env` exclusion, added after a real gap was found during Docker verification - see [DECISIONS/0015](DECISIONS/0015-docker-verification.md)).
- The GitHub Actions CI workflow (`.github/workflows/ci.yml`) contains no real credential - `JWT_SECRET_KEY` there is a dummy, CI-only value. Credential-gated tests (`real_model`, `requires_hf_token`, `requires_llm_credentials`) are excluded from the default CI run rather than run without the credential they need (which would either fail loudly or, worse, silently skip in a way that looks like coverage).
- This session/pass never printed, logged, or committed any real API key, token, or `.env` content - see [docs/FINAL_PRODUCTION_FREEZE.md](FINAL_PRODUCTION_FREEZE.md)'s security/secret audit.

## Broker and background-job security

- Celery's serializer is pinned to `json`, never `pickle` - a broker that can deserialize arbitrary pickled objects is a real remote-code-execution risk. Enforced by a regression test (`tests/unit/test_celery_config.py`).
- Only a small `job_id` UUID ever crosses the broker - never the real payload, never a secret (`test_large_input_never_crosses_the_broker_only_a_job_id_does`).
- `pipeline_runs` (the internal Celery task-status table) is never reachable over HTTP - structurally enforced by `tests/unit/test_pipeline_run_not_exposed.py`, which scans every endpoint module and the live FastAPI route table.

## Rate limiting (abuse protection, not a precise billing meter)

Redis-backed, fixed-window, four independently-configured categories (auth, expensive ML/LLM work, knowledge ingestion, ordinary traffic), identity keyed by authenticated user id where available and IP only for the three pre-auth endpoints. `X-Forwarded-For` is not trusted by default (no reverse proxy ships with this project's compose files) - see [docs/rate_limiting.md](rate_limiting.md) and [DECISIONS/0014](DECISIONS/0014-rate-limiting-fixed-window-dependency.md) for the full design and its one disclosed weakness (up to ~2x the configured budget across a single window boundary).

## LLM/RAG-specific defenses

- Real prompt-injection scanning filters malicious retrieved document content out before it ever reaches the model.
- Citations are validated application-side - a fabricated citation is filtered out before it reaches the client; `grounding_status` is computed by the app, never self-reported by the model.
- A real safety check (OpenAI's moderation API when configured, a deterministic keyword fallback otherwise - never skipped) can replace an unsafe answer with a refusal, and every decision is logged. See [docs/guardrails.md](guardrails.md).

## Object storage

Both the local-filesystem and S3/MinIO backends enforce ownership checks at the service layer before any read/write - a client can never address another user's object by guessing a key. See [docs/storage.md](storage.md).

## What this project does not claim

No WAF, no intrusion detection, no penetration test, no third-party security audit, no TLS termination (that's the responsibility of whatever reverse proxy/load balancer a real deployment would put in front of this - none is included here), and no secrets-management service integration (Vault, AWS Secrets Manager, etc.) - plain environment variables via `.env`/`env_file` are what this project actually uses. These are honest gaps for a portfolio-scale project, not omissions to be discovered later.
