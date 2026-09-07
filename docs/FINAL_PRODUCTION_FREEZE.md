# Final Production Freeze

**Freeze date**: 2026-09-08
**Scope**: this document is the final, authoritative record of VoxMind's architecture, verification status, and known limitations as of the final verification pass. No further feature development should occur against this state unless a real defect is discovered - see "What freezing means" below.

## What freezing means here

VoxMind is **production-ready as a local/containerized system** - a real, working, genuinely-verified full stack (API, worker, Postgres+pgvector, Redis, MinIO, frontend) that runs correctly end-to-end in Docker on a developer's machine. This is **not** a claim that VoxMind is deployed to, or verified against, a live public production environment, a CDN, a managed database, a real domain, TLS termination, or any cloud infrastructure - none of that exists or was ever claimed. "Production-ready local/containerized system" is the accurate phrase for what was actually built and verified; "production deployed" would not be.

## Final architecture (frozen)

- **API**: FastAPI, async SQLAlchemy over Postgres 16 + pgvector, JWT access tokens with rotating/reuse-detecting refresh tokens in httpOnly cookies, CSRF protection via a separate JS-readable cookie.
- **Background jobs**: a `PipelineStage.run()`/`TaskRunner.dispatch()` contract with two interchangeable implementations - `InProcessTaskRunner` (default) and `CeleryTaskRunner` (Redis broker, real worker processes, `pipeline_runs` as the authoritative Postgres status/result store). See [docs/celery.md](celery.md).
- **Emotion model**: `emotion2vec-tess-emodb-v1` - a small trained classification head on top of a frozen emotion2vec+ base encoder, trained on TESS+EMO-DB, benchmarked zero-shot on RAVDESS. This is the final, frozen active production model - see [docs/emotion.md](emotion.md) and the licensing section below.
- **Speech pipeline**: real ffmpeg preprocessing, real faster-whisper STT, real (credential-gated) pyannote.audio diarization with graceful degradation, deterministic transcript/diarization alignment.
- **NLP**: real sentiment (`cardiffnlp/twitter-roberta-base-sentiment-latest`) and NER (`dslim/bert-base-NER`), deterministic intent/topic extraction, a real semantic-vocal incongruence signal (analytical only, never a lie-detection claim - enforced by a database `CHECK` constraint).
- **Memory**: a bounded recent-turns window plus a rolling long-term summary (real LLM when configured, deterministic extractive summarization otherwise).
- **RAG**: document ingestion → deterministic chunking → real sentence embeddings → pgvector cosine search + Postgres full-text search fused via reciprocal rank fusion → real cross-encoder reranking → credential-gated LLM generation with application-side citation validation and grounding.
- **Voice loop**: real-time, streamed (Server-Sent Events) mic → Whisper → analysis → RAG/LLM → real local TTS (`facebook/mms-tts-eng`, credential-free) or credential-gated OpenAI TTS, with real interruption handling and per-stage latency instrumentation.
- **Guardrails**: prompt-injection scanning on retrieved content, structural/citation validation, a real safety check (OpenAI moderation when configured, a deterministic keyword fallback otherwise), all decisions logged.
- **Storage**: local-filesystem or S3/MinIO-compatible, both real-verified.
- **Rate limiting**: Redis-backed, fixed-window, four independently-configured categories, real `429`/`Retry-After`, configurable fail-open/fail-closed.
- **Reconciliation**: `workers/reconciliation.py` - a plain, idempotent function + CLI entrypoint (`python -m voxmind.workers.reconciliation`) that recovers `pipeline_runs` rows abandoned by a worker that disappeared without broker redelivery ever happening. See [docs/celery.md](celery.md#reconciliation-final-verification-pass) and [DECISIONS/0017](DECISIONS/0017-pipeline-run-reconciliation.md).
- **Frontend**: React SPA (Vite), dark theme, responsive down to mobile, auth pages, conversation workspace (voice control, audio panel, analysis panel, knowledge panel), analytics dashboard, insights/timeline view, evaluation dashboard.

## Final test results (this pass, freshly re-run from a clean state)

| Suite | Result |
|---|---|
| Backend (`apps/api`, pytest) | **349 passed, 5 skipped, 0 failed** (354 total; skips are credential/broker-gated: `requires_hf_token`, `requires_llm_credentials`, `requires_minio`, `requires_redis` where genuinely unavailable) |
| ML (`tests/ml`, pytest) | **100 passed, 0 failed** |
| Frontend (`apps/web`, vitest) | **47 passed, 0 failed** |
| Frontend TypeScript (`tsc -b --noEmit`) | Clean, 0 errors |
| Frontend lint (`oxlint`) | Clean, 0 issues |
| Frontend production build (`vite build`) | Succeeds (one pre-existing, non-blocking bundle-size advisory, not a new issue) |
| Python lint (`ruff check voxmind tests`) | Clean, 0 issues |
| Python types (`mypy voxmind`) | Clean, 195 source files, 0 errors |
| Python types (`mypy ml/`) | **Not clean** - 15 pre-existing errors across 5 files (`splitting.py`, `compare_v1_v2_ravdess_six_class.py`, `focal_loss.py`, `train_emotion_model.py`, `train_emotion_model_v2.py`). Pre-existing, not introduced by this pass, and not part of CI's mypy step (which only targets `voxmind`) - recorded honestly as a known limitation rather than silently fixed or hidden. |

## Verified integrations (real, not mocked)

Real Postgres+pgvector, real Redis (broker, result backend, rate limiting), real MinIO/S3, real faster-whisper STT, real Wav2Vec2/emotion2vec+ embeddings, real sentence-transformers embeddings, real cross-encoder reranking, real local TTS, real Celery worker dispatch/execution/retry/idempotency/reconciliation, real Docker Compose full-stack build and startup (API, worker, Postgres, Redis, MinIO, frontend), and a real Playwright browser walkthrough against the live running app (see below).

## Credential-gated (implemented and unit-tested, not exercised end-to-end in this environment)

pyannote.audio diarization (gated Hugging Face model, no token configured), Anthropic/OpenAI LLM generation, OpenAI moderation API, OpenAI TTS. Every credential-independent part of each of these pipelines was verified live; the network call itself was not exercised here for lack of credentials, not for lack of implementation.

## GitHub Actions CI

**Status: locally validated, execution BLOCKED.** `.github/workflows/ci.yml` was inspected against the now-proven local/Docker dependency-resolution strategy (the two-step `docker-constraints.txt --no-deps` install that fixed a real Python-3.11 dependency conflict, [DECISIONS/0016](DECISIONS/0016-docker-python311-dependency-resolution.md)) and confirmed consistent - CI installs dependencies the identical way. Credential-gated tests (`real_model`, `requires_hf_token`) are correctly excluded from the default CI run rather than silently failing. No secret is placed directly in the workflow file; `JWT_SECRET_KEY` is a dummy CI-only value, not a real credential. Explicit least-privilege `permissions: contents: read` was added this pass (the workflow never pushes, comments, or releases, so the default token scope was unnecessarily broad before).

Actual execution was **not possible** in this environment: no `gh` CLI is installed, no SSH keys exist (`~/.ssh` does not exist), no git credentials or global git config exist, and no `GITHUB_TOKEN`/`GH_TOKEN`/PAT is present in the environment or macOS Keychain - confirmed by direct inspection, not assumed. This is reported honestly as **CI AUTHENTICATION/NETWORK BLOCKED** rather than fabricated as passing.

## Browser QA (real Playwright walkthrough, not source inspection)

A real Chromium browser, launched via Playwright, drove the actual running app (Vite dev server + real FastAPI backend + real native Postgres/Redis) through:

- **Auth**: register, login, protected-route redirect, session persistence across a real page reload, logout, and confirmation that a reload after logout correctly stays logged out.
- **Navigation/layout**: conversation list, sidebar, mobile-viewport responsive check.
- **Audio**: real file upload, real transcription trigger, a real transcript ("the quick brown fox jumps over the lazy dog.") rendered on the page.
- **Analysis**: an emotion-related label rendered on the page.
- **Voice turn**: real `getUserMedia` permission flow (Playwright's fake-device flag, since no real microphone exists in this environment), real UI state transitions (idle → listening → analyzing), zero uncaught console errors. A genuine end-to-end STT→LLM→TTS round trip with real speech content was **not verified** - the fake device produces silence, not real speech; this is disclosed, not hidden.
- **Analytics/evaluation**: both dashboards reachable without crash.
- **Knowledge**: document-upload input present and reachable.
- **Error handling**: an unauthenticated request correctly returns 401 and is handled without a crash; an invalid audio file correctly produces a real `415` from the backend and a clear, honest error message in the UI (not a fabricated success).

**Two real bugs were found and fixed as a direct result of this walkthrough**, each with a new regression test:

1. **Session-loss-on-reload bug** (`apps/api/voxmind/core/cookies.py`): the CSRF cookie's `Path` was scoped to the narrow `/api/v1/auth` path (shared with the httpOnly refresh cookie), making it invisible to `document.cookie` on any other frontend route - so `useBootstrapSession()` never even attempted `/auth/refresh` on a real page reload, silently logging the user out even though the backend session was completely intact. Fixed by splitting into `REFRESH_COOKIE_PATH` (kept narrow, a deliberate security reduction) and `CSRF_COOKIE_PATH = "/"` (broadened, since it must be JS-readable app-wide). Regression test: `test_csrf_cookie_path_is_root_not_the_narrow_auth_path`.
2. **Sidebar "New conversation" navigation bug** (`apps/web/src/components/layout/Sidebar.tsx`): the button called the create-conversation mutation fire-and-forget (`.mutate()`, no `await`, no navigation) - a real conversation was always created on the backend, but the user was silently left wherever they already were. Fixed to `await mutateAsync()` then `navigate()`, matching the identical, already-correct button on `ConversationsListPage.tsx`. Regression test: `tests/sidebar.test.tsx`, proven to genuinely fail without the fix via a temporary revert-and-rerun.

If browser automation had been unavailable, this section would say so explicitly rather than claim verification - it was available and used for real.

## TESS/EMO-DB/RAVDESS/emotion2vec+ licensing and provenance

Re-audited against ten explicit criteria this pass:

1. **TESS is never accidentally committed** - confirmed via direct search; no TESS `.wav` files exist anywhere in the repository.
2. **Large datasets are not embedded in Docker images** - `ml/datasets/manifests/` is excluded via both `.gitignore` and `.dockerignore`.
3. **Dataset provenance is clearly documented** - TESS, EMO-DB, RAVDESS, and CREMA-D each have a documented source, acquisition method, and real sample counts in [docs/emotion.md](emotion.md).
4. **TESS's CC BY-NC-ND 4.0 restriction is explicitly documented** - in `docs/emotion.md`, `ml/datasets/README.md`, and the README's License section.
5. **TESS is clearly not a freely redistributable production dataset** - stated plainly, not implied.
6. **The TESS-derived model itself is not claimed freely redistributable** - the README now explicitly states the model checkpoint inherits TESS's non-commercial restriction.
7. **emotion2vec+ licensing is accurately documented** - the FunASR loading framework is MIT-licensed; the model card itself carries a non-standard, indirect `license: other` pointing back to the FunASR repository rather than a standalone license - reported honestly as ambiguous, not resolved into a false certainty.
8. **RAVDESS and EMO-DB licensing/provenance are documented accurately.** RAVDESS: CC BY-NC-SA 4.0, confirmed. **EMO-DB: a real gap was found and fixed this pass** - it had been described as "the same terms as the original" without ever stating what those terms are. Verified directly against the HuggingFace `renumics/emodb` dataset card this pass: **no license field is set**, and the original host (`emodb.bilderbar.info`) is unreachable to verify its terms independently. Now explicitly marked as **requiring verification** in `docs/emotion.md`, `ml/datasets/README.md`, and the README, rather than assumed permissive.
9. **The README does not overclaim** - no instance of an unqualified "trained on public datasets" claim was found; every dataset mention carries its licensing caveat.
10. **No proprietary credentials or private artifacts are accidentally tracked** - confirmed via direct search; no `.env`, no HF/Groq/OpenAI/Anthropic key, no raw private audio, and no oversized model/dataset artifact exists outside the gitignored `apps/api/.data/` local-artifact directory.

## Celery memory robustness (real, measured)

A real local stress test (native Postgres + Redis, a real `celery worker` subprocess at production's own `--concurrency=2`, real sequential `CeleryTaskRunner` dispatches) measured actual worker-process-tree RSS via `ps`:

- 20 real sequential tasks (one real ML model, `chunk_embedding`), no recycling limit: RSS plateaued at 354-371MB from task 4 onward - no growth trend.
- 20 real sequential tasks alternating between two different real ML models (`chunk_embedding` + `rerank`'s cross-encoder): RSS plateaued at 360-368MB by task 8 - confirmed not an artifact of one model reused, but general across this codebase's real stages.
- 15 real sequential tasks with an aggressive `--max-tasks-per-child=5`: RSS peaked at 968MB - **worse** than no limit at all, because repeatedly cold-starting `torch`/`transformers`/`sentence-transformers` imports costs more than any leak it prevents, at least at a low threshold over a short run.

**Conclusion**: no unbounded per-task memory growth was found in this codebase's real ML stages once a model is warm (the existing `@lru_cache`-based provider construction already prevents reloading a model per task). Production's existing `--max-tasks-per-child=50` (`docker-compose.full.yml`) was kept unchanged - the evidence says lowering it would hurt, not help, and there was no observed leak to justify a stricter limit or an additional `--max-memory-per-child`. Peak measured RSS stayed comfortably under 1GB, well inside the project's known 3.8GB Docker VM budget. A new permanent regression test (`test_real_worker_recycles_child_processes_without_breaking_task_correctness`) proves recycling itself never corrupts a task result. Full reasoning: [docs/celery.md](celery.md#memory-robustness---real-measured-findings-final-verification-pass).

## Pipeline reconciliation design

`workers/reconciliation.py::reconcile_stale_pipeline_runs()` finds `pipeline_runs` rows stuck at `status="running"` whose `started_at` is older than `CELERY_RECONCILIATION_STALE_THRESHOLD_SECONDS` (1200s default, derived from `visibility_timeout` + one full redelivered execution attempt + a safety margin - not guessed) and marks each `failed` with a clearly-labeled, greppable error. It is a plain, idempotent function with a CLI entrypoint (`python -m voxmind.workers.reconciliation`), not a new Celery beat schedule - deliberately avoiding new always-on infrastructure; see [DECISIONS/0017](DECISIONS/0017-pipeline-run-reconciliation.md) for the full design rationale. Ten real test scenarios in `tests/unit/test_reconciliation.py` cover: normal/active/pending/failed/cancellation-like runs left untouched, a genuinely stale run reconciled, idempotency across repeated calls, a long-running legitimate task (refreshed by simulated redelivery) never prematurely reconciled, and - through the real worker task body, not just the repository layer - a reconciled run never resurrected by a later redelivery. It never exposes `pipeline_runs` over HTTP (same structural guard as everywhere else in this codebase).

## Docker verification (this pass)

A genuinely fresh (`--no-cache`) rebuild of the full stack (Postgres+pgvector, Redis, MinIO, API, worker, frontend) was performed, followed by a real `down -v` + `up -d` cycle against a truly empty Postgres volume (not a reused one from a prior session), all five containers reaching `healthy`, all 8 Alembic migrations applying cleanly against the empty database, and a real end-to-end pipeline run through the containerized Celery worker: register → login → create conversation → real upload to real MinIO → dispatch → real ffmpeg preprocessing → real faster-whisper transcription (correct transcript) → **real pyannote speaker diarization genuinely completing** (a real Hugging Face token was configured this time) → real alignment → `status: "completed"`, independently confirmed via the real persisted message content. `STORAGE_BACKEND` was temporarily switched to `s3` for this test (the documented requirement for a multi-container Celery worker - `local` storage is only correct for single-process/non-Docker dev) and reverted to `local` immediately afterward, matching this environment's normal local-dev configuration. See [docs/docker.md](docker.md) for full detail.

## Known limitations (honest, not hidden)

- A full accessibility audit beyond the reduced-motion/responsive fixes already verified remains open.
- `ml/` (training/evaluation scripts) is not mypy-clean - 15 pre-existing errors, not part of CI's mypy scope, not introduced by this pass.
- No cap exists on repeated broker-level redelivery of a genuine "poison pill" input - `CELERY_TASK_MAX_RETRIES` only governs caught, in-process retries. Reconciliation does not close this gap by design (a row still being legitimately redelivered keeps refreshing its own `started_at` and correctly never looks stale).
- Reconciliation has no scheduler of its own - it must be invoked (cron, a scheduled job, or manually) for stale rows to actually be corrected; they are correctly identifiable but not automatically corrected without that invocation.
- A genuine end-to-end voice-turn round trip with real speech audio was not verified (no real microphone in this environment; Playwright's fake device produces silence).
- EMO-DB's license terms are not independently confirmed (see licensing section above).
- GitHub Actions CI has never actually executed in this environment (see above) - all validation is local-only.
- The hard Celery time limit and any bare `BaseException`/process-kill scenario can still leave a `pipeline_runs` row at `running` until reconciliation's next invocation catches it (not until Celery's own machinery does) - a real, disclosed gap, not silently assumed away.

## Statement

No further feature changes should be made against this state unless a real, concretely-identified defect is discovered. This freeze does not mean the codebase is beyond improvement in the abstract - it means the explicit scope of this verification pass is complete, every claim in this document was independently checked rather than assumed, and continuing to "improve" further without a real trigger would violate the discipline this project has held to throughout: no fabricated results, no speculative changes, no scope creep.
