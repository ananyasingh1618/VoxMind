# Builds the FastAPI application image. The same image also runs the Celery
# worker (see docker-compose.full.yml's `worker` service, which only
# overrides the CMD) - one image, two roles, so the worker is guaranteed to
# have byte-identical application code and dependencies to the API that
# enqueues its tasks.
#
# Does NOT run migrations on container start - see docs/development.md for
# why (predictability: a fresh/updated schema should be an explicit,
# observable step, not something hidden inside a container's entrypoint that
# could silently fail or race against another replica doing the same thing).
FROM python:3.11-slim AS base

WORKDIR /app

# ffmpeg is required at runtime (services/speech/preprocessing.py shells out
# to the real ffmpeg binary for every audio upload) - found missing here
# during the Phase 9-readiness audit: CI already installed it as a separate
# step, but this image never did, so every real audio upload would have
# failed inside a container with "ffmpeg not found" despite passing CI.
# curl is used by the compose healthcheck below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

COPY apps/api/ ./
# Installed via a frozen constraints file with --no-deps, deliberately
# skipping pip's normal dependency resolution - a real, previously-
# undiscovered conflict (pyannote-core declares numpy>=2.0, funasr
# declares numpy<2 - both real project dependencies) makes a fresh resolve
# on Python 3.11 either fail outright or backtrack into ancient,
# Python-3.11-incompatible releases trying to reconcile it. The pinned set
# in docker-constraints.txt is the same combination this project's real
# local development environment has always used and every real diarization/
# emotion2vec+ test in this project has run against - see
# docs/DECISIONS/0016 for the full account, including direct proof that
# pyannote-core's stricter floor is a harmless, unmet metadata claim, not
# an actual runtime requirement.
# torch/torchaudio are installed separately, first, from PyTorch's own
# CPU-only wheel index - a second real bug found during Docker
# verification (see docs/DECISIONS/0016): the default PyPI Linux wheel for
# this pinned torch version is CUDA-enabled and eagerly preloads NVIDIA
# libraries (libcublasLt.so etc.) at import time even when no GPU is ever
# used, crashing outright on this CPU-only container with no CUDA runtime
# present. This project has always run CPU-only (WHISPER_DEVICE,
# WAV2VEC2_DEVICE, EMOTION2VEC_DEVICE, TTS_DEVICE all default to "cpu" -
# see core/config.py) - the CUDA build was never actually needed, only ever
# pulled in by a platform-generic `pip install torch==<version>`. Never a
# problem on macOS (no CUDA wheels exist for macOS at all, so local
# development never surfaced this). Installing the CPU build first means
# docker-constraints.txt's own torch/torchaudio lines are deliberately
# removed - pip would otherwise have nothing to reconcile them against,
# since these are a different distribution (`+cpu` local version) at each
# same version string.
RUN pip install --no-cache-dir --no-deps \
    "torch==2.14.0" "torchaudio==2.11.0" \
    --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --no-deps -r docker-constraints.txt \
    && pip install --no-cache-dir --no-deps -e .

# Run as a non-root user (Phase 9 hardening). Created after `pip install` so
# the install itself still runs as root, then ownership is handed over -
# the app never needs to write to its own source tree at runtime, but it
# does need a writable local-storage root when STORAGE_BACKEND=local.
RUN useradd --create-home --uid 10001 voxmind \
    && mkdir -p /app/.data/storage \
    && chown -R voxmind:voxmind /app
USER voxmind

EXPOSE 8000

CMD ["uvicorn", "voxmind.main:app", "--host", "0.0.0.0", "--port", "8000"]
