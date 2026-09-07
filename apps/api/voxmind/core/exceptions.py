"""Domain exceptions and the global exception handlers that translate them
into the API's standard error envelope: {"data": null, "error": {...}, "request_id": ...}.

Rule enforced here: no endpoint or service is allowed to let an unhandled
exception reach the client as a bare 500 traceback. Anything not explicitly
caught below is logged with a stack trace and returned as a generic,
non-leaking internal_error response.
"""
from __future__ import annotations

import structlog
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = structlog.get_logger(__name__)


class VoxMindError(Exception):
    """Base class for all domain-level errors. Never raised directly."""

    status_code: int = status.HTTP_400_BAD_REQUEST
    code: str = "error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code:
            self.code = code


class NotFoundError(VoxMindError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"


class ConflictError(VoxMindError):
    status_code = status.HTTP_409_CONFLICT
    code = "conflict"


class InvalidCredentialsError(VoxMindError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "invalid_credentials"


class SessionExpiredError(VoxMindError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "session_expired"


class ReuseDetectedError(VoxMindError):
    """Raised when a revoked refresh token is presented again (theft signal)."""

    status_code = status.HTTP_401_UNAUTHORIZED
    code = "refresh_reuse_detected"


class CsrfValidationError(VoxMindError):
    status_code = status.HTTP_403_FORBIDDEN
    code = "csrf_invalid"


class ForbiddenError(VoxMindError):
    status_code = status.HTTP_403_FORBIDDEN
    code = "forbidden"


class UnsupportedAudioFormatError(VoxMindError):
    """Raised when the uploaded file's real (sniffed) content doesn't match
    a supported audio container - never based on trusting the client's
    declared Content-Type alone."""

    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    code = "unsupported_audio_format"


class AudioTooLargeError(VoxMindError):
    status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    code = "audio_too_large"


class InvalidAudioError(VoxMindError):
    """Raised when audio passes the ingestion-time format sniff but fails
    real decoding/preprocessing - empty, corrupt, truncated, or silent
    beyond the configured duration limits."""

    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    code = "invalid_audio"


class AudioProcessingError(VoxMindError):
    """Raised when a pipeline stage (preprocessing, STT, diarization,
    alignment) fails for a reason that isn't the caller's fault - a model
    execution failure, a missing/misconfigured provider, etc. Never carries
    a raw stack trace or secret in its message."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "audio_processing_failed"


class PipelineProcessingError(VoxMindError):
    """Phase 4 equivalent of `AudioProcessingError` for the non-audio
    pipeline stages (NLP, knowledge ingestion, retrieval, LLM generation) -
    a genuine execution failure, never the caller's fault. Never carries a
    raw stack trace or secret in its message."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "pipeline_processing_failed"


class DocumentTooLargeError(VoxMindError):
    status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    code = "document_too_large"


class UnsupportedDocumentFormatError(VoxMindError):
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    code = "unsupported_document_format"


class RateLimitExceededError(VoxMindError):
    """Raised by core/rate_limit.py when a category/identity has exceeded
    its configured request budget for the current window. Carries
    `retry_after_seconds` so the response can include a real, honest
    `Retry-After` header rather than a hardcoded/omitted one."""

    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    code = "rate_limited"

    def __init__(self, message: str, *, retry_after_seconds: int) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


def _envelope(request: Request, code: str, message: str, detail: object | None = None) -> dict:
    request_id = getattr(request.state, "request_id", None)
    return {
        "data": None,
        "error": {"code": code, "message": message, "detail": detail},
        "request_id": request_id,
    }


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(VoxMindError)
    async def handle_domain_error(request: Request, exc: VoxMindError) -> JSONResponse:
        headers = None
        if isinstance(exc, RateLimitExceededError):
            headers = {"Retry-After": str(exc.retry_after_seconds)}
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(request, exc.code, exc.message),
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_envelope(request, "validation_error", "Request validation failed.", exc.errors()),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(request, "http_error", str(exc.detail)),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_exception", path=request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_envelope(request, "internal_error", "An unexpected error occurred."),
        )
