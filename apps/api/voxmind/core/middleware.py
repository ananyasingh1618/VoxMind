"""Request correlation-ID middleware.

Binds a request_id (client-supplied via X-Request-ID, or generated) to
structlog's contextvars for the lifetime of the request, and echoes it back
in the response header so a client/log line pair can always be joined.

Implemented as a plain ASGI middleware (`__call__(scope, receive, send)`),
not a `starlette.middleware.base.BaseHTTPMiddleware` subclass. This is a
Phase 5 fix, not the original Phase 1 design: `BaseHTTPMiddleware` wraps
`receive`/`send` in a way that has a well-documented Starlette bug where
`await request.is_disconnected()` inside a streaming route hangs forever
once the client has actually disconnected (the wrapped receive channel never
resolves). This was found for real - not assumed - while building the
Phase 5 voice loop's interruption handling
(`VoiceTurnService.run()` polls `is_disconnected()` between stages to stop
real work early when a client disconnects): a live SSE request killed
client-side genuinely hung the whole worker instead of the pipeline
stopping early. Plain ASGI middleware does not have this problem because it
never intercepts `receive`.
"""
from __future__ import annotations

import uuid

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER = "X-Request-ID"
REQUEST_ID_HEADER_BYTES = REQUEST_ID_HEADER.encode("latin-1")


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        request_id = headers.get(REQUEST_ID_HEADER.lower().encode("latin-1"), b"").decode(
            "latin-1"
        ) or str(uuid.uuid4())

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        scope.setdefault("state", {})
        scope["state"]["request_id"] = request_id

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                message.setdefault("headers", [])
                message["headers"].append((REQUEST_ID_HEADER_BYTES, request_id.encode("latin-1")))
            await send(message)

        await self._app(scope, receive, send_with_request_id)
