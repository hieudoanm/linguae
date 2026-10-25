"""Custom middlewares for the Linguae FastAPI server.

Includes:
- Request ID / correlation ID injection
- Rate limiting (in-memory, sliding window per client)
- CORS headers
- Request timeout
- Response compression (gzip via Starlette)
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from .config import settings


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique request ID to every request and echo it in the response."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get(settings.request_id_header) or str(
            uuid.uuid4()
        )
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory sliding-window rate limiter keyed by client IP."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._hits: dict[str, list[float]] = defaultdict(list)

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if not settings.rate_limit_enabled:
            return await call_next(request)

        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window_start = now - settings.rate_limit_window_seconds
        hits = [t for t in self._hits[client] if t > window_start]
        self._hits[client] = hits

        if len(hits) >= settings.rate_limit_requests:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate limit exceeded",
                    "retry_after_seconds": settings.rate_limit_window_seconds,
                },
                headers={"Retry-After": str(settings.rate_limit_window_seconds)},
            )

        hits.append(now)
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(settings.rate_limit_requests)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, settings.rate_limit_requests - len(hits))
        )
        return response


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Tag responses with elapsed time; actual hard timeouts are enforced by the ASGI server."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start = time.monotonic()
        response = await call_next(request)
        elapsed = time.monotonic() - start
        response.headers["X-Response-Time-Seconds"] = f"{elapsed:.4f}"
        return response


def build_cors_middleware(app: ASGIApp) -> CORSMiddleware:
    """Construct a configured CORS middleware."""
    return CORSMiddleware(
        app=app,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )


def build_gzip_middleware(app: ASGIApp) -> GZipMiddleware:
    """Construct a GZip compression middleware."""
    return GZipMiddleware(app=app, minimum_size=500)


def add_middlewares(app: ASGIApp) -> ASGIApp:
    """Apply all middlewares in the correct order (outermost first)."""
    app = build_gzip_middleware(app)
    app = build_cors_middleware(app)
    app = TimeoutMiddleware(app)
    app = RateLimitMiddleware(app)
    app = RequestIDMiddleware(app)
    return app
