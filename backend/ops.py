"""
Phase 6 helpers — structured request logging + simple in-memory rate limits.

Wired from main.py:
  configure_logging()              → once at startup
  RequestLoggingMiddleware         → every request (latency + request_id)
  RateLimitMiddleware              → /chat* and /rag* only (HTTP 429)

No extra packages: stdlib logging + Starlette middleware.
Rate limits are per-process / in-RAM (fine for learning; use Redis in real prod
so multiple uvicorn workers share one counter).
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

# Named logger so uvicorn console shows: ai_chat.api request_id=… path=…
logger = logging.getLogger("ai_chat.api")

# Prefix match: "/chat" covers /chat and /chat/stream; "/rag" covers /rag,
# /rag-hybrid, /rag/rebuild, /rag/upload, etc.
RATE_LIMITED_PREFIXES = (
    "/chat",
    "/rag",
)


# =============================================================================
# LOGGING SETUP
# =============================================================================
def configure_logging() -> None:
    """
    Attach one StreamHandler to our logger (idempotent).

    Why not rely on uvicorn's default logger alone?
      We want a stable, grep-friendly line per request with latency_ms.
    Why propagate=False?
      Avoid double-printing the same line via the root logger.
    """
    if logger.handlers:
        return  # already configured (e.g. TestClient import + reload)
    handler = logging.StreamHandler()  # stderr → shows in the uvicorn terminal
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


# =============================================================================
# REQUEST LOGGING MIDDLEWARE
# =============================================================================
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Wrap every HTTP call:

      1. Assign / reuse a request_id
      2. Time call_next(request)  ← actual route handler
      3. Log status + latency_ms
      4. Echo request_id back as X-Request-Id (handy when debugging the UI)

    Starlette runs middleware outermost-last-added: in main.py this is added
    after rate-limit so 429 responses are still logged.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Prefer client-supplied id (BFF / curl -H) so logs line up across services.
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id  # available inside route handlers if needed
        started = time.perf_counter()  # high-res timer (not wall clock)
        try:
            response = await call_next(request)
        except Exception:
            # Unhandled crash — log with stack trace, then re-raise for FastAPI.
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.exception(
                "request_id=%s method=%s path=%s status=500 latency_ms=%.1f",
                request_id,
                request.method,
                request.url.path,
                elapsed_ms,
            )
            raise

        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-Id"] = request_id
        # Success / 4xx / handled 5xx all land here (no exception).
        logger.info(
            "request_id=%s method=%s path=%s status=%s latency_ms=%.1f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response


# =============================================================================
# RATE LIMIT MIDDLEWARE
# =============================================================================
class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window limit: max N hits per (client IP + path) per 60 seconds.

    Example with max_per_minute=60:
      hits at t=0,1,2,… → 60th OK, 61st within the same minute → 429

    Returns HTTP 429 + Retry-After (no stack traces to the client).
    """

    def __init__(self, app, *, max_per_minute: int) -> None:
        super().__init__(app)
        # Clamp so a misconfigured 0 here still means "at least 1" if this
        # middleware was mounted; main.py skips mounting when env is 0.
        self.max_per_minute = max(1, max_per_minute)
        # key → timestamps of recent hits (oldest on the left)
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def _client_key(self, request: Request) -> str:
        """
        Who is calling us?

        Behind a reverse proxy, request.client.host is often the proxy —
        prefer X-Forwarded-For (first hop = original client).
        """
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"

    def _is_limited_path(self, path: str) -> bool:
        """True for /chat, /chat/stream, /rag, /rag-hybrid, … — not /health."""
        return any(path == p or path.startswith(p + "/") for p in RATE_LIMITED_PREFIXES)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Cheap paths (/health, /, /docs) skip the counter entirely.
        if self.max_per_minute <= 0 or not self._is_limited_path(request.url.path):
            return await call_next(request)

        # Separate buckets per path so /chat spam doesn't block /rag rebuild.
        key = f"{self._client_key(request)}:{request.url.path}"
        now = time.monotonic()  # steady clock (immune to NTP jumps)
        window = self._hits[key]

        # Drop timestamps older than 60s — that's the "sliding" part.
        cutoff = now - 60.0
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= self.max_per_minute:
            # Too many recent hits — reject before hitting Groq / embeddings.
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        f"Rate limit exceeded ({self.max_per_minute}/min). "
                        "Wait a minute and try again."
                    )
                },
                headers={"Retry-After": "60"},
            )

        window.append(now)  # record this hit, then run the real handler
        return await call_next(request)
