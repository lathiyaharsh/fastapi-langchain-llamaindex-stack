"""
FastAPI tutorial lab — Part 4: Middleware (+ CORS peek).

Official docs:
  https://fastapi.tiangolo.com/tutorial/middleware/
  https://fastapi.tiangolo.com/tutorial/cors/

Run from backend/ (port 8004):

    .venv/bin/uvicorn labs.fastapi_lab_part4:app --reload --port 8004
    # open http://127.0.0.1:8004/docs
    # DevTools Network → response headers: X-Process-Time, X-Lab-Request-Id

How this maps to your REAL backend:
  RequestLoggingMiddleware  (ops.py)  ≈ timing + request_id middleware here
  RateLimitMiddleware       (ops.py)  ≈ "block some requests" middleware here
  CORSMiddleware            (main.py) ≈ CORS block at the bottom of this file

[JS] Express: app.use((req, res, next) => { ...; next(); })
     Middleware wraps EVERY route — before and after the handler.
"""

import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

app = FastAPI(title="FastAPI lab Part 4 — middleware")


# =============================================================================
# 1. FUNCTION MIDDLEWARE — tutorial style (@app.middleware("http"))
# =============================================================================
# Runs for every request. call_next(request) = "go run the route (and inner MW)".


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    # After the route finished — stamp how long it took
    response.headers["X-Process-Time"] = f"{time.perf_counter() - start:.4f}"
    return response


# =============================================================================
# 2. CLASS MIDDLEWARE — same idea as ops.RequestLoggingMiddleware
# =============================================================================


class LabRequestIdMiddleware(BaseHTTPMiddleware):
    """Assign X-Lab-Request-Id (like your real X-Request-Id)."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get("X-Lab-Request-Id") or uuid.uuid4().hex[:12]
        response = await call_next(request)
        response.headers["X-Lab-Request-Id"] = request_id
        return response


class LabBlockMiddleware(BaseHTTPMiddleware):
    """
    Tiny stand-in for RateLimitMiddleware.

    If path starts with /blocked → 429 before the route runs.
    Real app: RateLimitMiddleware counts /chat* and /rag* per IP.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.url.path.startswith("/blocked"):
            return JSONResponse(
                status_code=429,
                content={"detail": "lab block — pretend rate limit"},
            )
        return await call_next(request)


# =============================================================================
# ORDER MATTERS
# =============================================================================
# Last add_middleware = OUTERMOST on the way IN.
#
# In main.py you add RateLimit then Logging, so:
#   request  → Logging → RateLimit → route
#   response ← Logging ← RateLimit ← route
# so 429s are still logged.
#
# Here (after both adds below + the @middleware decorator which is also outer):
#   We add Block first, then RequestId → RequestId is outer among the two classes.

app.add_middleware(LabBlockMiddleware)
app.add_middleware(LabRequestIdMiddleware)

# CORS — same class your main.py uses (browser cross-origin rules)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Lab-Request-Id"],  # browser can read these
)


# =============================================================================
# ROUTES (tiny — middleware is the lesson)
# =============================================================================


@app.get("/")
def root():
    return {
        "lab": "part4-middleware",
        "try": [
            "GET /hello  → check X-Process-Time + X-Lab-Request-Id headers",
            "GET /blocked/x → 429 from LabBlockMiddleware (route never runs)",
        ],
        "real_app_map": {
            "ops.RequestLoggingMiddleware": "timing + X-Request-Id",
            "ops.RateLimitMiddleware": "429 on /chat* /rag*",
            "CORSMiddleware in main.py": "Next.js on :3000/:3001",
        },
    }


@app.get("/hello")
def hello():
    return {"message": "hello from inside the route"}


@app.get("/blocked/x")
def should_never_run():
    # Middleware returns 429 first — this body should not appear
    return {"message": "you should not see this"}
