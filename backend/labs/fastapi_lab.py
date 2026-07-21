"""
FastAPI fundamentals lab — Depends, Bearer/JWT, lifespan, BackgroundTasks.

A tiny self-contained app, separate from the real backend in main.py.

Run it from backend/ (port 8001 so it never clashes with the real app):

    .venv/bin/uvicorn labs.fastapi_lab:app --reload --port 8001

Then open http://127.0.0.1:8001/docs and try the endpoints in order:

    1. GET  /items          -> Depends (shared query params)
    2. POST /auth/login     -> get a JWT   (user: alice, password: wonderland)
    3. GET  /auth/me        -> use the JWT (click "Authorize" in /docs)
    4. POST /notify         -> BackgroundTasks
    5. GET  /notify/log     -> see what the background task wrote

JS/TS analogies appear in comments as  [JS] ...
"""

# [JS] `import x from "y"`  ->  `from y import x`. No braces, no semicolons.
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import jwt  # PyJWT — signs/verifies JSON Web Tokens
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

# =============================================================================
# 1. LIFESPAN — startup/shutdown hooks for the whole app
# =============================================================================
# [JS] Like running code once when your Express server boots, and cleanup on
#      SIGTERM. FastAPI gives you ONE async function; everything before
#      `yield` runs at startup, everything after runs at shutdown.
#
# Real apps use this to open DB pools, load ML models, warm caches.
# Here we just record the boot time in app_state.

# [JS] A plain dict is like a JS object literal: {"key": value}
app_state: dict = {}


@asynccontextmanager  # decorator: wraps the function below (like an HOF in JS)
async def lifespan(app: FastAPI):
    # ---- startup (runs once, before the first request is served) ----
    app_state["started_at"] = time.time()
    print("lab: startup complete")

    yield  # <- the app serves requests while "paused" here

    # ---- shutdown (runs once, when the server stops) ----
    app_state.clear()
    print("lab: shutdown complete")


app = FastAPI(title="FastAPI fundamentals lab", lifespan=lifespan)


@app.get("/")
def root() -> dict:
    """Prove lifespan ran: uptime comes from state set at startup."""
    return {
        "app": "fastapi-lab",
        "uptime_seconds": round(time.time() - app_state["started_at"], 1),
    }


# =============================================================================
# 2. DEPENDS — dependency injection
# =============================================================================
# [JS] Think NestJS providers or Express middleware that computes a value and
#      hands it to the route. A "dependency" is just a function; FastAPI calls
#      it for you and passes the RESULT into your endpoint as an argument.
#
# Why not just call the function inside the endpoint?
#   - FastAPI reads the dependency's OWN parameters (query params here) and
#     documents + validates them automatically.
#   - Dependencies can be reused by many routes and overridden in tests.

FAKE_ITEMS = [{"name": f"item-{i}"} for i in range(1, 26)]  # 25 fake rows
# [JS] the line above is a "list comprehension":
#      Array.from({length: 25}, (_, i) => ({name: `item-${i + 1}`}))


def pagination(skip: int = 0, limit: int = 10) -> dict:
    """A dependency. Its params become query params: /items?skip=0&limit=10.

    `skip: int = 0` means: type int, default 0.
    [JS] like function pagination(skip = 0, limit = 10) — but typed & validated:
         /items?limit=abc returns a 422 error automatically.
    """
    return {"skip": skip, "limit": min(limit, 50)}  # cap page size at 50


@app.get("/items")
def list_items(page: dict = Depends(pagination)) -> dict:
    # FastAPI ran pagination() first and injected its return value as `page`.
    start, stop = page["skip"], page["skip"] + page["limit"]
    return {"total": len(FAKE_ITEMS), "items": FAKE_ITEMS[start:stop], **page}
    # [JS] **page spreads a dict into another, like {...page}


# =============================================================================
# 3. SECURITY — Bearer tokens + JWT
# =============================================================================
# Flow (same as most real-world APIs):
#   POST /auth/login  with username+password  -> server returns a signed JWT
#   Client sends  Authorization: Bearer <jwt>  on every request
#   A dependency verifies the signature + expiry and returns the user
#
# [JS] Identical to jsonwebtoken in Node: jwt.sign() / jwt.verify().

JWT_SECRET = "learning-only-secret"  # real apps: env var, long & random
JWT_ALGORITHM = "HS256"
TOKEN_TTL_MINUTES = 30

# Fake user table. Real apps store HASHED passwords (bcrypt), never plaintext.
FAKE_USERS = {"alice": {"password": "wonderland", "full_name": "Alice Liddell"}}


class LoginRequest(BaseModel):
    """Pydantic model = typed request body. [JS] like a zod schema."""

    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@app.post("/auth/login", response_model=TokenResponse)
def login(body: LoginRequest) -> TokenResponse:
    user = FAKE_USERS.get(body.username)  # None if missing ([JS] users[k] ?? null)
    if user is None or user["password"] != body.password:
        # raise = throw. FastAPI turns HTTPException into a JSON error response.
        raise HTTPException(status_code=401, detail="Wrong username or password")

    expires = datetime.now(timezone.utc) + timedelta(minutes=TOKEN_TTL_MINUTES)
    token = jwt.encode(
        {"sub": body.username, "exp": expires},  # "sub" (subject) = who this is
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )
    return TokenResponse(access_token=token)


# HTTPBearer extracts "Authorization: Bearer xxx" and 403s if the header is
# missing. It also puts the padlock icon / Authorize button into /docs.
bearer_scheme = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    """Dependency that turns a Bearer token into a user dict.

    Any endpoint that depends on this is protected: no valid token, no entry.
    """
    try:
        payload = jwt.decode(
            credentials.credentials,  # the raw token string after "Bearer "
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],  # verify signature AND expiry
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")

    username = payload["sub"]
    return {"username": username, **FAKE_USERS[username]}


@app.get("/auth/me")
def read_me(user: dict = Depends(get_current_user)) -> dict:
    """Protected route. In /docs: Authorize -> paste token -> try it out."""
    return {"username": user["username"], "full_name": user["full_name"]}


# =============================================================================
# 4. BACKGROUNDTASKS — respond now, do slow work after
# =============================================================================
# [JS] like res.json(...) then setImmediate(() => sendEmail(...)) — the client
#      gets the response immediately; the task runs after it is sent.
# Good for: sending emails, writing audit logs. NOT for heavy jobs (use a real
# queue like Celery for those — same idea as BullMQ in Node).

notification_log: list[str] = []


def send_fake_email(to: str, message: str) -> None:
    time.sleep(2)  # pretend this is a slow SMTP call
    notification_log.append(f"sent to {to}: {message}")


class NotifyRequest(BaseModel):
    to: str
    message: str


@app.post("/notify", status_code=status.HTTP_202_ACCEPTED)
def notify(body: NotifyRequest, background_tasks: BackgroundTasks) -> dict:
    # Note: we pass the function and its args SEPARATELY — no parentheses —
    # because FastAPI calls it later. [JS] queue(sendEmail, to, msg), not
    # queue(sendEmail(to, msg)) which would run it immediately.
    background_tasks.add_task(send_fake_email, body.to, body.message)
    return {"status": "queued"}  # returned instantly, before the sleep(2)


@app.get("/notify/log")
def notify_log() -> dict:
    """Poll this ~2s after POST /notify to see the background task's result."""
    return {"sent": notification_log}
