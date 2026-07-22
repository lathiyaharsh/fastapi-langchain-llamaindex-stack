"""
Auth helpers for the real backend (Depends + Bearer JWT).

Same ideas as labs/fastapi_lab.py, but credentials come from env vars.

Endpoints that need protection add:
    user: dict = Depends(get_current_user)

Login lives in routers/auth.py (POST /auth/login, GET /auth/me).
"""

import os
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

# Learning defaults — override in backend/.env for anything public (FastAPI Cloud).
JWT_SECRET = os.getenv("JWT_SECRET", "learning-only-change-me")
JWT_ALGORITHM = "HS256"
JWT_TTL_MINUTES = int(os.getenv("JWT_TTL_MINUTES", "30"))
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "alice")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "wonderland")

bearer_scheme = HTTPBearer()


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserInfo(BaseModel):
    username: str


def issue_token(username: str) -> str:
    expires = datetime.now(timezone.utc) + timedelta(minutes=JWT_TTL_MINUTES)
    return jwt.encode(
        {"sub": username, "exp": expires},
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )


def authenticate_user(username: str, password: str):
    """Return username on success, None on failure."""
    if username == AUTH_USERNAME and password == AUTH_PASSWORD:
        return username
    return None


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    """Dependency: Authorization Bearer JWT → {"username": ...} or 401."""
    try:
        payload = jwt.decode(
            credentials.credentials,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )

    username = payload.get("sub")
    if not username or username != AUTH_USERNAME:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token subject",
        )
    return {"username": username}
