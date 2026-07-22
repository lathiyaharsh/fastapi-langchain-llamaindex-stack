"""
Auth routes: login for a JWT, and /auth/me to verify it.

Protect mutating RAG admin routes with Depends(get_current_user) from auth.py.
"""

from fastapi import APIRouter, Depends, HTTPException

from auth import (
    LoginRequest,
    TokenResponse,
    UserInfo,
    authenticate_user,
    get_current_user,
    issue_token,
)


def create_auth_router() -> APIRouter:
    router = APIRouter(tags=["auth"])

    @router.post("/auth/login", response_model=TokenResponse)
    def login(body: LoginRequest) -> TokenResponse:
        username = authenticate_user(body.username, body.password)
        if username is None:
            raise HTTPException(status_code=401, detail="Wrong username or password")
        return TokenResponse(access_token=issue_token(username))

    @router.get("/auth/me", response_model=UserInfo)
    def me(user: dict = Depends(get_current_user)) -> UserInfo:
        return UserInfo(username=user["username"])

    return router
