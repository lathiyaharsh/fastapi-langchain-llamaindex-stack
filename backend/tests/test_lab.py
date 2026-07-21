"""
Topic 5: TestClient patterns — tests for labs/fastapi_lab.py.

TestClient calls your FastAPI app IN MEMORY: no server, no port, no network.
[JS] like supertest in Node: request(app).get("/items").expect(200)

Run from backend/:
  .venv/bin/python -m unittest tests.test_lab -v
"""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from labs import fastapi_lab
from labs.fastapi_lab import app, get_current_user, pagination


class DependsTest(unittest.TestCase):
    def setUp(self) -> None:
        # [JS] beforeEach. A fresh client per test keeps tests independent.
        self.client = TestClient(app)

    def test_items_default_pagination(self) -> None:
        response = self.client.get("/items")
        self.assertEqual(response.status_code, 200)  # [JS] expect(x).toBe(y)
        body = response.json()
        self.assertEqual(len(body["items"]), 10)  # default limit
        self.assertEqual(body["total"], 25)

    def test_items_custom_pagination(self) -> None:
        response = self.client.get("/items", params={"skip": 20, "limit": 10})
        body = response.json()
        # only 5 items remain after skipping 20 of 25
        self.assertEqual(len(body["items"]), 5)

    def test_items_rejects_bad_types(self) -> None:
        # FastAPI validates the dependency's params: str where int expected -> 422
        response = self.client.get("/items", params={"limit": "abc"})
        self.assertEqual(response.status_code, 422)

    def test_dependency_override(self) -> None:
        # PATTERN: dependency_overrides — swap a real dependency for a fake.
        # This is THE reason dependencies beat plain function calls: in real
        # apps you override get_db / get_current_user to avoid touching
        # real databases or auth in tests.
        app.dependency_overrides[pagination] = lambda: {"skip": 0, "limit": 2}
        try:
            body = self.client.get("/items").json()
            self.assertEqual(len(body["items"]), 2)
        finally:
            app.dependency_overrides.clear()  # always undo, or tests leak


class AuthTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def _login(self) -> str:
        # Helper (leading _ = "private by convention"). Returns a fresh JWT.
        response = self.client.post(
            "/auth/login",
            json={"username": "alice", "password": "wonderland"},
        )
        self.assertEqual(response.status_code, 200)
        return response.json()["access_token"]

    def test_login_wrong_password(self) -> None:
        response = self.client.post(
            "/auth/login",
            json={"username": "alice", "password": "nope"},
        )
        self.assertEqual(response.status_code, 401)

    def test_me_without_token(self) -> None:
        response = self.client.get("/auth/me")
        self.assertEqual(response.status_code, 401)  # HTTPBearer: missing header

    def test_me_with_garbage_token(self) -> None:
        response = self.client.get(
            "/auth/me", headers={"Authorization": "Bearer not-a-jwt"}
        )
        self.assertEqual(response.status_code, 401)

    def test_full_login_flow(self) -> None:
        # PATTERN: multi-step flow — login, then use the token.
        token = self._login()
        response = self.client.get(
            "/auth/me", headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["username"], "alice")

    def test_auth_override_skips_login_entirely(self) -> None:
        # PATTERN: override auth so protected routes are testable without
        # tokens. Real projects use this constantly.
        app.dependency_overrides[get_current_user] = lambda: {
            "username": "test-user",
            "full_name": "Test User",
        }
        try:
            response = self.client.get("/auth/me")
            self.assertEqual(response.json()["username"], "test-user")
        finally:
            app.dependency_overrides.clear()


class BackgroundAndLifespanTest(unittest.TestCase):
    def test_notify_runs_background_task(self) -> None:
        fastapi_lab.notification_log.clear()
        client = TestClient(app)

        # monkeypatch-style: replace the slow sleep so tests stay fast.
        original_sleep = fastapi_lab.time.sleep
        fastapi_lab.time.sleep = lambda seconds: None
        try:
            response = client.post(
                "/notify", json={"to": "bob@example.com", "message": "hi"}
            )
            self.assertEqual(response.status_code, 202)
            # TestClient runs background tasks BEFORE returning, so the log
            # is already written here (in production it happens after the
            # response is sent).
            log = client.get("/notify/log").json()["sent"]
            self.assertEqual(log, ["sent to bob@example.com: hi"])
        finally:
            fastapi_lab.time.sleep = original_sleep
            fastapi_lab.notification_log.clear()

    def test_lifespan_runs_with_context_manager(self) -> None:
        # PATTERN: `with TestClient(app)` triggers lifespan startup/shutdown.
        # Without `with`, lifespan may not run and / would KeyError.
        with TestClient(app) as client:
            body = client.get("/").json()
            self.assertIn("uptime_seconds", body)
