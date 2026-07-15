"""
Lightweight FastAPI unit tests (stdlib unittest — no pytest install required).

Run from backend/:
  .venv/bin/python -m unittest tests.test_main -v
"""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

import main


class MainHelpersTest(unittest.TestCase):
    def setUp(self) -> None:
        main.chat_sessions.clear()
        main.rag_sessions.clear()

    def tearDown(self) -> None:
        main.chat_sessions.clear()
        main.rag_sessions.clear()

    def test_chat_request_accepts_optional_history(self) -> None:
        body = main.ChatRequest(
            message="Next?",
            session_id="s1",
            reply_mode="concise",
            history=[
                main.HistoryMessage(role="user", content="Hi"),
                main.HistoryMessage(role="assistant", content="Hello"),
            ],
        )
        main.sync_session_history(body.session_id, body.history)
        self.assertEqual(len(main.chat_sessions["s1"]), 2)
        messages = main.build_messages("s1", "Next?", "concise")
        self.assertTrue(messages[0].content.startswith("You are a helpful assistant"))
        self.assertEqual(len(messages), 4)

    def test_normalize_pg_url_encodes_password(self) -> None:
        url = main._normalize_pg_url("postgres://user:p@ss@host:5432/db")
        self.assertEqual(url, "postgresql://user:p%40ss@host:5432/db")


class MainRoutesTest(unittest.TestCase):
    def setUp(self) -> None:
        main.chat_sessions.clear()
        main.rag_sessions.clear()
        self.client = TestClient(main.app)

    def tearDown(self) -> None:
        main.chat_sessions.clear()
        main.rag_sessions.clear()

    def test_health(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("groq_key_configured", data)

    def test_clear_chat_session(self) -> None:
        main.chat_sessions["abc"] = []
        response = self.client.delete("/chat/session/abc")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["cleared"], "abc")
        self.assertNotIn("abc", main.chat_sessions)

    def test_clear_rag_session(self) -> None:
        main.rag_sessions["docs"] = object()  # type: ignore[assignment]
        response = self.client.delete("/rag/session/docs")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("docs", main.rag_sessions)


if __name__ == "__main__":
    unittest.main()
