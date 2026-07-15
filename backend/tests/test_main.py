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
        content = messages[0].content
        self.assertIsInstance(content, str)
        assert isinstance(content, str)
        self.assertTrue(content.startswith("You are a helpful assistant"))
        self.assertEqual(len(messages), 4)

    def test_normalize_pg_url_encodes_password(self) -> None:
        url = main._normalize_pg_url("postgres://user:p@ss@host:5432/db")
        self.assertEqual(url, "postgresql://user:p%40ss@host:5432/db")

    def test_is_conversational_query(self) -> None:
        self.assertTrue(main._is_conversational_query("hi"))
        self.assertTrue(main._is_conversational_query("Hello!"))
        self.assertTrue(main._is_conversational_query("  hey there  "))
        self.assertTrue(main._is_conversational_query("thank you"))
        self.assertTrue(main._is_conversational_query("HI MY NAME IS HARSH"))
        self.assertTrue(main._is_conversational_query("WHAT IS MY NAME"))
        self.assertTrue(main._is_conversational_query("my name is Ada"))
        self.assertFalse(main._is_conversational_query("What is the fridge password?"))
        self.assertFalse(main._is_conversational_query("hi, what is the fridge password?"))

    def test_history_to_llamaindex(self) -> None:
        msgs = main._history_to_llamaindex(
            [
                main.HistoryMessage(role="user", content="What is the fridge password?"),
                main.HistoryMessage(role="assistant", content="BANANA-42"),
            ]
        )
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0].role.value, "user")
        self.assertEqual(msgs[1].content, "BANANA-42")

    def test_format_rag_sources(self) -> None:
        from llama_index.core.schema import NodeWithScore, TextNode

        # Supabase: lower score = better. Weak 2nd hit should be dropped.
        nodes = [
            NodeWithScore(node=TextNode(text="fridge password BANANA-42"), score=0.18),
            NodeWithScore(node=TextNode(text="project notes stack"), score=0.35),
        ]
        sources = main._format_rag_sources(nodes)
        self.assertEqual(len(sources), 1)
        self.assertIn("BANANA-42", sources[0])

        # Two close matches both kept
        close = [
            NodeWithScore(node=TextNode(text="chunk a"), score=0.20),
            NodeWithScore(node=TextNode(text="chunk b"), score=0.24),
        ]
        self.assertEqual(len(main._format_rag_sources(close)), 2)


class MainRoutesTest(unittest.TestCase):
    client: TestClient

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
        # Placeholder engine entry — only need the key present for clear().
        main.rag_sessions["docs"] = None  # type: ignore[assignment]
        response = self.client.delete("/rag/session/docs")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("docs", main.rag_sessions)


if __name__ == "__main__":
    unittest.main()
