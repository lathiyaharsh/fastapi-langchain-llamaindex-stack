"""
Lightweight FastAPI unit tests (stdlib unittest — no pytest install required).

Run from backend/:
  .venv/bin/python -m unittest tests.test_main -v
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import main


class MainHelpersTest(unittest.TestCase):
    def setUp(self) -> None:
        main.chat_sessions.clear()
        main.rag_sessions.clear()

    def tearDown(self) -> None:
        main.chat_sessions.clear()
        main.rag_sessions.clear()

    def test_fetch_weather_formats_response(self) -> None:
        geo_response = MagicMock()
        geo_response.raise_for_status = MagicMock()
        geo_response.json.return_value = {
            "results": [
                {
                    "name": "London",
                    "country": "United Kingdom",
                    "latitude": 51.5,
                    "longitude": -0.12,
                }
            ]
        }

        wx_response = MagicMock()
        wx_response.raise_for_status = MagicMock()
        wx_response.json.return_value = {
            "current": {
                "temperature_2m": 18.0,
                "apparent_temperature": 17.0,
                "relative_humidity_2m": 62,
                "wind_speed_10m": 12.0,
                "precipitation": 0.0,
                "weather_code": 2,
            }
        }

        client = MagicMock()
        client.get.side_effect = [geo_response, wx_response]
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)

        with patch("main.httpx.Client", return_value=client):
            result = main.fetch_weather("London")

        self.assertIn("London, United Kingdom", result)
        self.assertIn("Partly cloudy", result)
        self.assertIn("18.0°C", result)

    def test_get_weather_tool(self) -> None:
        with patch.object(main, "fetch_weather", return_value="Mumbai: Clear sky."):
            result = main.get_weather.invoke({"location": "Mumbai"})
        self.assertEqual(result, "Mumbai: Clear sky.")

    def test_invoke_chat_with_tools(self) -> None:
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        weather_result = (
            "London, United Kingdom: Partly cloudy. "
            "Temperature 18.0°C (feels like 17.0°C)."
        )
        tool_call = {
            "name": "get_weather",
            "args": {"location": "London"},
            "id": "call_1",
        }
        first = AIMessage(content="", tool_calls=[tool_call])
        second = AIMessage(content="It's partly cloudy in London at 18°C.")
        seen: list[list[object]] = []

        class FakeBound:
            def __init__(self) -> None:
                self.calls = 0

            def invoke(self, messages: list[object]) -> AIMessage:
                self.calls += 1
                seen.append(list(messages))
                if self.calls == 1:
                    return first
                return second

        fake = FakeBound()
        with (
            patch.object(main, "get_model_with_tools", return_value=fake),
            patch.object(main, "fetch_weather", return_value=weather_result),
        ):
            reply = main.invoke_chat_with_tools(
                [HumanMessage(content="What's the weather in London?")]
            )
        self.assertEqual(reply, "It's partly cloudy in London at 18°C.")
        self.assertEqual(fake.calls, 2)
        self.assertEqual(len(seen[1]), 3)
        last = seen[1][-1]
        self.assertIsInstance(last, ToolMessage)
        assert isinstance(last, ToolMessage)
        self.assertEqual(last.content, weather_result)

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
        self.assertIn("weather tool", content)
        self.assertEqual(len(messages), 4)

    def test_normalize_pg_url_encodes_password(self) -> None:
        url = main._normalize_pg_url("postgres://user:p@ss@host:5432/db")
        self.assertEqual(url, "postgresql://user:p%40ss@host:5432/db")

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

    def test_safe_upload_filename(self) -> None:
        self.assertEqual(main._safe_upload_filename("notes.md"), "notes.md")
        self.assertEqual(
            main._safe_upload_filename("../../etc/passwd.txt"), "passwd.txt"
        )
        self.assertEqual(
            main._safe_upload_filename("My Notes!.md"), "My-Notes.md"
        )
        with self.assertRaises(main.HTTPException):
            main._safe_upload_filename("image.png")

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
        self.assertIn("chat_tools", data)
        self.assertEqual(data["chat_tools"], ["get_weather"])

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

    def test_upload_inserts_only_new_document(self) -> None:
        uploaded = main.DATA_DIR / "incremental-test.md"
        uploaded.unlink(missing_ok=True)
        index = object()
        try:
            with (
                patch.object(main, "get_rag_index", return_value=index) as get_index,
                patch.object(main, "_insert_uploaded_document") as insert_document,
            ):
                response = self.client.post(
                    "/rag/upload",
                    files={
                        "file": (
                            "incremental-test.md",
                            b"# Incremental upload\n",
                            "text/markdown",
                        )
                    },
                )

            self.assertEqual(response.status_code, 200)
            get_index.assert_called_once_with()
            insert_document.assert_called_once_with(index, uploaded)
        finally:
            uploaded.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
