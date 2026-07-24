"""
Tests for labs/deepagents_lab.py — construction + inspect (no live LLM required
except the optional live invoke test).

Run from backend/ with the Deep Agents venv:

  .venv-da/bin/python -m unittest tests.test_deepagents_lab -v
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from labs import deepagents_lab as lab


class InspectAndRootTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(lab.app)

    def test_root_lists_knobs(self) -> None:
        data = self.client.get("/").json()
        self.assertEqual(data["lab"], "deepagents-customization")
        self.assertIn("model", data["knobs"])
        self.assertIn("interrupt_on (HITL)", data["knobs"])

    def test_inspect_shows_wiring(self) -> None:
        data = self.client.get("/inspect").json()
        self.assertIn("get_weather", data["tools"])
        self.assertEqual(data["subagents"], ["research-agent"])
        self.assertEqual(data["hitl"]["interrupt_on"]["notify_email"], True)


class BuildAgentTest(unittest.TestCase):
    def test_builders_pass_expected_knobs(self) -> None:
        """Assert create_deep_agent gets the customization knobs (no live LLM)."""
        fake_model = object()
        fake_graph = MagicMock(name="CompiledGraph")
        fake_graph.invoke = MagicMock()

        with (
            patch.object(lab, "get_lab_model", return_value=fake_model),
            patch.object(lab, "create_deep_agent", return_value=fake_graph) as cda,
        ):
            lab.build_basic_agent()
            lab.build_subagent_agent()
            lab.build_hitl_agent()

        self.assertEqual(cda.call_count, 3)
        basic_kw = cda.call_args_list[0].kwargs
        self.assertIs(basic_kw["model"], fake_model)
        self.assertEqual(basic_kw["tools"], lab.LAB_TOOLS)
        self.assertEqual(basic_kw["system_prompt"], lab.SYSTEM_PROMPT)
        self.assertEqual(basic_kw["middleware"], [lab.log_tool_calls])

        sub_kw = cda.call_args_list[1].kwargs
        self.assertEqual(sub_kw["subagents"], [lab.RESEARCH_SUBAGENT])

        hitl_kw = cda.call_args_list[2].kwargs
        self.assertEqual(hitl_kw["interrupt_on"]["notify_email"], True)
        self.assertIs(hitl_kw["checkpointer"], lab.HITL_CHECKPOINTER)


@unittest.skipUnless(
    os.getenv("GROQ_API_KEY") and not os.getenv("GROQ_API_KEY", "").startswith("your_"),
    "GROQ_API_KEY not configured",
)
class LiveInvokeTest(unittest.TestCase):
    def test_plain_reply_without_tools(self) -> None:
        client = TestClient(lab.app)
        res = client.post(
            "/run",
            json={"message": "Say the word ok only.", "mode": "basic"},
        )
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertIn("ok", data["reply"].lower())


if __name__ == "__main__":
    unittest.main()
