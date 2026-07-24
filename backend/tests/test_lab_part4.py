"""
Tests for labs/fastapi_lab_part4.py (middleware).

Run from backend/:
  .venv/bin/python -m unittest tests.test_lab_part4 -v
"""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from labs.fastapi_lab_part4 import app


class MiddlewareLabTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_process_time_and_request_id_headers(self) -> None:
        response = self.client.get("/hello")
        self.assertEqual(response.status_code, 200)
        self.assertIn("x-process-time", response.headers)
        self.assertIn("x-lab-request-id", response.headers)
        # header value should parse as a float seconds
        float(response.headers["x-process-time"])

    def test_client_can_pass_request_id(self) -> None:
        response = self.client.get(
            "/hello",
            headers={"X-Lab-Request-Id": "fixed-id-123"},
        )
        self.assertEqual(response.headers["x-lab-request-id"], "fixed-id-123")

    def test_block_middleware_short_circuits(self) -> None:
        response = self.client.get("/blocked/x")
        self.assertEqual(response.status_code, 429)
        self.assertIn("rate limit", response.json()["detail"])
        # Still gets request-id (outer middleware ran)
        self.assertIn("x-lab-request-id", response.headers)

    def test_hello_body(self) -> None:
        self.assertEqual(self.client.get("/hello").json()["message"], "hello from inside the route")
