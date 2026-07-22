"""
Tests for labs/fastapi_lab_part2.py (errors + APIRouter).

Run from backend/:
  .venv/bin/python -m unittest tests.test_lab_part2 -v
"""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from labs.fastapi_lab_part2 import app


class ErrorsAndRoutersTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_product_ok(self) -> None:
        response = self.client.get("/shop/products/widget")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], "widget")

    def test_product_not_found_http_exception(self) -> None:
        response = self.client.get("/shop/products/nope")
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json()["detail"])
        self.assertEqual(response.headers.get("x-error"), "product-missing")

    def test_custom_exception_handler(self) -> None:
        response = self.client.get("/shop/boom")
        self.assertEqual(response.status_code, 418)
        body = response.json()
        self.assertEqual(body["error"], "stock_error")
        self.assertIn("widget", body["message"])

    def test_validation_handler_includes_body(self) -> None:
        # qty missing → 422; our handler echoes the request body
        response = self.client.post("/shop/orders", json={"product_id": "widget"})
        self.assertEqual(response.status_code, 422)
        body = response.json()
        self.assertEqual(body["error"], "validation_failed")
        self.assertEqual(body["body"], {"product_id": "widget"})

    def test_order_success(self) -> None:
        response = self.client.post(
            "/shop/orders",
            json={"product_id": "gadget", "qty": 2},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["qty"], 2)

    def test_admin_requires_router_dependency(self) -> None:
        denied = self.client.get("/admin/secret")
        self.assertEqual(denied.status_code, 401)

        allowed = self.client.get(
            "/admin/secret",
            headers={"X-Lab-Token": "lab-secret"},
        )
        self.assertEqual(allowed.status_code, 200)
        self.assertIn("welcome", allowed.json()["message"])

    def test_router_prefix_groups_routes(self) -> None:
        # prefix=/shop means path is /shop/products/... not /products/...
        self.assertEqual(self.client.get("/products/widget").status_code, 404)
        self.assertEqual(self.client.get("/shop/products/widget").status_code, 200)
