"""
Tests for labs/fastapi_lab_part3.py (SQLModel + SQLite CRUD).

Run from backend/:
  .venv/bin/python -m unittest tests.test_lab_part3 -v
"""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient
from sqlmodel import SQLModel

from labs import fastapi_lab_part3 as lab
from labs.fastapi_lab_part3 import app


class SqlLabTest(unittest.TestCase):
    def setUp(self) -> None:
        # Fresh tables for every test (same engine / SQLite file)
        SQLModel.metadata.drop_all(lab.engine)
        SQLModel.metadata.create_all(lab.engine)
        self.client = TestClient(app)

    def test_create_and_list(self) -> None:
        created = self.client.post(
            "/notes/",
            json={"title": "fridge", "body": "BANANA-42"},
        )
        self.assertEqual(created.status_code, 201)
        self.assertEqual(created.json()["title"], "fridge")
        self.assertIsInstance(created.json()["id"], int)

        listed = self.client.get("/notes/")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(len(listed.json()), 1)

    def test_read_update_delete(self) -> None:
        note_id = self.client.post(
            "/notes/",
            json={"title": "a", "body": "one"},
        ).json()["id"]

        got = self.client.get(f"/notes/{note_id}")
        self.assertEqual(got.json()["body"], "one")

        patched = self.client.patch(
            f"/notes/{note_id}",
            json={"body": "two"},
        )
        self.assertEqual(patched.status_code, 200)
        self.assertEqual(patched.json()["body"], "two")
        self.assertEqual(patched.json()["title"], "a")  # unchanged

        deleted = self.client.delete(f"/notes/{note_id}")
        self.assertEqual(deleted.status_code, 204)
        self.assertEqual(self.client.get(f"/notes/{note_id}").status_code, 404)

    def test_missing_note_404(self) -> None:
        self.assertEqual(self.client.get("/notes/999").status_code, 404)

    def test_validation_rejects_empty_title(self) -> None:
        response = self.client.post("/notes/", json={"title": "", "body": "x"})
        self.assertEqual(response.status_code, 422)
