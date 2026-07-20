"""
Lightweight RAG eval runner (learning-focused).

Runs fixed checks against /rag and /rag-hybrid and prints:
  - answer text
  - latency_ms
  - pass/fail based on required substrings

Usage:
  cd backend
  .venv/bin/python rag_eval.py --rebuild
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any

from fastapi.testclient import TestClient


@dataclass(frozen=True)
class EvalCase:
    name: str
    question: str
    required_substrings: tuple[str, ...]
    forbidden_substrings: tuple[str, ...] = ()


CASES: tuple[EvalCase, ...] = (
    EvalCase(
        name="fridge_password",
        question="What is the fridge password?",
        required_substrings=("BANANA-42",),
    ),
    EvalCase(
        name="fridge_setter_unknown",
        question="What is the fridge password and who set it?",
        required_substrings=("BANANA-42",),
        forbidden_substrings=("set by John", "set by Jane"),
    ),
    EvalCase(
        name="datasync_support",
        question="What does DataSync support?",
        required_substrings=("PostgreSQL", "MySQL", "Snowflake", "BigQuery"),
    ),
)


def _check_answer(answer: str, case: EvalCase) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for token in case.required_substrings:
        if token.lower() not in answer.lower():
            failures.append(f"missing required token: {token}")
    for token in case.forbidden_substrings:
        if token.lower() in answer.lower():
            failures.append(f"contains forbidden token: {token}")
    return (len(failures) == 0, failures)


def _call_endpoint(
    client: TestClient, endpoint: str, question: str, session_id: str
) -> tuple[int, dict[str, Any], float]:
    started = time.perf_counter()
    response = client.post(
        endpoint,
        json={"question": question, "session_id": session_id},
    )
    latency_ms = (time.perf_counter() - started) * 1000
    payload = response.json()
    return response.status_code, payload, latency_ms


def run_eval(*, rebuild: bool, chunk_size: int | None, chunk_overlap: int | None) -> dict[str, Any]:
    if chunk_size is not None:
        os.environ["RAG_CHUNK_SIZE"] = str(chunk_size)
    if chunk_overlap is not None:
        os.environ["RAG_CHUNK_OVERLAP"] = str(chunk_overlap)

    # Import after env overrides so main.py reads updated knobs.
    main = importlib.import_module("main")
    client = TestClient(main.app)
    report: dict[str, Any] = {
        "config": {
            "chunk_size": main.RAG_CHUNK_SIZE,
            "chunk_overlap": main.RAG_CHUNK_OVERLAP,
            "similarity_top_k": 3,
        },
        "results": [],
    }

    if rebuild:
        rebuild_resp = client.post("/rag/rebuild")
        report["rebuild"] = rebuild_resp.json()

    endpoints = ("/rag", "/rag-hybrid")
    pass_count = 0
    total = len(CASES) * len(endpoints)

    for case in CASES:
        for endpoint in endpoints:
            status, payload, latency_ms = _call_endpoint(
                client,
                endpoint,
                case.question,
                session_id=f"eval-{case.name}-{endpoint.replace('/', '')}",
            )

            answer = payload.get("answer", "") if isinstance(payload, dict) else ""
            passed, failures = _check_answer(answer, case)
            if passed and status == 200:
                pass_count += 1

            report["results"].append(
                {
                    "case": case.name,
                    "endpoint": endpoint,
                    "status_code": status,
                    "latency_ms": round(latency_ms, 1),
                    "passed": bool(passed and status == 200),
                    "failures": failures,
                    "answer": answer,
                    "retrieval_query": payload.get("retrieval_query")
                    if isinstance(payload, dict)
                    else None,
                }
            )

    report["summary"] = {
        "passed": pass_count,
        "total": total,
        "pass_rate": round((pass_count / total) * 100, 1) if total else 0.0,
    }
    return report


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="Run quick RAG eval checks.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Run POST /rag/rebuild before eval cases.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override RAG_CHUNK_SIZE for this run (e.g. 256).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Override RAG_CHUNK_OVERLAP for this run (e.g. 64).",
    )
    args = parser.parse_args()

    report = run_eval(
        rebuild=args.rebuild,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main_cli()
