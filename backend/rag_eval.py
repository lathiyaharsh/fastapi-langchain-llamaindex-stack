"""
Lightweight RAG eval + latency profiler (learning-focused).

Modes:
  1) Quality eval (default)
       .venv/bin/python rag_eval.py --rebuild
  2) Latency profile (/rag vs /rag-hybrid)
       .venv/bin/python rag_eval.py --profile

Eval prints pass/fail + latency per case.
Profile prints avg / p50 / p95 per endpoint over a fixed question batch.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
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

# Fixed batch for latency A/B (5 questions × 2 endpoints = 10 calls).
PROFILE_QUESTIONS: tuple[str, ...] = (
    "What is the fridge password?",
    "What does DataSync support?",
    "Where is Acme Analytics headquarters?",
    "What are mentoring office hours?",
    "What stack does this FastAPI learning backend use?",
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


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Nearest-rank percentile on a pre-sorted list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = max(0, min(len(sorted_vals) - 1, int(round((pct / 100) * (len(sorted_vals) - 1)))))
    return sorted_vals[rank]


def _latency_stats(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    return {
        "count": len(ordered),
        "avg_ms": round(statistics.fmean(ordered), 1) if ordered else 0.0,
        "min_ms": round(ordered[0], 1) if ordered else 0.0,
        "max_ms": round(ordered[-1], 1) if ordered else 0.0,
        "p50_ms": round(_percentile(ordered, 50), 1),
        "p95_ms": round(_percentile(ordered, 95), 1),
    }


def _load_main(*, chunk_size: int | None, chunk_overlap: int | None):
    if chunk_size is not None:
        os.environ["RAG_CHUNK_SIZE"] = str(chunk_size)
    if chunk_overlap is not None:
        os.environ["RAG_CHUNK_OVERLAP"] = str(chunk_overlap)
    return importlib.import_module("main")


def run_eval(*, rebuild: bool, chunk_size: int | None, chunk_overlap: int | None) -> dict[str, Any]:
    main = _load_main(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    client = TestClient(main.app)
    report: dict[str, Any] = {
        "mode": "eval",
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


def run_profile(*, rebuild: bool, chunk_size: int | None, chunk_overlap: int | None) -> dict[str, Any]:
    """
    Compare /rag vs /rag-hybrid latency on a fixed question batch.

    Warm-up: one ignored call per endpoint (first call often includes index load).
    Then run PROFILE_QUESTIONS against each endpoint and aggregate stats.
    """
    main = _load_main(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    client = TestClient(main.app)
    endpoints = ("/rag", "/rag-hybrid")
    report: dict[str, Any] = {
        "mode": "profile",
        "config": {
            "chunk_size": main.RAG_CHUNK_SIZE,
            "chunk_overlap": main.RAG_CHUNK_OVERLAP,
            "questions": list(PROFILE_QUESTIONS),
            "calls_total": len(PROFILE_QUESTIONS) * len(endpoints),
        },
        "samples": [],
        "by_endpoint": {},
    }

    if rebuild:
        rebuild_resp = client.post("/rag/rebuild")
        report["rebuild"] = rebuild_resp.json()

    # Warm-up (exclude from stats) — first request can pay index/embed setup cost.
    for endpoint in endpoints:
        _call_endpoint(
            client,
            endpoint,
            "What is the fridge password?",
            session_id=f"warmup-{endpoint.replace('/', '')}",
        )

    latencies: dict[str, list[float]] = {endpoint: [] for endpoint in endpoints}

    for idx, question in enumerate(PROFILE_QUESTIONS):
        for endpoint in endpoints:
            status, payload, latency_ms = _call_endpoint(
                client,
                endpoint,
                question,
                session_id=f"profile-{idx}-{endpoint.replace('/', '')}",
            )
            answer = payload.get("answer", "") if isinstance(payload, dict) else ""
            sample = {
                "question": question,
                "endpoint": endpoint,
                "status_code": status,
                "latency_ms": round(latency_ms, 1),
                "answer_preview": (answer[:120] + "…") if len(answer) > 120 else answer,
                "rerank_applied": payload.get("rerank_applied")
                if isinstance(payload, dict)
                else None,
            }
            report["samples"].append(sample)
            if status == 200:
                latencies[endpoint].append(latency_ms)

    for endpoint, samples_ms in latencies.items():
        report["by_endpoint"][endpoint] = _latency_stats(samples_ms)

    # Quick compare: which endpoint is faster on average?
    rag_avg = report["by_endpoint"].get("/rag", {}).get("avg_ms")
    hybrid_avg = report["by_endpoint"].get("/rag-hybrid", {}).get("avg_ms")
    if rag_avg is not None and hybrid_avg is not None and rag_avg > 0:
        delta = round(hybrid_avg - rag_avg, 1)
        report["comparison"] = {
            "faster_avg": "/rag" if rag_avg <= hybrid_avg else "/rag-hybrid",
            "hybrid_minus_rag_ms": delta,
            "note": (
                "Hybrid does retrieve + optional keyword rerank + LangChain answer; "
                "/rag uses LlamaIndex chat engine end-to-end. Absolute times depend on Groq."
            ),
        }
    return report


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="Run RAG eval or latency profile.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Run POST /rag/rebuild before cases.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Latency A/B: /rag vs /rag-hybrid over PROFILE_QUESTIONS.",
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

    if args.profile:
        report = run_profile(
            rebuild=args.rebuild,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    else:
        report = run_eval(
            rebuild=args.rebuild,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main_cli()
