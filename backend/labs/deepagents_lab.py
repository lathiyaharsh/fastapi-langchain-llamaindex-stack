"""
Deep Agents customization lab — Model, Tools, System prompt, Middleware,
Subagents, Backends, Sandboxes, Human-in-the-loop.

Official docs:
  https://docs.langchain.com/oss/python/deepagents/customization
  https://docs.langchain.com/oss/python/deepagents/overview

WHY a separate venv?
  `deepagents` needs Python >= 3.11. Your main backend `.venv` is 3.10.
  This lab uses `.venv-da` (Python 3.12) so main.py stays untouched.

Setup (once, from backend/):

    # if uv is missing: curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    uv python install 3.12
    uv venv .venv-da --python 3.12
    uv pip install --python .venv-da/bin/python -r requirements-deepagents.txt

Run (port 8005 — does not clash with main :8000 or FastAPI labs :8001–8004):

    .venv-da/bin/uvicorn labs.deepagents_lab:app --reload --port 8005
    # open http://127.0.0.1:8005/docs

How this maps to your REAL backend (main.py):

  create_agent(...)           ≈ thin tool loop
  create_deep_agent(...)      ≈ create_agent + planning + filesystem + subagents + more

  get_model() / ChatGroq      ≈ model= here
  CHAT_TOOLS / get_weather    ≈ tools= here
  chat_system_prompt()        ≈ system_prompt= here
  (no agent middleware yet)   ≈ middleware= here  (different from FastAPI ops.py MW!)
  (single agent only)         ≈ subagents= here
  (no agent filesystem)       ≈ backend= / sandboxes here
  (no approval gates)         ≈ interrupt_on= + checkpointer here

[JS] Think of create_deep_agent as a pre-wired "agent framework" on top of
     LangGraph — like NestJS modules vs a bare Express handler.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, cast

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain.agents.middleware import wrap_tool_call
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field, SecretStr

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    SubAgent,
    create_deep_agent,
    register_harness_profile,
)
from deepagents.backends import FilesystemBackend, StateBackend

# Load backend/.env (GROQ_API_KEY) — same keys as main.py
_BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(_BACKEND_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# -----------------------------------------------------------------------------
# Groq + Llama: Deep Agents' default BASE system prompt + many tools often make
# Llama emit XML-ish tool calls Groq rejects (`tool_use_failed`, e.g.
# `<function=get_weather{...}</function>`). Fix for this lab:
#   1) replace the huge base prompt
#   2) disable general-purpose subagent (no `task` unless mode=subagents)
#   3) exclude filesystem + write_todos tools
# Docs: https://docs.langchain.com/oss/python/deepagents/profiles
# -----------------------------------------------------------------------------
register_harness_profile(
    "groq",
    HarnessProfile(
        base_system_prompt=(
            "You are a concise lab assistant. "
            "Use get_weather for weather questions. "
            "Use notify_email only when asked to email someone. "
            "For greetings and small talk, reply in plain text with no tools. "
            "Never invent tool or subagent names."
        ),
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        excluded_tools=frozenset(
            {
                "ls",
                "read_file",
                "write_file",
                "edit_file",
                "glob",
                "grep",
                "execute",
                "write_todos",
            }
        ),
    ),
)

app = FastAPI(title="Deep Agents lab — customization")

# Tool-call log for the middleware demo (in-memory, resets on reload)
TOOL_LOG: list[str] = []


# =============================================================================
# 1. MODEL — which LLM drives the agent
# =============================================================================
# Docs: provider:model string OR a ChatModel instance.
# Your main.py already uses ChatGroq — pass that instance here.
#
# [JS] Like choosing OpenAI vs Anthropic client before calling chat.completions.


def get_lab_model(*, temperature: float = 0) -> ChatGroq:
    if not GROQ_API_KEY or GROQ_API_KEY.startswith("your_"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY missing in backend/.env")
    # cast(Any, …): ChatGroq stubs want pydantic SecretStr; with two venvs on
    # the analysis path Pyright can see two incompatible SecretStr classes.
    # Runtime is fine — ChatGroq accepts SecretStr (and plain str).
    return ChatGroq(
        api_key=cast(Any, SecretStr(GROQ_API_KEY)),
        model=GROQ_MODEL,
        temperature=temperature,
    )


# =============================================================================
# 2. TOOLS — domain actions the agent can call
# =============================================================================
# Deep Agents already ship planning / filesystem / task tools.
# You ADD domain tools (weather, search, email…) just like create_agent(tools=).
#
# NOTE: Groq + Llama sometimes mishandles MANY tools at once (deepagents adds
# several built-ins). Prefer short prompts; if tool_use_failed, retry or use
# a stronger tool-calling model.


@tool
def get_weather(city: str) -> str:
    """Return current weather for a city (lab fake — no network)."""
    return f"Lab weather: sunny, 22°C in {city}"


@tool
def notify_email(to: str, subject: str) -> str:
    """Pretend to send an email (sensitive — good HITL candidate)."""
    return f"Lab email queued to {to!r} subject={subject!r}"


LAB_TOOLS = [get_weather, notify_email]


# =============================================================================
# 3. SYSTEM PROMPT — your instructions prepended to the deep-agent base prompt
# =============================================================================
# Deep Agents keep a BASE prompt that teaches planning / files / subagents.
# system_prompt= is YOUR domain layer on top (like chat_system_prompt() in main).

SYSTEM_PROMPT = (
    "You are a concise lab assistant. "
    "Use get_weather for weather questions. "
    "Use notify_email only when the user asks to email someone. "
    "For hellos and small talk, reply in plain text with no tools. "
    "Keep answers short."
)


# =============================================================================
# 4. MIDDLEWARE — cross-cutting hooks around model/tool calls
# =============================================================================
# NOT the same as FastAPI middleware (ops.py RequestLoggingMiddleware).
# Agent middleware wraps the LangGraph agent loop (before/after tools, etc.).
# [JS] Closer to Express middleware for an internal RPC bus than HTTP.


def _tool_call_name(request: Any) -> str:
    """ToolCallRequest has .tool_call {name, args, id} — not .name."""
    tool_call = getattr(request, "tool_call", None)
    if isinstance(tool_call, dict):
        return str(tool_call.get("name") or "?")
    name = getattr(tool_call, "name", None)
    if name:
        return str(name)
    tool = getattr(request, "tool", None)
    return str(getattr(tool, "name", None) or "?")


@wrap_tool_call
def log_tool_calls(request, handler):
    """Log every tool call — demo of custom agent middleware."""
    name = _tool_call_name(request)
    TOOL_LOG.append(f"call:{name}")
    result = handler(request)
    TOOL_LOG.append(f"done:{name}")
    return result


# =============================================================================
# 5. SUBAGENTS — delegate heavy work to an isolated worker agent
# =============================================================================
# Main agent gets a `task` tool; worker runs with its own tools + prompt.
# Keeps the parent context small (coordinator / worker pattern).

# SubAgent is a TypedDict; optional keys use NotRequired. Some analyzers still
# treat those as required, so cast() documents the runtime-valid shape.
RESEARCH_SUBAGENT = cast(
    SubAgent,
    {
        "name": "research-agent",
        "description": "Look up weather details when the user wants a short report",
        "system_prompt": (
            "You research weather using get_weather and reply in 1-2 sentences."
        ),
        "tools": [get_weather],
        # Optional: "model": "..." to override the parent model
    },
)


# =============================================================================
# 6. BACKENDS — where filesystem tools read/write
# =============================================================================
# StateBackend (default): virtual files in LangGraph state (thread-scoped).
# FilesystemBackend: real disk under root_dir (use virtual_mode=True carefully).
#
# Sandboxes (next section) are special backends with isolated execute shells.


def state_backend() -> StateBackend:
    """Default: files live in graph state, not on your laptop disk."""
    return StateBackend()


def lab_filesystem_backend() -> FilesystemBackend:
    """Optional disk backend rooted at labs/ (virtual paths)."""
    return FilesystemBackend(
        root_dir=str(Path(__file__).resolve().parent),
        virtual_mode=True,
    )


# =============================================================================
# 7. SANDBOXES — isolated execute + filesystem (documented, not live here)
# =============================================================================
# Sandboxes are backends with an `execute` tool (run shell in isolation).
# Providers: LangSmithSandbox, Daytona, E2B, Modal, Runloop, Vercel…
# They need extra packages + API keys — so this lab only shows the shape:
#
#   from deepagents.backends import LangSmithSandbox
#   backend = LangSmithSandbox(sandbox=client.create_sandbox())
#   create_deep_agent(..., backend=backend)
#
# Prefer sandboxes when the agent should install deps / run pytest safely.


# =============================================================================
# 8. HUMAN-IN-THE-LOOP — pause before sensitive tools
# =============================================================================
# interrupt_on={tool_name: True | config} + checkpointer REQUIRED.
# Flow: invoke → interrupt → human decides → Command(resume=…) → continue.
#
# [JS] Like a confirmation modal before a destructive API call.


HITL_CHECKPOINTER = MemorySaver()


def build_basic_agent():
    """Model + tools + system prompt + logging middleware (no task/subagents)."""
    return create_deep_agent(
        model=get_lab_model(),
        tools=LAB_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=[log_tool_calls],
        # backend=StateBackend() is the default — shown explicitly for learning
        backend=state_backend(),
        # subagents omitted + groq profile disables general-purpose → no `task` tool
    )


def build_subagent_agent():
    """Opt-in research worker via task(); GP subagent still disabled by profile."""
    return create_deep_agent(
        model=get_lab_model(),
        tools=LAB_TOOLS,
        system_prompt=(
            SYSTEM_PROMPT
            + " Delegate multi-step weather research with the task tool "
            "to research-agent only — never invent other subagent names."
        ),
        middleware=[log_tool_calls],
        subagents=[RESEARCH_SUBAGENT],
        backend=state_backend(),
    )


def build_hitl_agent():
    """Pause before notify_email; get_weather runs freely."""
    return create_deep_agent(
        model=get_lab_model(),
        tools=LAB_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        interrupt_on={
            "notify_email": True,  # approve / edit / reject / respond
            "get_weather": False,
        },
        checkpointer=HITL_CHECKPOINTER,
        backend=state_backend(),
    )


# =============================================================================
# FastAPI surface — try each knob from /docs
# =============================================================================


class RunRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    mode: Literal["basic", "subagents"] = "basic"


class HitlStartRequest(BaseModel):
    message: str = Field(
        default="Email bob@example.com that the lab is ready",
        min_length=1,
        max_length=2000,
    )
    thread_id: str = Field(default="lab-hitl-1", max_length=64)


class HitlResumeRequest(BaseModel):
    thread_id: str = Field(default="lab-hitl-1", max_length=64)
    decision: Literal["approve", "reject"] = "approve"


def _final_text(result: dict[str, Any]) -> str:
    messages = result.get("messages") or []
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content.strip():
            return content
    return ""


def _interrupt_payload(result: Any) -> Any:
    """Normalize LangGraph interrupt info for JSON responses."""
    if isinstance(result, dict) and result.get("__interrupt__"):
        raw = result["__interrupt__"]
        return [getattr(i, "value", i) for i in raw]
    return None


@app.get("/")
def root():
    return {
        "lab": "deepagents-customization",
        "docs": "https://docs.langchain.com/oss/python/deepagents/customization",
        "knobs": [
            "model",
            "tools",
            "system_prompt",
            "middleware",
            "subagents",
            "backend",
            "sandboxes (see comments)",
            "interrupt_on (HITL)",
        ],
        "try": [
            "GET  /inspect",
            "POST /run  {message, mode: basic|subagents}",
            "GET  /middleware-log",
            "POST /hitl/start  then POST /hitl/resume",
        ],
        "vs_main_py": {
            "create_agent": "thin automatic tool loop on /chat",
            "create_deep_agent": "same idea + todos, files, subagents, HITL",
        },
        "venv": ".venv-da (Python 3.12) — deepagents needs >=3.11",
    }


@app.get("/inspect")
def inspect_knobs():
    """No LLM call — just show how the agent is wired."""
    return {
        "model": f"ChatGroq({GROQ_MODEL})",
        "tools": [t.name for t in LAB_TOOLS],
        "system_prompt_preview": SYSTEM_PROMPT[:80] + "…",
        "middleware": ["log_tool_calls (wrap_tool_call)"],
        "subagents": [RESEARCH_SUBAGENT["name"]],
        "groq_harness_profile": {
            "base_system_prompt": "short lab prompt (replaces deepagents default base)",
            "general_purpose_subagent": False,
            "excluded_tools": [
                "ls",
                "read_file",
                "write_file",
                "edit_file",
                "glob",
                "grep",
                "execute",
                "write_todos",
            ],
            "note": "avoids Groq tool_use_failed from XML-style Llama tool calls",
        },
        "backend_default": "StateBackend (in-graph virtual FS)",
        "backend_optional": "FilesystemBackend(root=labs/, virtual_mode=True)",
        "sandboxes": "LangSmith / Daytona / E2B / Modal / … — see section 7 comments",
        "hitl": {
            "interrupt_on": {"notify_email": True, "get_weather": False},
            "checkpointer": "MemorySaver (required for HITL)",
        },
    }


@app.post("/run")
def run_agent(body: RunRequest):
    """
    Invoke a deep agent.

    Prefer mode=basic for greets / weather (no `task` tool).
    mode=subagents adds research-agent — use for multi-step delegation demos.
    """
    TOOL_LOG.clear()
    agent = build_subagent_agent() if body.mode == "subagents" else build_basic_agent()
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": body.message}]})
    except Exception as exc:  # noqa: BLE001 — surface provider quirks in /docs
        detail = f"Agent invoke failed ({type(exc).__name__}): {exc}"
        if "tool_use_failed" in str(exc):
            detail += (
                " | Tip: use mode=basic for simple chat; Groq/Llama often breaks "
                "on deepagents' task/filesystem tools. Profile trims those for groq."
            )
        raise HTTPException(status_code=502, detail=detail) from exc
    return {
        "mode": body.mode,
        "reply": _final_text(result if isinstance(result, dict) else {}),
        "middleware_log": list(TOOL_LOG),
        "message_count": len((result or {}).get("messages", []))
        if isinstance(result, dict)
        else 0,
    }


@app.get("/middleware-log")
def middleware_log():
    return {"tool_log": list(TOOL_LOG)}


@app.post("/hitl/start")
def hitl_start(body: HitlStartRequest):
    """
    Start a run that should pause before notify_email.

    If the model calls notify_email, response includes `interrupted: true`
    and an interrupt payload. Then call POST /hitl/resume with the same thread_id.
    """
    agent = build_hitl_agent()
    config: RunnableConfig = {"configurable": {"thread_id": body.thread_id}}
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": body.message}]},
            config=config,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"HITL start failed ({type(exc).__name__}): {exc}",
        ) from exc

    interrupts = _interrupt_payload(result)
    if interrupts:
        return {
            "interrupted": True,
            "thread_id": body.thread_id,
            "interrupts": interrupts,
            "next": "POST /hitl/resume with decision=approve|reject",
        }
    return {
        "interrupted": False,
        "thread_id": body.thread_id,
        "reply": _final_text(result if isinstance(result, dict) else {}),
        "note": "Model finished without hitting interrupt_on tools",
    }


@app.post("/hitl/resume")
def hitl_resume(body: HitlResumeRequest):
    """Resume after /hitl/start interrupt (same thread_id)."""
    agent = build_hitl_agent()
    config: RunnableConfig = {"configurable": {"thread_id": body.thread_id}}
    # approve → run the tool; reject → skip / cancel that tool call
    decision = {"type": body.decision}
    # Command defaults goto=(), so inference is Command[tuple[()]]; invoke
    # expects Command[Unknown] — cast keeps the HITL resume shape type-clean.
    resume_cmd = cast(Any, Command(resume={"decisions": [decision]}))
    try:
        result = agent.invoke(resume_cmd, config=config)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"HITL resume failed ({type(exc).__name__}): {exc}",
        ) from exc

    interrupts = _interrupt_payload(result)
    return {
        "interrupted": bool(interrupts),
        "interrupts": interrupts,
        "reply": _final_text(result if isinstance(result, dict) else {}),
        "decision": body.decision,
    }
