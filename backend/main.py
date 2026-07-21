"""
AI Chat Learning Backend

What this file does:
  - FastAPI HTTP API (routes, validation, CORS, SSE)
  - LangChain + Groq for chat (/chat, /chat/stream, session memory, weather tool)
  - LlamaIndex RAG over ./data (/rag) with vectors in Supabase pgvector

Request flow (high level):
  Chat mode:
    POST /chat        → create_agent (automatic tool loop)
    POST /chat/stream → manual bind_tools loop → SSE tokens to Next.js BFF → UI

  Docs mode:
    POST /rag → sync_rag_session_history → chat_engine.chat
      → embed question → retrieve top-k from Supabase → Groq answers with context

Two kinds of "session" (both in-memory, lost on server restart):
  - chat_sessions  → general chat (/chat) — LangChain message history
  - rag_sessions   → doc Q&A (/rag) — cached LlamaIndex chat engine per session

The Next.js UI also sends optional `history` on /chat and /rag so context
survives uvicorn reload. With the UI, the browser is the source of truth;
server memory is a fallback for direct API calls (curl, /docs).

Run (from backend/, with venv active):
  uvicorn main:app --reload --host 127.0.0.1 --port 8000
  Docs: http://127.0.0.1:8000/docs
"""

import os
import re
import time
import warnings
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Sequence, cast

import httpx

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, SecretStr
from rag_hybrid import create_rag_hybrid_router, hybrid_rag_sessions
from ops import (
    RateLimitMiddleware,
    RequestLoggingMiddleware,
    configure_logging,
)

# --- LangChain: orchestrates chat, prompts, memory, tools ---
from langchain_groq import ChatGroq
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain.agents import create_agent

# --- LlamaIndex: load docs → embed → search → answer (RAG) ---
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    StorageContext,
)
from llama_index.llms.groq import Groq

# Cloud embeddings via HF Inference API (no local torch ~2GB)
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding

# Persist vectors in Supabase Postgres (pgvector) instead of RAM only
from llama_index.vector_stores.supabase import SupabaseVectorStore
from llama_index.core.schema import NodeWithScore, TransformComponent
from llama_index.core.chat_engine.types import BaseChatEngine, ChatMode
from llama_index.core.llms import ChatMessage as LIChatMessage, MessageRole
from llama_index.core.node_parser import SentenceSplitter

# vecs — low-level client used to delete Supabase collections on /rag/rebuild
import vecs

# =============================================================================
# CONFIG — read secrets from backend/.env (never commit .env)
# =============================================================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
# Sampling randomness (Part 3 apply): 0 ≈ focused, 0.7 default, 1.0+ more varied
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.7"))
# Phase 6: fail hung Groq calls instead of waiting forever
GROQ_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", "60"))
# Phase 6: per-IP sliding window on /chat* and /rag* (0 = disable)
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# Embedding vector length — must match Supabase collection dimension (384 for bge-small)
HF_EMBED_DIM = int(os.getenv("HF_EMBED_DIM", "384"))

# Supabase Postgres URI — use postgresql:// (not postgres://)
# Dashboard → Connect → copy URI (Session pooler recommended)
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL", "")
# Table name inside schema "vecs" — visible in Supabase as vecs.ai_chat_docs
SUPABASE_COLLECTION = os.getenv("SUPABASE_COLLECTION", "ai_chat_docs")

# Markdown files for RAG (lab-secret.md, project-notes.md, …)
DATA_DIR = Path(__file__).parent / "data"

ALLOWED_UPLOAD_SUFFIXES = {".md", ".txt"}
MAX_UPLOAD_BYTES = int(os.getenv("RAG_MAX_UPLOAD_BYTES", str(2 * 1024 * 1024)))

# SupabaseVectorStore scores are ~1-exp(-distance): lower = better match.
# Only show source chunks within this gap of the best score (drops weak extras).
# Note: do NOT use SimilarityPostprocessor with SupabaseVectorStore — its
# scores are ~1-exp(-distance) (lower = better), so a "min similarity" cutoff
# drops the best matches and yields Empty Response.
RAG_SOURCE_SCORE_GAP = float(os.getenv("RAG_SOURCE_SCORE_GAP", "0.08"))

# Chunking knobs (Part 4 apply experiment) — change + POST /rag/rebuild to compare.
# Defaults match the learning-doc experiment: SentenceSplitter(512, 64).
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "512"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "64"))

# Prompt for CONDENSE_PLUS_CONTEXT — uses docs AND chat history
RAG_CONTEXT_PROMPT = """\
You are a helpful assistant for document Q&A with conversation memory.

Relevant documents (may be empty or unrelated to this turn):
---------------------
{context_str}
---------------------

Rules:
- For greetings and small talk, reply naturally. Do not invent document facts.
- If the user asks about something they said earlier in this chat \
(e.g. their name), answer from the conversation history — not the documents.
- For questions about the documents / project / lab secrets, use the documents.
- If neither the documents nor the chat history contain the answer, say you don't know.
"""

# ---------------------------------------------------------------------------
# In-process caches (not the same as Supabase — these reset when uvicorn restarts)
# ---------------------------------------------------------------------------

# Document index: one shared index for all RAG sessions (vectors live in Supabase)
rag_index: VectorStoreIndex | None = None

# /chat memory: session_id → list of HumanMessage / AIMessage
chat_sessions: dict[str, list[BaseMessage]] = {}

# /rag: session_id → cached CondensePlusContextChatEngine (retriever + LLM setup).
# Conversation turns live in engine memory; optional client `history` rehydrates it.
# Same session_id on /chat and /rag does NOT share memory — separate dicts.
rag_sessions: dict[str, BaseChatEngine] = {}


# =============================================================================
# RAG HELPERS — Supabase pgvector + LlamaIndex
# =============================================================================
def _normalize_pg_url(url: str) -> str:
    """
    Fix Supabase connection strings for psycopg2 / vecs.

    Two common problems:
      1. Supabase gives postgres:// but SQLAlchemy wants postgresql://
      2. Passwords with @ # % * break URL parsing unless encoded
    """
    from urllib.parse import quote_plus, unquote

    url = (url or "").strip().strip('"').strip("'")
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]

    if "://" not in url:
        return url

    scheme, rest = url.split("://", 1)
    # Split on the LAST @ — password itself may contain @
    if "@" not in rest:
        return f"{scheme}://{rest}"

    userinfo, hostpart = rest.rsplit("@", 1)
    if ":" not in userinfo:
        return f"{scheme}://{userinfo}@{hostpart}"

    user, password = userinfo.split(":", 1)
    password_enc = quote_plus(unquote(password))
    return f"{scheme}://{user}:{password_enc}@{hostpart}"


def _safe_upload_filename(raw: str) -> str:
    """Strip path components and allow only .md / .txt basenames."""
    name = Path((raw or "upload").strip()).name
    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename")

    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_UPLOAD_SUFFIXES))
        raise HTTPException(
            status_code=400,
            detail=f"Only {allowed} files are allowed",
        )

    stem = re.sub(r"[^\w\-]+", "-", Path(name).stem, flags=re.UNICODE).strip("-")
    if not stem:
        stem = "upload"
    return f"{stem[:80]}{suffix}"


def _save_upload_to_data_dir(file: UploadFile) -> Path:
    """Write one new uploaded document into DATA_DIR."""
    filename = _safe_upload_filename(file.filename or "upload.md")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / filename
    if dest.exists():
        raise HTTPException(
            status_code=409,
            detail=(
                f"{filename} already exists. Rename the upload, or replace the "
                "file in backend/data and call /rag/rebuild."
            ),
        )

    size = 0
    chunks: list[bytes] = []
    while True:
        chunk = file.file.read(1024 * 64)
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large (max {MAX_UPLOAD_BYTES // 1024} KB)",
            )
        chunks.append(chunk)

    content = b"".join(chunks)
    if not content.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    dest.write_bytes(content)
    return dest


def _rag_text_splitter() -> SentenceSplitter:
    """Split docs into embeddable chunks (Part 4 — size/overlap trade-offs)."""
    return SentenceSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
    )


def _rag_transformations() -> list[TransformComponent]:
    """Ingest pipeline steps shared by rebuild and upload."""
    return [_rag_text_splitter()]


def _insert_uploaded_document(index: VectorStoreIndex, path: Path) -> None:
    """Embed and insert only one new file into the existing Supabase index."""
    documents = SimpleDirectoryReader(
        input_files=[path],
        filename_as_id=True,
        raise_on_error=True,
    ).load_data()
    if not documents:
        raise HTTPException(status_code=400, detail="No document content found")

    # Same splitter as rebuild — otherwise uploads would use LlamaIndex defaults.
    nodes = _rag_text_splitter().get_nodes_from_documents(documents)

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            index.insert_nodes(nodes)
            # Cached RAG engines were built against the old index state — drop them.
            rag_sessions.clear()
            return
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(1.5 * attempt)

    assert last_error is not None
    raise last_error


def _configure_llm_and_embeddings() -> None:
    """Set global LlamaIndex defaults used by RAG query + ingest."""
    # Groq answers the final question after retrieval
    Settings.llm = Groq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        timeout=GROQ_TIMEOUT_SECONDS,
    )
    # HF Inference API turns text chunks into vectors (remote, no torch)
    Settings.embed_model = HuggingFaceInferenceAPIEmbedding(
        model_name=HF_EMBED_MODEL,
        token=HUGGINGFACE_API_KEY,
        timeout=60.0,
        pooling=None,
    )
    # Explicit chunking (was LlamaIndex default until this apply experiment)
    Settings.transformations = _rag_transformations()


def _delete_supabase_collection() -> None:
    """Drop vecs.ai_chat_docs so rebuild starts fresh (used by /rag/rebuild)."""
    client = vecs.create_client(_normalize_pg_url(SUPABASE_DB_URL))
    try:
        client.delete_collection(SUPABASE_COLLECTION)
    except Exception:
        pass  # OK if collection never existed
    finally:
        try:
            client.disconnect()
        except Exception:
            pass


def _make_vector_store() -> SupabaseVectorStore:
    """Connect LlamaIndex to the Supabase pgvector collection."""
    return SupabaseVectorStore(
        postgres_connection_string=_normalize_pg_url(SUPABASE_DB_URL),
        collection_name=SUPABASE_COLLECTION,
        dimension=HF_EMBED_DIM,
    )


def _collection_is_empty(vector_store: SupabaseVectorStore) -> bool:
    """True = first run → we need to embed docs and write to Supabase."""
    try:
        collection = vector_store._collection
        if collection is None:
            return True  # no collection yet → treat as empty, will ingest
        # Dummy vector search: if zero rows, Supabase has no embeddings yet.
        # Querying without an index emits a warning from vecs; suppress here
        # because this check intentionally runs before index creation on first ingest.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Query does not have a covering index for cosine_distance.*",
            )
            rows = collection.query(
                data=[0.0] * HF_EMBED_DIM,
                limit=1,
                include_value=False,
                include_metadata=False,
            )
        return len(rows) == 0
    except Exception:
        return True


def _ensure_supabase_vector_index(vector_store: SupabaseVectorStore) -> None:
    """
    Ensure cosine index exists on vecs collection for faster retrieval.

    Safe to call repeatedly; no-op when already indexed.
    """
    try:
        collection = vector_store._collection
        if collection is None:
            return
        if not collection.is_indexed_for_measure(vecs.IndexMeasure.cosine_distance):
            collection.create_index(
                measure=vecs.IndexMeasure.cosine_distance,
                method=vecs.IndexMethod.auto,
                replace=False,
            )
    except Exception:
        # Index creation is an optimization, not correctness-critical.
        pass


def _build_rag_index(*, force_rebuild: bool = False) -> VectorStoreIndex:
    """
    Build or load the RAG index.

    Flow:
      force_rebuild → delete old collection
      collection empty → read DATA_DIR, embed via HF API, upsert to Supabase
      collection has rows → load from Supabase (fast, survives restart)
    """
    _configure_llm_and_embeddings()

    if force_rebuild:
        _delete_supabase_collection()

    vector_store = _make_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    needs_ingest = force_rebuild or _collection_is_empty(vector_store)
    if not needs_ingest:
        _ensure_supabase_vector_index(vector_store)
        # Vectors already in Supabase — just attach the index to them
        return VectorStoreIndex.from_vector_store(vector_store)

    # Load all files under backend/data/
    docs = SimpleDirectoryReader(str(DATA_DIR)).load_data()

    # HF Inference API can return transient 500s — retry a few times
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            index = VectorStoreIndex.from_documents(
                docs,
                storage_context=storage_context,
                transformations=_rag_transformations(),
            )
            _ensure_supabase_vector_index(vector_store)
            return index
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(1.5 * attempt)
    assert last_error is not None
    raise last_error


def get_rag_index(*, force_rebuild: bool = False) -> VectorStoreIndex:
    """
    Return cached RAG index, building from Supabase on first use.

    Call with force_rebuild=True after editing files in DATA_DIR.
    """
    global rag_index, rag_sessions
    if rag_index is not None and not force_rebuild:
        return rag_index

    if force_rebuild:
        # Old chat engines still point at the previous index — drop them all
        rag_sessions.clear()
        hybrid_rag_sessions.clear()

    if not GROQ_API_KEY or GROQ_API_KEY.startswith("your_"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured")

    if not HUGGINGFACE_API_KEY or HUGGINGFACE_API_KEY.startswith("your_"):
        raise HTTPException(
            status_code=500,
            detail="HUGGINGFACE_API_KEY is not configured in backend/.env",
        )

    if not SUPABASE_DB_URL or "your_" in SUPABASE_DB_URL or "<" in SUPABASE_DB_URL:
        raise HTTPException(
            status_code=500,
            detail=(
                "SUPABASE_DB_URL is not configured. "
                "Add your Supabase Postgres URI to backend/.env "
                "(must start with postgresql://)."
            ),
        )

    rag_index = _build_rag_index(force_rebuild=force_rebuild)
    return rag_index


def get_rag_chat_engine(session_id: str) -> BaseChatEngine:
    """
    Return a per-session RAG chat engine with conversation memory.

    Why not use query_engine.query()?
      - query() is stateless — each question is independent
      - chat() remembers prior turns in this session

    CONDENSE_PLUS_CONTEXT (not CONDENSE_QUESTION):
      - Rewrites follow-ups into standalone questions for retrieval
      - Still passes chat history + retrieved docs into the final LLM prompt
      - So "my name is Harsh" → "what is my name?" works from memory,
        while "fridge password?" still answers from documents

    Memory is in-process only (lost on uvicorn restart). Pass `history` from
    the client to restore follow-up context after a restart.
    """
    if session_id in rag_sessions:
        return rag_sessions[session_id]

    index = get_rag_index()
    engine = index.as_chat_engine(
        chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT,
        similarity_top_k=3,
        context_prompt=RAG_CONTEXT_PROMPT,
    )
    rag_sessions[session_id] = engine
    return engine


def _format_rag_sources(
    source_nodes: Sequence[NodeWithScore] | None,
    *,
    score_gap: float = RAG_SOURCE_SCORE_GAP,
) -> list[str]:
    """
    Short previews of retrieved chunks that actually grounded the answer.

    Supabase scores are lower-is-better. We keep the best match and any other
    chunk within `score_gap` of it — so a weak hit like project-notes.md is
    dropped when lab-secret.md is clearly the winner.
    """
    items = list(source_nodes or [])
    if not items:
        return []

    scored: list[tuple[float | None, NodeWithScore]] = [
        (item.score, item) for item in items
    ]
    known_scores = [s for s, _ in scored if s is not None]
    best = min(known_scores) if known_scores else None

    sources: list[str] = []
    for score, item in scored:
        if best is not None and score is not None and score > best + score_gap:
            continue
        text = item.get_content()
        sources.append(text[:240] + ("..." if len(text) > 240 else ""))
    return sources


# =============================================================================
# CHAT TOOLS — LangChain @tool + bind_tools loop
#
# How tool calling works (Project B):
#   1. User asks "What's the weather in London?"
#   2. Groq returns tool_calls (not text) → we run get_weather → ToolMessage
#   3. Groq reads the tool result and writes the final natural-language reply
#
# bind_tools() tells Groq which functions exist; we execute them locally and
# feed results back. The model never calls Open-Meteo directly.
# =============================================================================
# WMO weather codes returned by Open-Meteo — map to readable labels.
WEATHER_CODE_LABELS: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Rain showers",
    95: "Thunderstorm",
}

MAX_CHAT_TOOL_ROUNDS = 5  # safety cap — prevents infinite tool loops


def fetch_weather(location: str) -> str:
    """
    Look up current weather via Open-Meteo (free, no API key).

    Two HTTP calls:
      1. Geocoding API — city name → lat/lon
      2. Forecast API   — lat/lon → current conditions
    """
    place = location.strip()
    if not place:
        return "Error: location cannot be empty."

    try:
        with httpx.Client(timeout=10.0) as client:
            # Step 1: resolve "London" → coordinates
            geo = client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={
                    "name": place,
                    "count": 1,
                    "language": "en",
                    "format": "json",
                },
            )
            geo.raise_for_status()
            results = geo.json().get("results") or []
            if not results:
                return f"Could not find a place named '{place}'."

            hit = results[0]
            name = str(hit.get("name", place))
            country = str(hit.get("country", ""))
            lat = float(hit["latitude"])
            lon = float(hit["longitude"])

            # Step 2: fetch live weather for those coordinates
            wx = client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": (
                        "temperature_2m,relative_humidity_2m,apparent_temperature,"
                        "precipitation,weather_code,wind_speed_10m"
                    ),
                    "timezone": "auto",
                },
            )
            wx.raise_for_status()
            current = wx.json().get("current") or {}
    except httpx.HTTPError as exc:
        return f"Weather lookup failed: {exc}"
    except (KeyError, TypeError, ValueError) as exc:
        return f"Weather lookup failed: {exc}"

    code = current.get("weather_code")
    label = (
        WEATHER_CODE_LABELS.get(int(code), "Unknown conditions")
        if code is not None
        else "Unknown conditions"
    )
    location_label = f"{name}, {country}" if country else name

    temp = current.get("temperature_2m")
    feels = current.get("apparent_temperature")
    humidity = current.get("relative_humidity_2m")
    wind = current.get("wind_speed_10m")
    precip = current.get("precipitation")

    parts = [f"{location_label}: {label}"]
    if temp is not None and feels is not None:
        parts.append(f"Temperature {temp}°C (feels like {feels}°C)")
    elif temp is not None:
        parts.append(f"Temperature {temp}°C")
    if humidity is not None:
        parts.append(f"humidity {humidity}%")
    if wind is not None:
        parts.append(f"wind {wind} km/h")
    if precip is not None:
        parts.append(f"precipitation {precip} mm")
    return ". ".join(parts) + "."


@tool
def get_weather(location: str) -> str:
    """
    Get current weather for a city or place (e.g. London, Mumbai, New York).

    The @tool decorator registers name + docstring for Groq tool-calling.
    fetch_weather() is separate so unit tests can mock HTTP without LangChain.
    """
    return fetch_weather(location)


CHAT_TOOLS = [get_weather]
CHAT_TOOL_BY_NAME = {item.name: item for item in CHAT_TOOLS}


# =============================================================================
# CHAT HELPERS — LangChain + Groq
# =============================================================================
def get_model(*, temperature: float | None = None) -> ChatGroq:
    """Create Groq chat model for /chat endpoints."""
    if not GROQ_API_KEY or GROQ_API_KEY.startswith("your_"):
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured in backend/.env",
        )
    # Per-request override (Swagger A/B) else env / default 0.7
    temp = GROQ_TEMPERATURE if temperature is None else temperature
    return ChatGroq(
        api_key=SecretStr(
            GROQ_API_KEY
        ),  # Pydantic SecretStr — type checker expects this
        model=GROQ_MODEL,
        temperature=temp,
        timeout=GROQ_TIMEOUT_SECONDS,  # pydantic alias → request_timeout
    )


def get_model_with_tools(*, temperature: float | None = None) -> ChatGroq:
    """Groq chat model with weather tool bound for tool-calling."""
    # bind_tools returns a Runnable; ChatGroq type is close enough for our use.
    return get_model(temperature=temperature).bind_tools(CHAT_TOOLS)  # type: ignore[return-value]


def chat_system_prompt(reply_mode: str = "concise") -> str:
    """System instructions shared by /chat (agent) and /chat/stream (manual loop)."""
    style = (
        "Keep answers short (1–3 sentences)."
        if reply_mode == "concise"
        else "Give a thorough, well-structured answer with useful detail."
    )
    return (
        "You are a helpful assistant with a weather tool. "
        "Use get_weather for current conditions instead of guessing. "
        f"{style}"
    )


def get_chat_agent(reply_mode: str = "concise", *, temperature: float | None = None):
    """
    LangChain create_agent graph for POST /chat (non-streaming).

    Same tools as the manual loop, but LangGraph runs the tool-calling
    loop for you — useful to compare with invoke_chat_with_tools / astream.
    """
    return create_agent(
        get_model(temperature=temperature),
        tools=CHAT_TOOLS,
        system_prompt=chat_system_prompt(reply_mode),
    )


def _ai_text(message: AIMessage) -> str:
    content = message.content
    return content if isinstance(content, str) else str(content)


def _final_ai_text(messages: list[BaseMessage]) -> str:
    """Last AIMessage content from an agent result message list."""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return _ai_text(message)
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return _ai_text(message)
    return ""


def _run_tool_call(tool_call: object) -> str:
    """Execute one tool call Groq requested; return text for ToolMessage."""
    # Groq/LangChain may send tool_call as dict or structured object.
    if isinstance(tool_call, dict):
        name = str(tool_call.get("name", ""))
        args = tool_call.get("args", {})
    else:
        name = str(getattr(tool_call, "name", ""))
        args = getattr(tool_call, "args", {})
    tool_fn = CHAT_TOOL_BY_NAME.get(name)
    if tool_fn is None:
        return f"Unknown tool: {name}"
    if not isinstance(args, dict):
        return f"Invalid tool args for {name}"
    return str(tool_fn.invoke(args))


def _tool_call_id(tool_call: object) -> str:
    if isinstance(tool_call, dict):
        return str(tool_call.get("id", ""))
    return str(getattr(tool_call, "id", ""))


def invoke_chat_with_agent(
    session_id: str,
    message: str,
    reply_mode: str = "concise",
    *,
    temperature: float | None = None,
) -> str:
    """
    POST /chat path: LangChain create_agent (automatic tool loop).

    Pass only prior turns + the new human message. System prompt is set on
    the agent itself (not duplicated as SystemMessage in the list).
    """
    history = chat_sessions.setdefault(session_id, [])
    agent = get_chat_agent(reply_mode, temperature=temperature)
    result = agent.invoke(
        cast(Any, {"messages": [*history, HumanMessage(content=message)]})
    )
    messages = result.get("messages", []) if isinstance(result, dict) else []
    return _final_ai_text(list(messages))


def invoke_chat_with_tools(messages: list[BaseMessage]) -> str:
    """
    Manual tool-calling loop (learning contrast to create_agent).

    Used by tests / optional non-stream experiments. Production /chat uses
    invoke_chat_with_agent; /chat/stream uses astream_chat_with_tools.
    """
    model = get_model_with_tools()
    conversation = list(messages)
    last_ai: AIMessage | None = None

    for _ in range(MAX_CHAT_TOOL_ROUNDS):
        last_ai = model.invoke(conversation)
        if not last_ai.tool_calls:
            return _ai_text(last_ai)

        # Model wants a tool — append its request, then our tool results.
        conversation.append(last_ai)
        for tool_call in last_ai.tool_calls:
            conversation.append(
                ToolMessage(
                    content=_run_tool_call(tool_call),
                    tool_call_id=_tool_call_id(tool_call),
                )
            )

    return _ai_text(last_ai) if last_ai else ""


async def astream_chat_with_tools(
    messages: list[BaseMessage],
    *,
    temperature: float | None = None,
) -> AsyncIterator[str]:
    """
    Tool-calling loop for POST /chat/stream.

    Important UX note: during a tool round the UI sees nothing for ~1–2s because:
      - Groq streams tool_call_chunks (not user-visible text)
      - we run get_weather (two HTTP calls to Open-Meteo)
    Only the final answer round yields text tokens to SSE.
    """
    model = get_model_with_tools(temperature=temperature)
    conversation = list(messages)

    for _ in range(MAX_CHAT_TOOL_ROUNDS):
        gathered: AIMessage | None = None
        async for chunk in model.astream(conversation):
            # Tool-call tokens — accumulate, do not stream to client yet.
            if chunk.tool_call_chunks:
                gathered = chunk if gathered is None else gathered + chunk
                continue

            text = chunk.content if isinstance(chunk.content, str) else ""
            if text:
                yield text
            gathered = chunk if gathered is None else gathered + chunk

        if gathered is None:
            return

        if gathered.tool_calls:
            # Tool round finished — run tool(s), then loop for final answer.
            conversation.append(gathered)
            for tool_call in gathered.tool_calls:
                conversation.append(
                    ToolMessage(
                        content=_run_tool_call(tool_call),
                        tool_call_id=_tool_call_id(tool_call),
                    )
                )
            continue

        # Final text answer — already streamed above.
        return


class HistoryMessage(BaseModel):
    """One prior turn from the Next.js client (role + content)."""

    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=8000)


def _history_to_llamaindex(prior: list[HistoryMessage] | None) -> list[LIChatMessage]:
    """Convert Next.js-style history into LlamaIndex chat messages."""
    if not prior:
        return []
    out: list[LIChatMessage] = []
    for item in prior:
        text = item.content.strip()
        if not text:
            continue
        role = MessageRole.USER if item.role == "user" else MessageRole.ASSISTANT
        out.append(LIChatMessage(role=role, content=text))
    return out[-20:]


def sync_rag_session_history(
    session_id: str, history: list[HistoryMessage] | None
) -> BaseChatEngine:
    """
    Replace RAG engine memory with client-sent history when provided.

    Same idea as sync_session_history for /chat — needed after Clear, page
    refresh, or uvicorn reload (rag_sessions lives only in RAM).
    """
    engine = get_rag_chat_engine(session_id)
    if history is None:
        return engine
    engine.reset()
    # LlamaIndex chat engines keep history on a private memory attribute.
    memory = getattr(engine, "memory", None) or getattr(engine, "_memory", None)
    if memory is not None and hasattr(memory, "set"):
        memory.set(_history_to_llamaindex(history))
    return engine


def _history_to_langchain(prior: list[HistoryMessage] | None) -> list[BaseMessage]:
    """Convert Next.js-style history into LangChain message objects."""
    if not prior:
        return []
    out: list[BaseMessage] = []
    for item in prior:
        text = item.content.strip()
        if not text:
            continue
        if item.role == "user":
            out.append(HumanMessage(content=text))
        else:
            out.append(AIMessage(content=text))
    return out[-20:]


def sync_session_history(session_id: str, history: list[HistoryMessage] | None) -> None:
    """
    Replace server memory with client-sent history when provided.

    Pattern: Next.js stores chat in localStorage and sends `history` on every
    turn (all prior messages except the current one). That way uvicorn reload
    or Clear+continue does not lose context. If history is omitted (curl), we
    keep whatever is already in chat_sessions.
    """
    if history is None:
        return
    chat_sessions[session_id] = _history_to_langchain(history)


def build_messages(
    session_id: str, message: str, reply_mode: str = "concise"
) -> list[BaseMessage]:
    """
    Build the message list Groq receives for /chat/stream (manual tool loop):
      [SystemMessage] + past turns + new HumanMessage

    reply_mode mirrors Next.js Concise / Detailed toggle.
    """
    history = chat_sessions.setdefault(session_id, [])
    return [
        SystemMessage(content=chat_system_prompt(reply_mode)),
        *history,
        HumanMessage(content=message),
    ]


def remember(session_id: str, human: str, ai: str) -> None:
    """Save this turn so the next message in the same session has context."""
    history = chat_sessions.setdefault(session_id, [])
    history.append(HumanMessage(content=human))
    history.append(AIMessage(content=ai))
    # Keep last 20 messages (~10 turns) to stay within Groq context limits.
    chat_sessions[session_id] = history[-20:]


def _is_llm_timeout(exc: BaseException) -> bool:
    """True for hung Groq / HTTP client timeouts (Phase 6)."""
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        return True
    name = type(exc).__name__.lower()
    return "timeout" in name


def _http_error_for_llm(exc: BaseException, *, kind: str = "LLM") -> HTTPException:
    if _is_llm_timeout(exc):
        return HTTPException(
            status_code=504,
            detail=f"{kind} request timed out after {GROQ_TIMEOUT_SECONDS:.0f}s",
        )
    return HTTPException(status_code=502, detail=f"{kind} request failed")


# =============================================================================
# FASTAPI APP
# =============================================================================
configure_logging()
app = FastAPI(title="AI Chat Learning Backend")
app.include_router(
    create_rag_hybrid_router(
        get_rag_index=lambda: get_rag_index(),
        get_model=lambda: get_model(),
        format_sources=lambda nodes: _format_rag_sources(nodes),
    )
)

# CORS: local Next.js + optional ALLOWED_ORIGINS (comma-separated) for cloud frontend
_default_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]
_extra_origins = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_default_origins + _extra_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Phase 6: rate limit chat/RAG (added before logging so 429s are still logged)
if RATE_LIMIT_PER_MINUTE > 0:
    app.add_middleware(RateLimitMiddleware, max_per_minute=RATE_LIMIT_PER_MINUTE)
app.add_middleware(RequestLoggingMiddleware)


# =============================================================================
# REQUEST / RESPONSE MODELS (Pydantic validates JSON automatically)
# =============================================================================
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    session_id: str = Field(default="default", min_length=1, max_length=64)
    reply_mode: str = Field(default="concise", pattern="^(concise|detailed)$")
    # Optional A/B for Part 3: omit → GROQ_TEMPERATURE (default 0.7)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    # Optional: prior turns from Next.js (excluding the current user message).
    # When set, replaces chat_sessions[session_id] before this turn.
    history: list[HistoryMessage] | None = None


class ChatResponse(BaseModel):
    reply: str


class RagRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=8000)
    # Reuse the same session_id across calls for follow-up questions
    session_id: str = Field(default="default", min_length=1, max_length=64)
    # Optional: prior turns from Next.js (excluding the current question).
    # When set, replaces rag engine memory before this turn.
    history: list[HistoryMessage] | None = None


class RagResponse(BaseModel):
    answer: str
    sources: list[str]  # relevant chunk previews (API only; UI may ignore)


class RagUploadResponse(BaseModel):
    status: str
    filename: str
    files_seen: int
    data_dir: str


# =============================================================================
# ROUTES — basic
# =============================================================================
@app.get("/")
def root():
    return {"message": "Hello from FastAPI! Open /docs to try the API."}


@app.get("/health")
def health():
    """Quick check that keys are set — never returns actual secret values."""
    return {
        "status": "ok",
        "groq_key_configured": bool(
            GROQ_API_KEY and not GROQ_API_KEY.startswith("your_")
        ),
        "huggingface_key_configured": bool(
            HUGGINGFACE_API_KEY and not HUGGINGFACE_API_KEY.startswith("your_")
        ),
        "supabase_configured": bool(
            SUPABASE_DB_URL
            and "your_" not in SUPABASE_DB_URL
            and "<" not in SUPABASE_DB_URL
        ),
        "groq_model": GROQ_MODEL,
        "groq_temperature": GROQ_TEMPERATURE,
        "groq_timeout_seconds": GROQ_TIMEOUT_SECONDS,
        "rate_limit_per_minute": RATE_LIMIT_PER_MINUTE,
        "hf_embed_model": HF_EMBED_MODEL,
        "supabase_collection": SUPABASE_COLLECTION,
        "chat_tools": [item.name for item in CHAT_TOOLS],
        "chat_nonstream": "create_agent",
        "chat_stream": "manual_bind_tools_loop",
        "rag_default": "llamaindex_chat_engine",
        "rag_hybrid": "llamaindex_retriever_plus_langchain_answer",
    }


# =============================================================================
# ROUTES — chat + RAG (split into routers/ for Phase 1 optional cleanup)
#
# Chat (routers/chat.py):
#   POST /chat        → create_agent (automatic tool loop)
#   POST /chat/stream → manual bind_tools + astream loop (UI uses this)
#
# RAG (routers/rag.py):
#   POST /rag, /rag/rebuild, /rag/upload, DELETE /rag/session/{id}
#
# Hybrid RAG stays in rag_hybrid.py (included above).
# =============================================================================
from routers.chat import create_chat_router
from routers.rag import create_rag_router

app.include_router(create_chat_router())
app.include_router(create_rag_router())
