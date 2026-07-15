"""
AI Chat Learning Backend

What this file does:
  - FastAPI HTTP API (routes, validation, CORS, SSE)
  - LangChain + Groq for chat (/chat, /chat/stream, session memory)
  - LlamaIndex RAG over ./data (/rag) with vectors in Supabase pgvector

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
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, SecretStr

# --- LangChain: orchestrates chat, prompts, memory ---
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

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
from llama_index.core.schema import NodeWithScore
from llama_index.core.chat_engine.types import BaseChatEngine, ChatMode
from llama_index.core.llms import ChatMessage as LIChatMessage, MessageRole

# vecs — low-level client used to delete Supabase collections on /rag/rebuild
import vecs

# =============================================================================
# CONFIG — read secrets from backend/.env (never commit .env)
# =============================================================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

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

# SupabaseVectorStore scores are ~1-exp(-distance): lower = better match.
# Only show source chunks within this gap of the best score (drops weak extras).
RAG_SOURCE_SCORE_GAP = float(os.getenv("RAG_SOURCE_SCORE_GAP", "0.08"))

# Greetings / intros / chat-memory questions — answer without showing doc sources.
# Note: do NOT use SimilarityPostprocessor with SupabaseVectorStore — its
# scores are ~1-exp(-distance) (lower = better), so a "min similarity" cutoff
# drops the best matches and yields Empty Response.
_CONVERSATIONAL_RE = re.compile(
    r"^(?:"
    r"hi(?:\s+there)?|hello(?:\s+there)?|hey(?:\s+there)?|hiya|howdy|"
    r"good\s+(?:morning|afternoon|evening|night)|"
    r"thanks?|thank\s+you|thx|ty|"
    r"bye|goodbye|see\s+ya|see\s+you|"
    r"ok(?:ay)?|cool|nice"
    r")[!?. ]*$",
    re.IGNORECASE,
)

# "hi my name is…", "what is my name", etc. — not document questions
_CHAT_MEMORY_RE = re.compile(
    r"(?is)^(?:"
    r"(?:hi|hello|hey)[,!]?\s+.*\bmy\s+name\s+is\b.*"
    r"|my\s+name\s+is\b.*"
    r"|i(?:'m|\s+am)\s+[a-z][a-z'-]{1,30}\s*[!.]*"
    r"|what(?:'s|\s+is)\s+my\s+name\b.*"
    r"|who\s+am\s+i\b.*"
    r"|do\s+you\s+remember\b.*"
    r")$",
)

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


def _configure_llm_and_embeddings() -> None:
    """Set global LlamaIndex defaults used by RAG query + ingest."""
    # Groq answers the final question after retrieval
    Settings.llm = Groq(api_key=GROQ_API_KEY, model=GROQ_MODEL)
    # HF Inference API turns text chunks into vectors (remote, no torch)
    Settings.embed_model = HuggingFaceInferenceAPIEmbedding(
        model_name=HF_EMBED_MODEL,
        token=HUGGINGFACE_API_KEY,
        timeout=60.0,
        pooling=None,
    )


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
        # Dummy vector search: if zero rows, Supabase has no embeddings yet
        rows = collection.query(
            data=[0.0] * HF_EMBED_DIM,
            limit=1,
            include_value=False,
            include_metadata=False,
        )
        return len(rows) == 0
    except Exception:
        return True


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
        # Vectors already in Supabase — just attach the index to them
        return VectorStoreIndex.from_vector_store(vector_store)

    # Load all files under backend/data/
    docs = SimpleDirectoryReader(str(DATA_DIR)).load_data()

    # HF Inference API can return transient 500s — retry a few times
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            return VectorStoreIndex.from_documents(
                docs, storage_context=storage_context
            )
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


def _is_conversational_query(question: str) -> bool:
    """True for greetings / intros / chat-memory asks that should not cite docs."""
    text = question.strip()
    return bool(_CONVERSATIONAL_RE.match(text) or _CHAT_MEMORY_RE.match(text))


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
# CHAT HELPERS — LangChain + Groq
# =============================================================================
def get_model() -> ChatGroq:
    """Create Groq chat model for /chat endpoints."""
    if not GROQ_API_KEY or GROQ_API_KEY.startswith("your_"):
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured in backend/.env",
        )
    return ChatGroq(
        api_key=SecretStr(
            GROQ_API_KEY
        ),  # Pydantic SecretStr — type checker expects this
        model=GROQ_MODEL,
        temperature=0.7,  # higher = more creative; lower = more focused
    )


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

    Needed after page refresh, Clear, or uvicorn reload. The Next.js UI sends
    `history` on every chat turn so Groq stays in sync with localStorage.
    """
    if history is None:
        return
    chat_sessions[session_id] = _history_to_langchain(history)


def build_messages(
    session_id: str, message: str, reply_mode: str = "concise"
) -> list[BaseMessage]:
    """
    Build the message list Groq receives:
      [SystemMessage] + past turns + new HumanMessage

    reply_mode mirrors Next.js Concise / Detailed toggle.
    """
    history = chat_sessions.setdefault(session_id, [])
    style = (
        "Keep answers short (1–3 sentences)."
        if reply_mode == "concise"
        else "Give a thorough, well-structured answer with useful detail."
    )
    return [
        SystemMessage(content=f"You are a helpful assistant. {style}"),
        *history,
        HumanMessage(content=message),
    ]


def remember(session_id: str, human: str, ai: str) -> None:
    """Save this turn so the next message in the same session has context."""
    history = chat_sessions.setdefault(session_id, [])
    history.append(HumanMessage(content=human))
    history.append(AIMessage(content=ai))
    chat_sessions[session_id] = history[-20:]  # cap context size


# =============================================================================
# FASTAPI APP
# =============================================================================
app = FastAPI(title="AI Chat Learning Backend")

# Allow browser calls from Next.js dev server (direct FastAPI testing / CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# REQUEST / RESPONSE MODELS (Pydantic validates JSON automatically)
# =============================================================================
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    session_id: str = Field(default="default", min_length=1, max_length=64)
    reply_mode: str = Field(default="concise", pattern="^(concise|detailed)$")
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
    sources: list[str]  # relevant chunk previews (empty for greetings / chat-memory)


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
        "hf_embed_model": HF_EMBED_MODEL,
        "supabase_collection": SUPABASE_COLLECTION,
    }


# =============================================================================
# ROUTES — chat (LangChain)
# =============================================================================
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Full reply in one JSON response (invoke = wait for complete answer)."""
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        sync_session_history(request.session_id, request.history)
        model = get_model()
        messages = build_messages(request.session_id, message, request.reply_mode)
        ai_message = model.invoke(messages)
        reply = (
            ai_message.content
            if isinstance(ai_message.content, str)
            else str(ai_message.content)
        )
        remember(request.session_id, message, reply)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail="LLM request failed") from exc

    return ChatResponse(reply=reply)


@app.post("/chat/stream")
async def chat_stream(request: Request, body: ChatRequest):
    """
    Stream tokens over SSE as Groq generates them.

    Event format: data: <chunk>\\n\\n  then  data: [DONE]\\n\\n
    On client disconnect, stops without calling remember().
    """
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        sync_session_history(body.session_id, body.history)
        model = get_model()
        messages = build_messages(body.session_id, message, body.reply_mode)
    except HTTPException:
        raise

    async def event_generator():
        parts: list[str] = []
        disconnected = False
        try:
            async for chunk in model.astream(messages):
                if await request.is_disconnected():
                    disconnected = True
                    break
                text = chunk.content if isinstance(chunk.content, str) else ""
                if text:
                    parts.append(text)
                    safe = text.replace("\n", "\\n")  # keep SSE one-line per event
                    yield f"data: {safe}\n\n"
            if disconnected:
                return
            remember(body.session_id, message, "".join(parts))
            yield "data: [DONE]\n\n"
        except Exception:
            yield "data: [ERROR] LLM request failed\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@app.delete("/chat/session/{session_id}")
def clear_session(session_id: str):
    """Clear in-memory chat history for one session."""
    chat_sessions.pop(session_id, None)
    return {"cleared": session_id}


# =============================================================================
# ROUTES — RAG (LlamaIndex + Supabase)
#
# Stateless: document vectors in Supabase (survives restart)
# Stateful:  rag_sessions caches chat engines; turns restored via `history`
#
# Chat mode: CONDENSE_PLUS_CONTEXT — condense follow-ups, retrieve docs,
# then answer using both retrieved context and chat history.
#
# Test flow in /docs:
#   1. POST /rag/rebuild
#   2. POST /rag  {"question": "What is the fridge password?", "session_id": "docs-1"}
#   3. POST /rag  {"question": "Who set it?", "session_id": "docs-1",
#                  "history": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
# =============================================================================
@app.post("/rag", response_model=RagResponse)
def rag(request: RagRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # Rehydrate engine memory from client when provided (survives reload)
        chat_engine = sync_rag_session_history(request.session_id, request.history)
        # chat() → condense question → retrieve top-k → answer with docs + history
        result = chat_engine.chat(question)

        sources = (
            []
            if _is_conversational_query(question)
            else _format_rag_sources(result.source_nodes)
        )
        return RagResponse(answer=result.response, sources=sources)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail="RAG request failed") from exc


@app.delete("/rag/session/{session_id}")
def clear_rag_session(session_id: str):
    """Forget RAG conversation for one session (does not delete Supabase vectors)."""
    rag_sessions.pop(session_id, None)
    return {"cleared": session_id}


@app.post("/rag/rebuild")
def rag_rebuild():
    """
    Re-embed everything in backend/data/ and write fresh vectors to Supabase.

    Call this after editing .md files (no server restart needed).
    View result in Supabase: schema vecs → table ai_chat_docs
    """
    try:
        get_rag_index(force_rebuild=True)
        file_count = len(list(DATA_DIR.glob("*"))) if DATA_DIR.exists() else 0
        return {
            "status": "rebuilt",
            "vector_store": "supabase_pgvector",
            "collection": SUPABASE_COLLECTION,
            "data_dir": str(DATA_DIR),
            "files_seen": file_count,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Rebuild error: {exc}") from exc
