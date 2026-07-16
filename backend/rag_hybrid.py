"""
Hybrid RAG — learning module (separate from main.py).

Compare two RAG styles side by side:

  POST /rag         → LlamaIndex chat engine does retrieval + answer (one library)
  POST /rag-hybrid  → LlamaIndex retrieves chunks, LangChain/Groq writes answer

Per-request flow for /rag-hybrid:
  1. Load session history (optional client `history` rehydrates after reload)
  2. LangChain/Groq condenses follow-ups into a standalone retrieval query
  3. LlamaIndex retriever searches Supabase pgvector (top-k=3)
  4. LangChain/Groq answers using retrieved chunks + chat history
  5. Save turn to hybrid_rag_sessions; return answer, sources, retrieval_query

`retrieval_query` in the response is intentional — inspect what actually got
embedded/searched (useful when the user says "who set it?" after a prior turn).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from fastapi import APIRouter, HTTPException
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from pydantic import BaseModel, Field


# =============================================================================
# REQUEST / RESPONSE MODELS
# =============================================================================
class HistoryMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=8000)


class RagHybridRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=8000)
    session_id: str = Field(default="default", min_length=1, max_length=64)
    # Same pattern as /chat and /rag: Next.js sends prior turns so context survives reload.
    history: list[HistoryMessage] | None = None


class RagHybridResponse(BaseModel):
    answer: str
    sources: list[str]  # chunk previews (reuses main._format_rag_sources via injection)
    retrieval_query: str  # what LlamaIndex actually searched — key for learning/debugging


# =============================================================================
# SESSION MEMORY (in-process; lost on uvicorn restart)
# =============================================================================
# Separate from chat_sessions (/chat) and rag_sessions (/rag) — hybrid has its own.
hybrid_rag_sessions: dict[str, list[BaseMessage]] = {}


# =============================================================================
# PROMPTS — two LangChain/Groq calls per request (when history exists)
# =============================================================================
# Call 1: rewrite "who set it?" → "who set the fridge password?" for better retrieval.
_CONDENSE_PROMPT = """\
Rewrite the latest user question into a standalone retrieval query.

Rules:
- Use prior chat turns only to resolve references like "it", "that file", or "who set it".
- Preserve important names, codes, filenames, and technical terms.
- Return only the rewritten query text.
- If the latest question is already standalone, return it unchanged.
"""

# Call 2: answer using retrieved chunks. {context} is filled from LlamaIndex nodes.
_ANSWER_PROMPT = """\
You are a helpful assistant answering questions about the indexed documents.

Use the retrieved context when it is relevant. If the documents do not contain
the answer, say you do not know instead of inventing facts.

Retrieved context:
---------------------
{context}
---------------------

Original user question: {question}
Retrieval query used: {retrieval_query}
"""


def _history_to_langchain(history: list[HistoryMessage]) -> list[BaseMessage]:
    """Convert API history JSON → LangChain message types for Groq."""
    messages: list[BaseMessage] = []
    for item in history:
        if item.role == "user":
            messages.append(HumanMessage(content=item.content))
        else:
            messages.append(AIMessage(content=item.content))
    return messages


def sync_hybrid_session_history(
    session_id: str, history: list[HistoryMessage] | None
) -> list[BaseMessage]:
    """
    Rehydrate hybrid session from client when `history` is sent.

    If history is omitted (curl without history), keep existing server memory.
    """
    if history is not None:
        hybrid_rag_sessions[session_id] = _history_to_langchain(history)
    return hybrid_rag_sessions.setdefault(session_id, [])


def remember_hybrid_turn(session_id: str, question: str, answer: str) -> None:
    """Append this Q&A pair; trim to last 20 messages to stay within context limits."""
    history = hybrid_rag_sessions.setdefault(session_id, [])
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))
    hybrid_rag_sessions[session_id] = history[-20:]


def _render_history(history: Sequence[BaseMessage]) -> str:
    """Plain-text history (helper for future prompt experiments)."""
    lines: list[str] = []
    for message in history:
        role = "User"
        if isinstance(message, AIMessage):
            role = "Assistant"
        lines.append(f"{role}: {message.content}")
    return "\n".join(lines)


def _context_from_nodes(source_nodes: Sequence[NodeWithScore]) -> str:
    """Join retrieved LlamaIndex nodes into one context block for the answer prompt."""
    parts: list[str] = []
    for idx, node in enumerate(source_nodes, start=1):
        text = node.node.get_content().strip()
        if text:
            parts.append(f"[Chunk {idx}]\n{text}")
    return "\n\n".join(parts)


def _condense_question(
    *,
    get_model: Callable[[], object],
    question: str,
    history: Sequence[BaseMessage],
) -> str:
    """
    LLM call #1 — turn follow-ups into a standalone search query.

    Skip when there is no history: first question is already standalone
    (saves one Groq round-trip).
    """
    if not history:
        return question

    model = get_model()
    response = model.invoke(  # type: ignore[attr-defined]
        [
            SystemMessage(content=_CONDENSE_PROMPT),
            *history[-8:],  # only recent turns — condense does not need full transcript
            HumanMessage(content=question),
        ]
    )
    content = getattr(response, "content", "")
    if isinstance(content, str):
        condensed = content.strip()
    else:
        condensed = str(content).strip()
    return condensed or question


def _answer_question(
    *,
    get_model: Callable[[], object],
    question: str,
    retrieval_query: str,
    history: Sequence[BaseMessage],
    context: str,
) -> str:
    """
    LLM call #2 — LangChain/Groq writes the final grounded answer.

    System prompt holds retrieved chunks; message list holds chat history + question.
    """
    model = get_model()
    response = model.invoke(  # type: ignore[attr-defined]
        [
            SystemMessage(
                content=_ANSWER_PROMPT.format(
                    context=context or "(no relevant chunks retrieved)",
                    question=question,
                    retrieval_query=retrieval_query,
                )
            ),
            *history[-8:],
            HumanMessage(content=question),
        ]
    )
    content = getattr(response, "content", "")
    return content if isinstance(content, str) else str(content)


def create_rag_hybrid_router(
    *,
    get_rag_index: Callable[[], VectorStoreIndex],
    get_model: Callable[[], object],
    format_sources: Callable[[Sequence[NodeWithScore] | None], list[str]],
) -> APIRouter:
    """
    Build FastAPI routes; dependencies come from main.py.

    Uses callables (lambdas in main) so tests can patch get_rag_index / get_model.
    Shared pieces injected from main: index, Groq model, source formatting.
    """
    router = APIRouter()

    @router.post("/rag-hybrid", response_model=RagHybridResponse)
    def rag_hybrid(request: RagHybridRequest) -> RagHybridResponse:
        """
        Hybrid learning path:
          LlamaIndex retrieves chunks
          LangChain/Groq writes the final grounded answer
        """
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        try:
            # --- memory ---
            history = sync_hybrid_session_history(request.session_id, request.history)

            # --- LangChain: condense (optional) ---
            retrieval_query = _condense_question(
                get_model=get_model, question=question, history=history
            )

            # --- LlamaIndex: retrieve only (no chat engine) ---
            retriever = get_rag_index().as_retriever(similarity_top_k=3)
            source_nodes = retriever.retrieve(retrieval_query)
            context = _context_from_nodes(source_nodes)

            # --- LangChain: grounded answer ---
            answer = _answer_question(
                get_model=get_model,
                question=question,
                retrieval_query=retrieval_query,
                history=history,
                context=context,
            )

            remember_hybrid_turn(request.session_id, question, answer)
            return RagHybridResponse(
                answer=answer,
                sources=format_sources(source_nodes),
                retrieval_query=retrieval_query,
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=502, detail="Hybrid RAG request failed"
            ) from exc

    @router.delete("/rag-hybrid/session/{session_id}")
    def clear_hybrid_session(session_id: str) -> dict[str, str]:
        """Clear hybrid conversation memory (does not delete Supabase vectors)."""
        hybrid_rag_sessions.pop(session_id, None)
        return {"cleared": session_id}

    return router
