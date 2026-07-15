# Learning Checklist: LangChain / LlamaIndex + FastAPI

Track progress by changing `[ ]` to `[x]` as you complete each item.

**Stack focus**

| Tool | Use for |
| --- | --- |
| FastAPI | HTTP APIs, streaming, validation |
| LangChain | Chains, agents, tools, memory |
| LlamaIndex | Document ingestion, indexing, RAG |

**Goal:** Build a FastAPI backend that can chat (LangChain) and answer from documents (LlamaIndex), wired to the Next.js chat UI in `web/`.

---

## Phase 0 — Setup

- [x] Create a Python virtualenv (`python -m venv .venv`)
- [x] Activate the venv and upgrade pip
- [x] Install FastAPI + Uvicorn (`fastapi`, `uvicorn[standard]`)
- [x] Install LangChain core packages (`langchain`, `langchain-core`, `langchain-community`)
- [x] Install one chat model provider package (e.g. `langchain-groq` or `langchain-openai`)
- [x] Install LlamaIndex (`llama-index`)
- [x] Copy API keys into a `.env` (never commit secrets)
- [x] Run a “hello” FastAPI app on `http://127.0.0.1:8000`
- [x] Open `/docs` (Swagger UI) and call a sample endpoint

---

## Phase 1 — FastAPI fundamentals

- [x] Understand path operations (`@app.get`, `@app.post`)
- [x] Define a Pydantic request model (e.g. `ChatRequest`)
- [x] Define a Pydantic response model (e.g. `ChatResponse`)
- [x] Return JSON from a POST `/chat` endpoint
- [x] Add CORS so a Next.js frontend can call the API
- [x] Load settings from environment variables
- [x] Add basic error handling (`HTTPException`)
- [x] Add request validation (message length / count limits)
- [x] Write a health check endpoint `GET /health`
- [ ] (Optional) Split routers: `routers/chat.py`, `routers/rag.py`

### Streaming (match the Next.js chat UX)

- [x] Understand SSE (Server-Sent Events) conceptually
- [x] Return a `StreamingResponse` that yields tokens
- [x] Stream plain text chunks from a dummy generator
- [x] Stream events in a format your frontend can parse
- [x] Handle client disconnect / cancellation

---

## Phase 2 — LangChain basics

- [x] Call a chat model directly (no chain yet)
- [x] Use `ChatPromptTemplate` with system + human messages
- [x] Build a simple LCEL chain: `prompt | model | parser`
- [x] Understand `invoke` vs `stream` vs `ainvoke` / `astream`
- [x] Parse string output with `StrOutputParser`
- [x] Pass conversation history into the prompt
- [x] Add concise vs detailed reply style via prompt variables
- [x] Wire LangChain streaming into FastAPI `StreamingResponse`
- [ ] Log latency and token/provider errors in a readable way

### Memory

- [x] Understand chat message types (`SystemMessage`, `HumanMessage`, `AIMessage`)
- [x] Keep per-session history in memory (dict / store)
- [x] Clear session history via an API endpoint
- [x] Sync client-sent `history` so context survives reload (Next.js + localStorage)
- [ ] (Optional) Persist history (Redis / SQLite / file)

### Tools & agents (intro)

- [ ] Define one simple tool (e.g. calculator or time)
- [ ] Bind tools to a chat model
- [ ] Run a basic agent / tool-calling loop once
- [ ] Expose a FastAPI endpoint that uses the tool-aware chain
- [ ] Understand when *not* to use an agent (simple Q&A vs multi-step)

---

## Phase 3 — LlamaIndex + RAG

- [x] Understand RAG: load → chunk → embed → index → retrieve → generate
- [x] Load local documents (`.md`, `.txt`, or PDF)
- [x] Split / chunk documents
- [x] Create a vector index (start with local / simple store)
- [x] Query the index with a natural-language question
- [x] Inspect retrieved source nodes / citations
- [x] Tune top-k retrieval and filter weak source hits (`similarity_top_k=3`, score-gap filter)
- [x] Wrap RAG in a FastAPI `POST /rag` endpoint
- [x] Return answer + source snippets in the JSON response
- [x] Add `POST /rag/rebuild` so editing `data/` updates answers without restart
- [ ] (Optional) Stream RAG answers over SSE
- [x] Persist vectors in Supabase pgvector (survives restart)
- [x] Use `CONDENSE_PLUS_CONTEXT` so chat memory + docs both inform answers
- [x] Sync client-sent `history` on `/rag` for follow-ups after reload
- [ ] (Optional) Add an ingest endpoint to upload new files

---

## Phase 4 — LangChain vs LlamaIndex (know the split)

- [x] Can explain: LangChain = orchestration / agents / tools
- [x] Can explain: LlamaIndex = data ingestion / retrieval / RAG
- [ ] Build one flow that uses **LlamaIndex for retrieval** and **LangChain (or plain LLM) for answering**
- [x] Document in notes: which library you prefer for which task

### Your notes (filled in during learning)

| Need | Prefer |
| --- | --- |
| Chat API, memory, streaming, tools/agents | **LangChain** (what `/chat` uses) |
| Load docs, embed, search, grounded Q&A | **LlamaIndex** (what `/rag` uses) |
| FastAPI | HTTP layer for both |
| Later hybrid | LlamaIndex retrieves chunks → LangChain/LLM writes the answer |

**RAG chat engine choice:** `CONDENSE_QUESTION` only uses retrieved docs in the final answer — fine for doc Q&A, bad for “what is my name?” after an intro. **`CONDENSE_PLUS_CONTEXT`** condenses follow-ups for retrieval *and* passes chat history into the final prompt.

**Supabase score gotcha:** `SupabaseVectorStore` scores are `~1 - exp(-distance)` (lower = better). Do **not** use `SimilarityPostprocessor` with a “min similarity” cutoff — it drops the best matches and returns `Empty Response`.

---

## Phase 5 — Integrate with this AI-Chat project

- [x] Keep Next.js as the UI only
- [x] Point the frontend chat API to FastAPI (`FASTAPI_URL` server-side BFF)
- [x] Match existing message payload shape (`messages`, provider, reply mode)
- [x] Match SSE event shape if you already use streaming on the client
- [x] Support stop/cancel from the UI
- [x] Add a UI mode or route for “Ask my docs” (RAG)
- [x] Confirm CORS, errors, and rate limiting still feel correct
- [x] Write a short note in this file about what broke and how you fixed it

### Phase 5 integration notes

| Issue | Fix |
| --- | --- |
| FastAPI SSE was plain text (`data: token`); UI expected JSON (`type/chunk/done`) | Next.js BFF in `lib/api/fastapi-client.ts` translates chunks |
| FastAPI wanted single `message` + server memory; UI sends full `messages[]` | Adapter sends last user turn + `history`; FastAPI `sync_session_history` |
| RAG lost context after uvicorn reload | Optional `history` on `/rag`; `sync_rag_session_history` rehydrates engine memory |
| Stop mid-stream still saved assistant text | `/chat/stream` checks `request.is_disconnected()` and skips `remember()` |
| RAG non-streaming | `/api/rag` emits synthetic JSON SSE with `sources` on chunk/done |
| Dual sessions | Separate `chat_*` / `docs_*` session IDs in localStorage; Clear only deletes active mode |
| Greetings showed irrelevant doc sources | `_is_conversational_query` hides sources for hi / intros / “what is my name?” |
| Weak RAG hits listed as sources (e.g. project-notes for password) | `_format_rag_sources` keeps chunks within score gap of best match |
| `CONDENSE_QUESTION` forgot chat facts (name, etc.) | Switched to `CONDENSE_PLUS_CONTEXT` + custom `RAG_CONTEXT_PROMPT` |
| `SimilarityPostprocessor(0.5)` broke all RAG answers | Removed — Supabase scores are lower-is-better, not cosine similarity |

---

## Phase 6 — Quality & production basics

- [ ] Add structured logging
- [ ] Add rate limiting on chat / rag routes
- [ ] Add timeouts for LLM calls
- [x] Hide raw provider errors from clients (friendly messages)
- [x] Write at least 2–3 unit tests (validation, health, one service function)
- [x] Add a `requirements.txt` (or `pyproject.toml`) with pinned versions
- [x] Add a short `backend/README.md` with run instructions
- [x] Configure Pyright/basedpyright to use `backend/.venv` (fixes “import could not be resolved”)
- [ ] (Optional) Dockerize the FastAPI service

> Note: Next.js rate-limits at `/api/*`. FastAPI errors are sanitized for chat/RAG streams. Backend `unittest` covers health, history sync, conversational query detection, source filtering, session clear, URL normalize.

---

## Mini projects (mark when done)

- [x] **Project A:** FastAPI `/chat` with LangChain streaming (no tools)
- [ ] **Project B:** Same chat + one tool (search or calculator)
- [x] **Project C:** FastAPI `/rag` over a folder of markdown notes
- [x] **Project D:** Next.js UI → FastAPI backend (full loop)
- [x] **Project E:** RAG answers include clickable/citable sources

---

## Concepts checklist (can you explain each?)

- [ ] Prompt template vs system prompt
- [ ] Tokens and context window
- [x] Streaming vs non-streaming
- [x] Embeddings and vector similarity
- [ ] Chunking trade-offs
- [x] Hallucination vs grounded RAG answers
- [ ] Tool calling / function calling
- [ ] Agents vs plain chains
- [ ] Sync vs async FastAPI endpoints
- [x] Why keep secrets server-side only
- [x] Client `history` vs server session memory (who is source of truth?)

---

## Resources (fill in as you use them)

- [ ] FastAPI docs — https://fastapi.tiangolo.com/
- [ ] LangChain docs — https://python.langchain.com/
- [ ] LlamaIndex docs — https://docs.llamaindex.ai/
- [ ] Your notes / blog / loom links: _add here_

---

## Progress log

| Date | What I finished | Blockers / learnings |
| --- | --- | --- |
| 2026-07-14 | Phase 0: venv, packages, .env, hello FastAPI + /docs | requirements.txt so others can reproduce |
| 2026-07-14 | Phase 1: POST /chat, CORS, .env, HTTPException, SSE stream | Echo + fake token stream before LLM |
| 2026-07-14 | Phase 2: Groq chain, astream, session memory | Remember name across turns |
| 2026-07-14 | Phase 3 core: POST /rag + sources + comments | First call slow; edit .md needs index rebuild |
| 2026-07-14 | POST /rag/rebuild + Phase 4 split notes | LangChain=chat; LlamaIndex=docs |
| 2026-07-15 | Phase 5: Next.js BFF → FastAPI for Groq + Ask My Docs UI | SSE translation + history sync were the hard parts |
| 2026-07-15 | RAG: `CONDENSE_PLUS_CONTEXT`, client `history`, source filtering | `CONDENSE_QUESTION` ignores chat memory; Supabase scores are lower-is-better |
| 2026-07-15 | Pyright config + comment/doc cleanup | Point venv at `backend/.venv`; don't use SimilarityPostprocessor with Supabase |

---

## Current focus

> Phase 5 done (hybrid Groq via FastAPI, Docs mode, RAG polish). Optional next: tools/agents (Project B) or Phase 6 logging/timeouts.
