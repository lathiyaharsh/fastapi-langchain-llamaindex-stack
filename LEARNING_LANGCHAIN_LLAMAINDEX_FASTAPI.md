# Learning Checklist: LangChain / LlamaIndex + FastAPI

Track progress by changing `[ ]` to `[x]` as you complete each item.

**Stack focus**

| Tool | Use for |
| --- | --- |
| FastAPI | HTTP APIs, streaming, validation |
| LangChain | Chains, agents, tools, memory |
| LlamaIndex | Document ingestion, indexing, RAG |

**Goal:** Build a FastAPI backend that can chat (LangChain) and answer from documents (LlamaIndex), wired to the Next.js chat UI in `web/`.

### Endpoints at a glance

| UI mode | Next.js | FastAPI | LLM stack | Tools |
| --- | --- | --- | --- | --- |
| Chat | `POST /api/chat` | `POST /chat/stream` (manual loop) · `POST /chat` (create_agent) | LangChain + Groq | `get_weather` |
| Ask My Docs | `POST /api/rag` | `POST /rag` | LlamaIndex + Groq | none |
| Upload doc | `POST /api/rag/upload` | `POST /rag/upload` | LlamaIndex embed | none |

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

- [x] Define one simple tool (e.g. calculator or time)
- [x] Bind tools to a chat model
- [x] Run a basic agent / tool-calling loop once
- [x] Expose a FastAPI endpoint that uses the tool-aware chain
- [x] Understand when *not* to use an agent (simple Q&A vs multi-step)
- [x] Compare **`create_agent`** (automatic loop) vs **manual `bind_tools` loop** on two endpoints

**Current tool:** `get_weather(location)` on **Chat** only (`/chat`, `/chat/stream`) — not on `/rag`.

How Groq “knows” about the tool (every request):

1. `@tool` on `get_weather` → name, description, `location: str` schema
2. Tools are registered either via **`create_agent(..., tools=...)`** (`/chat`) or **`bind_tools([get_weather])`** (`/chat/stream`)
3. System prompt nudges: “use get_weather for current conditions”
4. Groq may reply with `tool_calls` (not text) → backend runs `fetch_weather()` → Open-Meteo
5. Backend sends `ToolMessage` with result → Groq writes the final natural-language answer

### Two tool-loop patterns (intentional split for learning)

| Endpoint | Pattern | Main code | Who runs the loop? |
| --- | --- | --- | --- |
| `POST /chat` | **`create_agent`** | `get_chat_agent()` → `invoke_chat_with_agent()` | LangChain / LangGraph (automatic) |
| `POST /chat/stream` | **Manual `bind_tools`** | `build_messages()` → `astream_chat_with_tools()` | You (explicit `for` loop, max 5 rounds) |
| Tests only | Manual non-stream | `invoke_chat_with_tools()` | Same as stream, but `invoke` not `astream` |

**Try `create_agent`:** Swagger `/docs` → `POST /chat` with `{"message": "What's the weather in Helsinki?"}`.

**Try manual loop:** Web UI (Chat mode) → Next.js `POST /api/chat` → FastAPI `POST /chat/stream`. Same weather question; ~1–2s blank bubble before first token is normal (tool round + Open-Meteo).

**`GET /health`** reports `chat_nonstream: create_agent` and `chat_stream: manual_bind_tools_loop`.

Key difference: with `create_agent`, you pass `{"messages": [...]}` and read the final `AIMessage` from the result. With the manual loop, you watch `tool_calls` on each `AIMessage`, call `_run_tool_call()`, append `ToolMessage`, and repeat until Groq returns text (or you hit `MAX_CHAT_TOOL_ROUNDS`).

**Streaming UX:** tool round + Open-Meteo HTTP (~1–2s) happens before first visible token — empty bubble + cursor is normal.

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
- [x] Return answer + source snippets in the JSON response (API still returns `sources`; UI hides them)
- [x] Add `POST /rag/rebuild` so editing `data/` updates answers without restart
- [ ] (Optional) Stream RAG answers over SSE
- [x] Persist vectors in Supabase pgvector (survives restart)
- [x] Use `CONDENSE_PLUS_CONTEXT` so chat memory + docs both inform answers
- [x] Sync client-sent `history` on `/rag` for follow-ups after reload
- [x] Add an ingest endpoint to upload new files (`POST /rag/upload`)

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

**When to use `create_agent` vs manual loop**

| Situation | Prefer |
| --- | --- |
| Learning how tool calling works step-by-step | Manual `bind_tools` loop (`/chat/stream`) |
| Production non-stream JSON reply, less boilerplate | `create_agent` (`/chat`) |
| SSE token streaming with tools | Manual loop today (agent streaming is a separate API surface) |
| Unit tests for loop logic | `invoke_chat_with_tools()` (kept alongside agent path) |

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
| Weak RAG hits in API `sources` | `_format_rag_sources` keeps chunks within score gap of best match |
| `CONDENSE_QUESTION` forgot chat facts (name, etc.) | Switched to `CONDENSE_PLUS_CONTEXT` + custom `RAG_CONTEXT_PROMPT` |
| `SimilarityPostprocessor(0.5)` broke all RAG answers | Removed — Supabase scores are lower-is-better, not cosine similarity |
| Sources panel cluttered Docs UI | Removed Sources UI in `FastApiChat.tsx`; API still returns `sources` for curl/Swagger |
| Streamed text had no spaces (`Thecurrentweather…`) | BFF `parseSseDataLine` — do not `.trim()` SSE payloads; spaces are real tokens |
| Tool questions: blank bubble ~2s before first token | Expected — Groq tool round + Open-Meteo before final answer streams |
| Compare agent vs manual tool loop | `/chat` = `create_agent`; `/chat/stream` = manual loop; UI still uses stream only |

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

> Note: Next.js rate-limits at `/api/*`. FastAPI errors are sanitized for chat/RAG streams. Backend `unittest` covers health, history sync, weather tool, **`invoke_chat_with_agent`**, manual tool loop, source filtering, session clear, URL normalize, upload.

---

## Mini projects (mark when done)

- [x] **Project A:** FastAPI `/chat` with LangChain streaming (no tools)
- [x] **Project B:** Same chat + one tool (`get_weather` via Open-Meteo)
- [x] **Project C:** FastAPI `/rag` over a folder of markdown notes
- [x] **Project D:** Next.js UI → FastAPI backend (full loop)
- [x] **Project E:** RAG answers include source snippets in API (UI sources panel removed later)

---

## Concepts checklist (can you explain each?)

- [x] Prompt template vs system prompt
- [x] Tokens and context window
- [x] Streaming vs non-streaming
- [x] Embeddings and vector similarity
- [x] Chunking trade-offs
- [x] Hallucination vs grounded RAG answers
- [x] Tool calling / function calling (`bind_tools`, `ToolMessage`, tool loop)
- [x] Agents vs plain chains (`/chat` = `create_agent`; `/chat/stream` = manual tool loop)
- [x] Sync vs async FastAPI endpoints (`/rag/upload` is sync `def` — LlamaIndex + event loop conflict if `async`)
- [x] Why keep secrets server-side only
- [x] Client `history` vs server session memory (who is source of truth?)

---

## Concepts deep-dive

### 1. Prompt template vs system prompt

Both shape what the model does, but at different layers.

| Concept | What it is | In this project |
| --- | --- | --- |
| **System prompt** | Standing instructions for the whole conversation — role, rules, tone | `chat_system_prompt()` → passed to `create_agent(..., system_prompt=...)` on `/chat`, or as a `SystemMessage` in `build_messages()` on `/chat/stream` |
| **Prompt template** | A reusable pattern with **slots** you fill per request | `RAG_CONTEXT_PROMPT` — LlamaIndex fills `{context_str}` with retrieved chunks; `reply_mode` fills concise vs detailed style in chat |

**Mental model**

- **System prompt** = “who you are and how to behave” (stable across turns)
- **Prompt template** = “this turn’s layout” (variables change each call)

**Two chat paths, same system text, different wiring**

```
/chat (create_agent):
  system_prompt on agent  +  messages = [history, HumanMessage]

/chat/stream (manual loop):
  messages = [SystemMessage(system_prompt), *history, HumanMessage]
```

**RAG uses a template, not a separate system message**

`RAG_CONTEXT_PROMPT` is a `context_prompt` for `CONDENSE_PLUS_CONTEXT`. LlamaIndex builds the final prompt roughly as:

```
[condensed question for retrieval]
+ retrieved chunks → {context_str}
+ chat history
+ user question
```

So chat = LangChain message types; RAG = LlamaIndex template with `{context_str}`.

**Rule of thumb:** put durable rules in the system prompt / template header; put turn-specific content in human messages or template variables.

---

### 2. Tokens and context window

A **token** is a small piece of text the model reads and writes (not always a whole word — `"ing"` might be one token).

The **context window** is the max tokens in one API call: everything you send **in** plus everything the model generates **out**.

**What counts toward context in this stack**

| Piece | Endpoint | Notes |
| --- | --- | --- |
| System prompt | `/chat`, `/chat/stream` | Weather rules + concise/detailed style |
| Tool schemas | `/chat`, `/chat/stream` | `get_weather` name, description, args — sent every request |
| Chat history | `/chat`, `/chat/stream` | Capped at **last 20 messages** in `remember()` |
| User message | all | Up to 8000 chars (Pydantic limit) |
| Retrieved chunks | `/rag` | `similarity_top_k=3` chunks in `RAG_CONTEXT_PROMPT` |
| Tool round-trip | `/chat` | Extra `AIMessage` (tool_calls) + `ToolMessage` before final answer |

**Why Groq usage can look high (~100k+ input over a session)**

- Every turn resends full history + system + tools
- RAG adds embedded chunk text to the prompt
- Tool calls add messages the user never sees
- Long “Detailed” replies cost more output tokens

**What we already do to stay in bounds**

- `remember()` trims to 20 messages (~10 turns)
- `similarity_top_k=3` limits RAG context
- Concise mode nudges shorter answers

**If context overflows:** older turns get dropped (chat trim) or the provider returns an error. Mitigations: summarize old history, lower `top_k`, smaller chunks, or a model with a larger window.

---

### 3. Chunking trade-offs

**Chunking** = splitting documents into smaller pieces before embedding and storing in the vector DB.

Flow in this project: `SimpleDirectoryReader` → `VectorStoreIndex.from_documents()` → LlamaIndex default splitter → HF embeddings → Supabase pgvector → retrieve top-k at query time.

**Why chunk at all?**

- Embeddings work best on focused passages, not whole files
- Retrieval returns only relevant pieces, not entire docs
- Fits more diverse sources into the context window

**Trade-offs**

| Smaller chunks | Larger chunks |
| --- | --- |
| More precise retrieval for narrow facts | More surrounding context per hit |
| Risk: answer needs info split across chunks | Risk: irrelevant filler dilutes the prompt |
| More rows in Supabase | Fewer rows, cheaper storage |

**Knobs in this project (today vs future)**

| Knob | Current value | Effect |
| --- | --- | --- |
| LlamaIndex default splitter | implicit in `from_documents()` | ~1024-token chunks, small overlap (library default) |
| `similarity_top_k` | `3` | How many chunks enter `RAG_CONTEXT_PROMPT` |
| `RAG_SOURCE_SCORE_GAP` | `0.08` | Filters weak source previews in API response |
| `HF_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding quality vs speed/cost |

**Symptoms and fixes**

| Symptom | Likely cause | Try |
| --- | --- | --- |
| “I don’t know” but answer is in the doc | Chunk boundary split the fact | Smaller chunks + overlap, or merge related sections before ingest |
| Wrong / vague RAG answers | Chunk too big or weak retrieval | Lower chunk size, raise `top_k` slightly, tune score gap |
| Irrelevant doc snippets in `sources` | Broad chunk retrieved | Tighter chunks, lower `top_k`, stricter score gap |

**Chunking is not one-size-fits-all** — tune for your doc shape (short notes vs long manuals). For learning, defaults are fine; Phase 4 hybrid RAG is a good place to experiment with explicit `SentenceSplitter(chunk_size=..., chunk_overlap=...)`.

---

## Resources (fill in as you use them)

- [ ] FastAPI docs — https://fastapi.tiangolo.com/
- [x] LangChain **Tools** — https://docs.langchain.com/oss/python/langchain/tools
- [x] LangChain **Agents** (`create_agent`) — https://docs.langchain.com/oss/python/langchain/agents
- [x] LangChain **Tool calling** — https://docs.langchain.com/oss/python/langchain/tool-calling (optional deepen)
- [x] LlamaIndex **Introduction to RAG** — https://docs.llamaindex.ai/en/stable/understanding/rag/
- [ ] LlamaIndex chat engines / vector stores — explore from Understanding hub
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
| 2026-07-15 | `POST /rag/upload` + UI Upload doc button | Saves to `backend/data/`, embeds/inserts only the new file |
| 2026-07-15 | Project B: tool calling on `/chat` + `/chat/stream` | `bind_tools` + `invoke_chat_with_tools` / `astream_chat_with_tools` |
| 2026-07-15 | Replaced calculator with `get_weather` (Open-Meteo) | Groq decides when to call; backend runs HTTP; no extra API key |
| 2026-07-15 | Removed Sources panel from Docs UI | Simpler UX; dropped per-query source-hiding logic in `/rag` |
| 2026-07-15 | Fixed SSE space stripping in BFF | `parseSseDataLine` — preserve leading spaces in streamed tokens |
| 2026-07-15 | Comment pass on `backend/main.py` | Documented tool loop, streaming delay, history sync |
| 2026-07-16 | Split chat tool paths: `/chat` = `create_agent`, `/chat/stream` = manual loop | Same `get_weather` tool; compare via Swagger vs UI; `invoke_chat_with_tools` kept for tests |
| 2026-07-16 | Concepts deep-dive: prompt template vs system prompt, tokens/context, chunking | Mapped each concept to `main.py` — system prompt, `RAG_CONTEXT_PROMPT`, `remember()` trim, default LlamaIndex splitter |

---

## Current focus

> **Done:** Concepts checklist complete. `/chat` = `create_agent`; `/chat/stream` = manual loop.

**Next options**

1. **Hybrid RAG (Phase 4)** — LlamaIndex retrieve → LangChain/Groq answer in one flow
2. **Optional UI toggle** — call `POST /chat` from the browser to try `create_agent` without Swagger
3. **Chunking experiment** — explicit `SentenceSplitter` + compare RAG quality on your `data/` files
4. **Phase 6** — logging, rate limits, LLM timeouts (when you want production polish)
