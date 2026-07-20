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
- [x] (Optional) Split routers: `routers/chat.py`, `routers/rag.py`

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
- [x] Build one flow that uses **LlamaIndex for retrieval** and **LangChain (or plain LLM) for answering**
- [x] Document in notes: which library you prefer for which task

### Your notes (filled in during learning)

| Need | Prefer |
| --- | --- |
| Chat API, memory, streaming, tools/agents | **LangChain** (what `/chat` uses) |
| Load docs, embed, search, grounded Q&A | **LlamaIndex** (what `/rag` uses) |
| FastAPI | HTTP layer for both |
| Hybrid compare path | LlamaIndex retrieves chunks → LangChain/LLM writes the answer (`/rag-hybrid`) |

**When to use `create_agent` vs manual loop**

| Situation | Prefer |
| --- | --- |
| Learning how tool calling works step-by-step | Manual `bind_tools` loop (`/chat/stream`) |
| Production non-stream JSON reply, less boilerplate | `create_agent` (`/chat`) |
| SSE token streaming with tools | Manual loop today (agent streaming is a separate API surface) |
| Unit tests for loop logic | `invoke_chat_with_tools()` (kept alongside agent path) |

**RAG chat engine choice:** `CONDENSE_QUESTION` only uses retrieved docs in the final answer — fine for doc Q&A, bad for “what is my name?” after an intro. **`CONDENSE_PLUS_CONTEXT`** condenses follow-ups for retrieval *and* passes chat history into the final prompt.

**Supabase score gotcha:** `SupabaseVectorStore` scores are `~1 - exp(-distance)` (lower = better). Do **not** use `SimilarityPostprocessor` with a “min similarity” cutoff — it drops the best matches and returns `Empty Response`.

**Now you have both RAG styles to compare**

| Endpoint | Retrieval | Answer orchestration | Good for learning |
| --- | --- | --- | --- |
| `/rag` | LlamaIndex | LlamaIndex chat engine + Groq | End-to-end RAG with less code |
| `/rag-hybrid` | LlamaIndex retriever | LangChain/Groq prompt + answer | Clear split: retrieve with one library, answer with another |

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
| `main.py` getting too large | Added `backend/rag_hybrid.py` for the new hybrid compare endpoint |
| Hybrid flow was harder to follow than `/rag` | Added learning comments in `backend/rag_hybrid.py` for memory, condense, retrieve, and answer steps |

---

## Phase 6 — Quality & production basics

- [x] Add structured logging
- [x] Add rate limiting on chat / rag routes
- [x] Add timeouts for LLM calls
- [x] Hide raw provider errors from clients (friendly messages)
- [x] Write at least 2–3 unit tests (validation, health, one service function)
- [x] Add a `requirements.txt` (or `pyproject.toml`) with pinned versions
- [x] Add a short `backend/README.md` with run instructions
- [x] Configure Pyright/basedpyright to use `backend/.venv` (fixes “import could not be resolved”)
- [ ] (Optional) Dockerize the FastAPI service

> Note: Next.js rate-limits at `/api/*`. FastAPI also rate-limits `/chat*` + `/rag*` (`RATE_LIMIT_PER_MINUTE`). Errors are sanitized (502/504). Request logs + `X-Request-Id` in `ops.py`. Backend `unittest` covers health, history sync, weather tool, **`invoke_chat_with_agent`**, manual tool loop, source filtering, session clear, URL normalize, upload, **rate limit 429**, timeout helper.

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

Flow in this project: `SimpleDirectoryReader` → `SentenceSplitter(512, 64)` → embed → Supabase pgvector → retrieve top-k at query time.

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

**Knobs in this project**

| Knob | Current value | Effect |
| --- | --- | --- |
| `RAG_CHUNK_SIZE` | `512` | `SentenceSplitter` max chunk size |
| `RAG_CHUNK_OVERLAP` | `64` | Shared text between neighbor chunks |
| `similarity_top_k` | `3` | How many chunks enter `RAG_CONTEXT_PROMPT` |
| `RAG_SOURCE_SCORE_GAP` | `0.08` | Filters weak source previews in API response |
| `HF_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding quality vs speed/cost |

**Symptoms and fixes**

| Symptom | Likely cause | Try |
| --- | --- | --- |
| “I don’t know” but answer is in the doc | Chunk boundary split the fact | Smaller chunks + overlap, or merge related sections before ingest |
| Wrong / vague RAG answers | Chunk too big or weak retrieval | Lower chunk size, raise `top_k` slightly, tune score gap |
| Irrelevant doc snippets in `sources` | Broad chunk retrieved | Tighter chunks, lower `top_k`, stricter score gap |

**Chunking is not one-size-fits-all** — tune for your doc shape (short notes vs long manuals). Change `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` then `POST /rag/rebuild`.

---

## Resources (fill in as you use them)

- [ ] FastAPI docs — https://fastapi.tiangolo.com/
- [x] LangChain **Tools** — https://docs.langchain.com/oss/python/langchain/tools
- [x] LangChain **Agents** (`create_agent`) — https://docs.langchain.com/oss/python/langchain/agents
- [x] LangChain **Tool calling** — https://docs.langchain.com/oss/python/langchain/tool-calling (optional deepen)
- [x] LlamaIndex **Introduction to RAG** — https://docs.llamaindex.ai/en/stable/understanding/rag/
- [ ] LlamaIndex chat engines / vector stores — explore from Understanding hub
- [x] Neural Networks beginner guide — https://www.geeksforgeeks.org/deep-learning/neural-networks-a-beginners-guide/
- [x] Google **Introduction to Machine Learning** — https://developers.google.com/machine-learning/intro-to-ml/
- [ ] Your notes / blog / loom links: _add here_

---

## ML fundamentals notes (from GFG beginner guide)

**Neural network in one line:** layers of connected neurons that learn patterns from data by adjusting **weights** and **biases** — not by hard-coded `if` rules.

**Building blocks learned**

| Piece | Meaning |
| --- | --- |
| Neuron | Takes inputs → weighted sum + bias → activation → output |
| Weights / biases | Learnable knobs that control how strong each connection is |
| Input → Hidden → Output layers | Data in → feature processing → prediction / text / label out |
| Forward propagation | Data flows input → output to make a prediction |
| Loss | How wrong the prediction was |
| Backpropagation | Compute gradients; update weights to reduce loss |
| Activation (ReLU, sigmoid, tanh) | Adds non-linearity so the net can learn complex patterns |

**Types to remember later**

| Type | Typical use |
| --- | --- |
| Feedforward / MLP | Tabular / simple classification |
| CNN | Images |
| RNN / LSTM | Sequences (older NLP) |
| Transformers | Modern NLP / LLMs (what Groq models are based on) |

**How this maps to your FastAPI stack**

| In the guide | In your project |
| --- | --- |
| Network learns patterns from data | Groq LLM already trained; you call it via LangChain |
| Forward pass → output | Prompt in → tokens / answer out (`/chat`, `/rag`, `/rag-hybrid`) |
| Feature vector example (email spam) | Embedding model turns text into a vector for Supabase search |
| Supervised learning idea | Embeddings + RAG use pretrained models; you don't train from scratch here |

You are **using** neural nets (LLM + embeddings), not training them in this repo — still useful to know what sits under Groq / HF.

### Explored further (Google Intro to ML + NLP stack basics)

Sources:
- [Google Introduction to Machine Learning](https://developers.google.com/machine-learning/intro-to-ml/)
- [GFG Neural Networks beginner guide](https://www.geeksforgeeks.org/deep-learning/neural-networks-a-beginners-guide/)
- Plus topic exploration: **Dataset preparation & tokenization**, **word embeddings**, **Transformer architecture**

| Topic | What it means | In your FastAPI stack |
| --- | --- | --- |
| **Dataset prep** | Clean/split/label data before training | You skip training; prep = `backend/data/*.md`, upload, `/rag/rebuild` |
| **Tokenization** | Split text into model tokens (subwords) | Groq counts/bills tokens; context window = token budget; SSE streams tokens |
| **Word embeddings** | Tokens/words → dense vectors | RAG: HF `bge-small` → 384-d vectors in Supabase; LLM: internal token embeddings |
| **Transformer architecture** | Attention + FFN blocks (modern NLP) | Groq Llama / embed models — see RAG Parts 2–3 in this file |
| **Google Intro to ML** | Features, labels, train vs predict, overfitting basics | Mental model for “pretrained model + your prompt/data” vs training from scratch |

**Checklist**

- [x] Dataset preparation & tokenization (explored)
- [x] Word embeddings (explored + mapped to RAG vectors)
- [x] Transformer architecture (explored + Parts 2–3 notes)
- [x] Google Intro to ML path
- [x] GFG neural networks beginner guide

---

## RAG internals deep-dive (embeddings → Transformer pieces)

### Part 1 — Embeddings & vector search (what `/rag` actually does)

**Embedding** = turn text into a fixed-length list of numbers (a **vector**) so similar meaning → nearby vectors.

| In this project | Value |
| --- | --- |
| Model | `BAAI/bge-small-en-v1.5` via HF Inference API |
| Dimension | **384** (`HF_EMBED_DIM`) — must match Supabase collection |
| Store | Supabase **pgvector** (`vecs.ai_chat_docs`) |
| Retrieve | `similarity_top_k=3` |

**Ingest (once / on rebuild / upload)**

```text
.md file → chunk → embed each chunk → store [vector + text] in Supabase
```

**Query (every `/rag` or `/rag-hybrid` ask)**

```text
question → embed question → find nearest chunk vectors → return text → LLM answers
```

**Why vectors?** Keyword search misses “fridge code” vs “fridge password”. Embedding search matches **meaning**.

**Similarity (intuition)**

- Cosine similarity / distance: how “aligned” two vectors are
- Your Supabase store reports scores as `~1 - exp(-distance)` → **lower = better**
- That’s why `SimilarityPostprocessor(min_similarity=0.5)` broke RAG earlier

**Checklist for Part 1**

- [x] Embedding = meaning as numbers
- [x] Dim must match store (384)
- [x] Ingest embeds docs; query embeds question; nearest neighbors = retrieval
- [x] `top_k` = how many chunks enter the LLM prompt

### Part 2 — Transformer building blocks (what Groq’s LLM uses)

Modern LLMs (and many embedders) are **Transformers**. Four ideas:

| Block | Job in one line |
| --- | --- |
| **Token embeddings** | Words/subwords → vectors (like RAG embeddings, but inside the LLM) |
| **Positional encoding** | Inject *order* (“who” before “set”) — without this, bag-of-words |
| **Multi-head attention** | Each token looks at other tokens; many “heads” = many viewpoints |
| **Feed-forward network (FFN)** | Per-token MLP that mixes features after attention |

**Stack of one Transformer block (simplified)**

```text
tokens
  → embed + position
  → Multi-Head Attention  (who relates to whom?)
  → Feed-Forward          (process each token’s features)
  → … repeat many layers …
  → predict next token
```

#### 2a. Positional encoding — why order matters

Attention alone is **permutation-invariant**: without position info,  
`"dog bites man"` and `"man bites dog"` look the same as bags of tokens.

**Positional encoding** adds a position signal to each token embedding:

```text
token_vector + position_vector  →  model knows "who" is at index 0, "set" at index 1, …
```

Classic paper used sine/cosine patterns; modern LLMs often use **RoPE** (rotary) or learned positions — same goal: encode *where* in the sequence.

**In your app:** chat history order, system prompt before user message, and retrieved chunks in `RAG_CONTEXT_PROMPT` all rely on position so Groq can tell “User said X, then Assistant said Y.”

#### 2b. Attention — Query / Key / Value

For each token, the model builds three vectors from its embedding:

| Name | Intuition | Analogy |
| --- | --- | --- |
| **Query (Q)** | “What am I looking for?” | Search query |
| **Key (K)** | “What do I contain / advertise?” | Document title / index |
| **Value (V)** | “What content do I give if selected?” | Document body |

**Scaled dot-product attention (one head):**

```text
scores = Q · Kᵀ          # how much each token matches each other token
weights = softmax(scores) # turn into probabilities that sum to 1
output  = weights · V     # blend the Values using those weights
```

Example: when generating an answer about the fridge password, tokens in the *question* get high attention weights on tokens in the *retrieved chunk* that contain `BANANA-42`.

#### 2c. Multi-head attention — why more than one head

One head = one way of relating tokens. **Multi-head** runs several attentions in parallel, then concatenates:

| Head might specialize in… | Example |
| --- | --- |
| Syntax | subject ↔ verb |
| Coreference | “it” ↔ “fridge password” |
| Copying facts | question ↔ retrieved chunk numbers/codes |

```text
Head1(Q,K,V) ‖ Head2(Q,K,V) ‖ … ‖ HeadH(Q,K,V)  →  linear mix → output
```

More heads ≈ more viewpoints in one layer. Big models (70B) have many heads × many layers.

#### 2d. Feed-forward network (FFN)

After attention mixes information *across* tokens, an **FFN** processes *each token alone*:

```text
for each token independently:
  x → Linear → activation (e.g. GELU/ReLU) → Linear → x'
```

Think: attention = “who should I listen to?”; FFN = “given what I heard, update my features.”  
Often the FFN is where a lot of “factual / lexical” capacity lives (huge middle dimension).

#### 2e. RAG embeddings vs LLM token embeddings

| | RAG embedding (`bge-small`) | LLM token embedding (Groq) |
| --- | --- | --- |
| Input | Whole chunk or question | Each subword/token |
| Output | One 384-d vector | Sequence of vectors |
| Job | Find similar docs in pgvector | Start of generating / understanding the prompt |
| Stored? | Yes — Supabase | No — computed per request inside the model |

Same family of idea (text → vectors); different place in the pipeline.

**How this connects to RAG**

| Piece | Your stack |
| --- | --- |
| Embedding model | Separate small Transformer → vectors for pgvector |
| Groq LLM | Large Transformer → reads prompt (system + history + retrieved chunks) → generates answer tokens |
| Attention (intuition) | Model “pays attention” to fridge password chunk when answering “what’s the code?” |
| Position | History + chunk order in the prompt stay meaningful |
| FFN | Refines each token’s representation before next layer / next-token prediction |

**Full mental model (your `/rag` request)**

```text
1. Embed question          ← small embedding model (Part 1)
2. Vector search top-k     ← pgvector
3. Build prompt            ← system + chunks + history + question
4. Groq Transformer:       ← Part 2
     embed tokens + positions
     → [attention + FFN] × many layers
     → next-token probabilities
5. Stream / return answer
```

**Checklist for Part 2**

- [x] Positional encoding: why order matters
- [x] Attention: Query / Key / Value intuition
- [x] Multi-head: why more than one head
- [x] FFN: what happens after attention
- [x] Embeddings (RAG) vs token embeddings (LLM): related but different jobs

### Part 3 — Next-token prediction, temperature, context as working memory

#### 3a. Next-token prediction (what “generation” really is)

An LLM does **not** invent a full paragraph in one shot. It repeatedly:

```text
prompt tokens  →  Transformer  →  probability for every token in vocabulary
               →  pick one token
               →  append it to the prompt
               →  repeat until stop / max length
```

Example (simplified):

```text
"The fridge password is"
  → model: P("BANANA") high, P("apple") low, …
  → pick "BANANA"
"The fridge password is BANANA"
  → pick "-42"
→ …
```

Your `/chat/stream` SSE is exactly this loop made visible: **one (or a few) tokens at a time**.

Training (Karpathy talk): predict the next token on huge text → weights learn language patterns.  
Inference (your API): freeze weights; only run the forward pass + sampling.

#### 3b. Temperature — how “random” the pick is

After the Transformer outputs logits (raw scores), they become probabilities. **Temperature** scales those scores before softmax:

| Temperature | Behavior | When |
| --- | --- | --- |
| **Low** (≈0–0.3) | Peakier probs → more deterministic / focused | Facts, codes, tool JSON |
| **Medium** (≈0.7) | Balanced | Default via `GROQ_TEMPERATURE` / `get_model()` |
| **High** (≈1.0+) | Flatter probs → more creative / varied / riskier | Brainstorming |

```text
probs = softmax(logits / temperature)
```

**Apply (done):** `POST /chat` accepts optional `temperature` (0–2). Omit → env default `0.7`.

```bash
curl -s -X POST http://127.0.0.1:8000/chat -H 'Content-Type: application/json' \
  -d '{"message":"Write three slogans for a password fridge","temperature":0.0,"session_id":"t0"}'
curl -s -X POST http://127.0.0.1:8000/chat -H 'Content-Type: application/json' \
  -d '{"message":"Write three slogans for a password fridge","temperature":1.0,"session_id":"t1"}'
```

Temperature does **not** change what the model “knows” — only how boldly it samples. RAG still matters for facts; low temp alone won’t invent a doc that wasn’t retrieved.

Related knobs (you may see later): `top_p` / `top_k` sampling — limit the candidate set before picking.

#### 3c. Context window = working memory (not long-term memory)

The **context window** is everything in *this* API call the model can attend over:

```text
system + tools schema + history + retrieved chunks + user message + tokens generated so far
```

Karpathy analogy: LLM weights ≈ long-term knowledge (lossy, trained once);  
**context window ≈ RAM / working memory** for this conversation turn.

| Your control | Effect on working memory |
| --- | --- |
| `remember()` keeps last **20** messages | Caps chat history tokens |
| `similarity_top_k=3` | Caps RAG chunk tokens |
| `reply_mode` concise vs detailed | Nudges shorter/longer *output* |
| Pydantic `max_length=8000` | Caps single user message chars |
| Tool round-trips | Extra hidden messages (tool_calls + ToolMessage) before final text |

**Why RAG helps:** weights may not “remember” your fridge password; putting the chunk in context puts it in **working memory** so attention can copy it into the answer.

**Why history sync from Next.js matters:** server RAM clears on uvicorn reload; client `history` reloads working memory for the next call.

#### 3d. Streaming vs one-shot (same math)

| Endpoint | UX | Internals |
| --- | --- | --- |
| `/chat` (`create_agent`) | Wait → full JSON | Still next-token under the hood; you only see the end |
| `/chat/stream` | Tokens arrive live | Same loop; each token yielded as SSE |

Tool delay ~1–2s: model may emit `tool_calls` first (not user text), then weather HTTP, then final answer tokens stream.

#### 3e. Hallucination vs grounded answers (Part 1–3 together)

| Failure | Cause | Mitigation you already use / can use |
| --- | --- | --- |
| Wrong doc fact | Bad retrieval / weak chunks | Better chunking, `top_k`, score filtering |
| Confident wrong answer | Model sampling from weights, not context | RAG + prompt “say you don’t know”; lower temperature for facts |
| Forgot earlier turn | Dropped from context | `history` sync + `remember()` trim awareness |

**Checklist for Part 3**

- [x] Generation = repeated next-token prediction
- [x] Temperature scales randomness of sampling
- [x] Context window = working memory for one call
- [x] Streaming = same loop, tokens exposed early
- [x] RAG puts facts into working memory so attention can use them

### Part 4 — Chunking (what you embed and retrieve)

**Chunking** = split each document into smaller pieces **before** embedding.  
Each chunk becomes **one vector** in Supabase. Retrieval returns chunks, not whole files.

```text
lab-secret.md
  → [chunk A] [chunk B] [chunk C]
  → embed A, B, C  →  3 rows in pgvector
  → query finds nearest chunks  →  those texts go into the LLM prompt
```

#### 4a. Why not embed the whole file?

| Whole-file embedding | Chunked embedding |
| --- | --- |
| One vector averages the whole doc | Each vector focuses on a local passage |
| Query must match “overall” meaning | Query can hit a specific fact |
| Hard to fit long docs in prompt | Send only top-k small pieces |

Your `data/` files are short, so defaults work well — chunking still matters as docs grow (uploads, manuals).

#### 4b. What your project does today

```text
SimpleDirectoryReader → SentenceSplitter(chunk_size=512, chunk_overlap=64)
  → VectorStoreIndex.from_documents(..., transformations=[splitter])
```

Env: `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP`. After change: call `POST /rag/rebuild` or vectors stay stale.

#### 4c. Knobs that matter

| Knob | Meaning | Trade-off |
| --- | --- | --- |
| **chunk_size** | Max size of one piece | Too small → fact split across chunks; too big → diluted / noisy retrieval |
| **chunk_overlap** | Shared text between neighbors | Helps if a sentence straddles a boundary; costs more storage |
| **similarity_top_k** | How many chunks enter the prompt | Higher = more context + more tokens/noise |
| **score gap** (`RAG_SOURCE_SCORE_GAP`) | Drop weak source previews | Cleanup for API `sources`, not the LLM path itself |

#### 4d. Overlap intuition

```text
… the fridge password is BANANA-42 and was set by …
          ^-- if the cut falls here, one chunk may miss the code
With overlap, both neighbors may still contain "BANANA-42"
```

#### 4e. Symptoms → fix (use Parts 1–3 vocabulary)

| Symptom | Likely cause | Try |
| --- | --- | --- |
| “I don’t know” but fact is in the file | Fact split / wrong chunk retrieved | Smaller chunks + overlap; check `retrieval_query` on `/rag-hybrid` |
| Answer mixes irrelevant notes | Chunk too large or `top_k` too high | Smaller chunks; lower `top_k` |
| Good answer, weird `sources` | Weak neighbors near best hit | Tighten `RAG_SOURCE_SCORE_GAP` |
| After editing `.md`, old answers | Vectors not rebuilt | `POST /rag/rebuild` |

#### 4f. Chunking experiment (applied)

```python
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
# Wired in main.py via RAG_CHUNK_SIZE / RAG_CHUNK_OVERLAP → transformations=
# After changing env or defaults: POST /rag/rebuild
```

Compare the same question before/after; inspect `/rag-hybrid` `retrieval_query` + `sources`.

#### 4g. How Parts 1–4 fit together

```text
Chunking (Part 4)  →  what text becomes a vector
Embeddings (Part 1) →  how that text becomes numbers + nearest-neighbor search
Transformer (Part 2) →  how Groq attends over retrieved chunk text in the prompt
Next-token (Part 3)  →  how the answer is sampled into your UI / JSON
```

**Checklist for Part 4**

- [x] Chunk = unit of embedding + retrieval
- [x] size / overlap / top_k trade-offs
- [x] Project uses `SentenceSplitter` (`RAG_CHUNK_SIZE`/`OVERLAP`, default 512/64)
- [x] Rebuild required after changing docs or splitter
- [x] Bad RAG often starts at chunk boundaries, not “the LLM is dumb”
- [x] Applied: wired splitter + rebuild; compared `/rag` vs `/rag-hybrid`

---

## RAG retrieval Part 5 — Reranking (theory + light apply)

### Why retrieve then rerank?

Vector search is a **recall** step: “give me ~6 chunks that might be related.”  
Rerank is a **precision** step: “reorder those 6 and keep the best 3 for the LLM.”

```text
question
  → embed → pgvector top-6          (semantic neighbors)
  → rerank                          (second scoring pass)
  → keep top-3 → put in prompt → Groq answers
```

In this project that pipeline lives only on **`/rag-hybrid`** (`rag_hybrid.py`).  
`/rag` uses LlamaIndex’s chat engine retrieval as-is (no keyword rerank).

### Two families of rerankers

| Kind | How it scores | Pros | Cons | In this stack |
| --- | --- | --- | --- | --- |
| **Keyword / lexical** | Count query tokens inside chunk text | Fast, no extra model, easy to debug | Misses synonyms (“DB” ≠ “PostgreSQL”) | **What you have** (`rerank_nodes_by_keywords`) |
| **Cross-encoder** | Neural model scores *(query, chunk)* pairs together | Better relevance when wording differs | Heavier (often local model / API cost) | Future upgrade, not required for learning |

Bi-encoder (your HF `bge-small`) embeds query and chunk **separately**, then compares vectors.  
Cross-encoder reads **both** in one pass — slower but usually more accurate for the final shortlist.

### Your keyword score (exact formula)

```text
tokens = alphanumeric words in query (length ≥ 2)
score  = (# of those tokens found in chunk text) / (# of query tokens)
```

Example from today’s apply (`What does DataSync support?`):

| Chunk | Keyword score | Why |
| --- | --- | --- |
| Mission / trustworthy AI | `0.00` | no “datasync” / “support” |
| DataSync + PostgreSQL + MySQL | `0.50` | hits “datasync” + “support” |
| Office hours | `0.00` | unrelated |

After rerank, DataSync chunk jumps to **#1** even if vector search ranked a generic chunk higher.

### Knobs

| Env | Role |
| --- | --- |
| `RAG_RERANK_ENABLED` | on/off |
| `RAG_RETRIEVE_TOP_K` | candidates from vectors (default 6) |
| `RAG_RERANK_TOP_K` | chunks kept for Groq (default 3) |

Response field `rerank_applied` tells you whether the reorder ran.

### Light apply (done 2026-07-20)

Same question on `/rag-hybrid`:

- **rerank on** → `rerank_applied: true`; sources reordered (handbook / FAQ-ish chunks promoted by tokens)
- **rerank off** → `rerank_applied: false`; raw vector order (FAQ + project-notes + handbook)

Both still answered DataSync DBs correctly on this easy question — rerank matters more when vector top hits are *noisy* and you need the exact product name chunk first.

**Checklist for Part 5**

- [x] Retrieve = recall; rerank = precision
- [x] Keyword vs cross-encoder trade-off
- [x] Mapped to `rag_hybrid.rerank_nodes_by_keywords`
- [x] A/B: `RAG_RERANK_ENABLED` true vs false on DataSync question

---

## Memory & context (learning deep-dive)

Three different “memories” in this stack — easy to mix up:

| Store | Where | Survives uvicorn reload? | Used by |
| --- | --- | --- | --- |
| **LLM weights** | Groq model | N/A (trained knowledge) | Always — but doesn’t know *your* fridge password |
| **Context window** | One API call’s prompt | No — rebuilt every request | Everything the model can attend to *this turn* |
| **Session / client history** | Server dicts + browser `localStorage` | Server: **no**; client: **yes** | Follow-ups (“what’s my name?” / “who set it?”) |
| **Vector DB** | Supabase pgvector | **Yes** | RAG facts from `backend/data/` |

### Who is source of truth?

```text
UI (localStorage)  --history-->  FastAPI sync_*  -->  chat_sessions / rag engine / hybrid_rag_sessions
                                      ^
                                      |  if history omitted (curl): keep whatever is already in RAM
```

With the **Next.js UI**, the browser is source of truth.  
With **curl /docs** and no `history`, the server dict is the only memory (lost on reload).

### Three separate server dicts

| Dict | Endpoint family |
| --- | --- |
| `chat_sessions` | `/chat`, `/chat/stream` |
| `rag_sessions` | `/rag` (cached chat engines) |
| `hybrid_rag_sessions` | `/rag-hybrid` |

**Same `session_id` does not share memory across modes.**  
Light apply: told chat “favorite color is blue” with `session_id=shared-id`, then asked `/rag` the same id → RAG said it doesn’t know (correct — no leak from chat).

### Trim = working-memory budget

`remember()` / `remember_hybrid_turn()` keep **last 20 messages** (~10 turns).  
Older turns fall out of the prompt even if the UI still shows them in the sidebar — unless the client keeps sending them via `history`.

### When RAG beats chat memory (and vice versa)

| Need | Prefer |
| --- | --- |
| Fact from uploaded / `data/` docs | **RAG** (`/rag` or `/rag-hybrid`) |
| User’s name / preferences this chat | **Chat history** |
| Follow-up that points at a doc (“who set **it**?”) | Both: history for condense + RAG for the fact |

Light apply on `/rag-hybrid`:

| Case | `retrieval_query` | Lesson |
| --- | --- | --- |
| Prior turn about fridge, then “Who set it?” | `Who set the fridge password?` | History → condense → better search |
| “Who set it?” alone (fresh session) | `Who set it?` | Vague query; retrieval weaker / less targeted |

### Checklist

- [x] Context window ≠ long-term memory ≠ vector store
- [x] Client `history` rehydrates after server wipe
- [x] Chat / RAG / hybrid sessions are separate
- [x] Condense needs history for pronoun follow-ups
- [x] Trim (20 msgs) caps prompt size

---

## Agents vs chains (deeper learning)

### Plain chain (no agent)

```text
prompt → LLM → text answer
```

One model call. Good for: math, rewriting, “explain X”, anything that needs **no external action**.

### Tool-calling loop (what people call an “agent” here)

```text
prompt → LLM
       ↘ tool_calls? → run tool(s) → ToolMessage → LLM again → …
       ↘ text?       → done
```

The model **decides** whether to call `get_weather`. Your backend **executes** the tool. Groq never hits Open-Meteo itself.

### Your two harnesses (same tool)

| | `/chat` | `/chat/stream` (UI) |
| --- | --- | --- |
| Pattern | `create_agent` (LangGraph loop) | Manual `bind_tools` + `for` loop |
| Who runs the loop? | Library | You (`astream_chat_with_tools`, max 5 rounds) |
| Output | One JSON `reply` | SSE tokens |
| Best for learning | Less boilerplate | See every tool round |

Both use the same `@tool get_weather` and the same system nudge.

### When *not* to use an agent

| Situation | Prefer |
| --- | --- |
| Doc Q&A from `data/` | **RAG** (`/rag`), not chat tools |
| Pure reasoning / formatting | Plain LLM (no tools bound) |
| Live weather / APIs / calculators | Tool loop |
| Multi-step “look up then decide” | Agent / tool loop |

Light apply (2026-07-20):

| Case | Result |
| --- | --- |
| `2+2` on both endpoints | `4` — no tool needed |
| Weather in London | Both returned ~20.4°C clear sky — tool ran |
| Fridge password via `/chat` only | **“I don't know.”** — no RAG, no inventing `BANANA-42` |

Lesson: **tools ≠ document memory**. Weather tool doesn’t make the model know lab secrets; RAG does.

### Failure modes to watch

| Failure | What happens | Your mitigation |
| --- | --- | --- |
| Infinite tool calls | Model keeps requesting tools | `MAX_CHAT_TOOL_ROUNDS = 5` |
| Tool error | Bad location / HTTP fail | `fetch_weather` returns error string → model can apologize |
| Blank UI ~1–2s | Tool round before first token | Expected on `/chat/stream` |
| Middleware `latency_ms` short on SSE | Times until stream **starts**, not finishes | Time client-side or script wall clock |

### Checklist

- [x] Chain = single LLM call; agent/tool-loop = LLM may call tools then answer
- [x] `create_agent` vs manual loop mapped to `/chat` vs `/chat/stream`
- [x] Know when RAG beats tools (and when tools beat RAG)
- [x] Light apply: math / weather / no-hallucinate fridge via chat

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
| 2026-07-16 | Built `/rag-hybrid` in new `backend/rag_hybrid.py` module | LlamaIndex retrieves chunks; LangChain/Groq answers; includes `retrieval_query` for debugging |
| 2026-07-16 | Commented `backend/rag_hybrid.py` | Top-to-bottom notes: memory → condense → retrieve → answer |
| 2026-07-16 | Neural networks beginner guide (GFG) | Neurons, layers, forward/backprop, activations; mapped to Groq LLM + HF embeddings in this stack |
| 2026-07-17 | Started RAG internals: embeddings + vector search | Mapped to `bge-small` / 384-dim / Supabase pgvector / `top_k=3`; sketched Transformer blocks for next |
| 2026-07-17 | Transformer Part 2: position, attention, multi-head, FFN | Q/K/V + heads + FFN; contrasted RAG embeddings vs LLM token embeddings; full `/rag` mental model |
| 2026-07-17 | Part 3: next-token, temperature, context as working memory | Mapped to `temperature=0.7`, SSE stream, `remember()[-20]`, `top_k=3`, tool delay |
| 2026-07-17 | Part 4: chunking for RAG | Default LlamaIndex splitter; size/overlap/top_k; rebuild after doc/splitter changes; symptoms→fixes |
| 2026-07-17 | Explored Google Intro to ML + tokenization, word embeddings, Transformers | Mapped to data prep / tokens / bge-small / Groq; GFG NN guide already logged |
| 2026-07-20 | Apply: `/rag` vs `/rag-hybrid` same Q + follow-up | Hybrid exposes `retrieval_query`; both grounded BANANA-42, no invented “who set it” |
| 2026-07-20 | Apply: `SentenceSplitter(512,64)` + `/rag/rebuild` | Env `RAG_CHUNK_SIZE`/`OVERLAP`; rebuild returns chunk knobs; fridge still retrieves |
| 2026-07-20 | Apply: temperature A/B on `/chat` | Optional `temperature`; slogans vary at 1.0; `17*19` stayed 323 at 0 and 1 |
| 2026-07-20 | Phase 6: logging + rate limit + LLM timeout | `ops.py`; `GROQ_TIMEOUT_SECONDS`; `RATE_LIMIT_PER_MINUTE`; 429/504 |
| 2026-07-20 | Chunk tuning: `RAG_CHUNK_SIZE=256`, overlap 64 (experiment) | `/rag` + `/rag-hybrid` still answer DataSync + BANANA-42 correctly; source snippets shifted toward smaller sections |
| 2026-07-20 | Added `backend/rag_eval.py` for repeatable RAG checks | 3 eval cases × 2 endpoints, latency + pass/fail; `--chunk-size/--chunk-overlap` overrides |
| 2026-07-20 | Reranking on `/rag-hybrid` | Vector top-6 → keyword overlap rerank → top-3; `rerank_applied` in response |
| 2026-07-20 | pgvector cosine index auto-create | `_ensure_supabase_vector_index` after ingest/load; fewer query warnings |
| 2026-07-20 | Split routers: `routers/chat.py`, `routers/rag.py` | HTTP wiring moved out of `main.py`; helpers/models stay shared |
| 2026-07-20 | Latency profile: `rag_eval.py --profile` | 5 Q × `/rag`+`/rag-hybrid`; `/rag` ~1.5s avg, hybrid ~1.9s (+~322ms); fixed router `request` annotation 422 |
| 2026-07-20 | Rerank theory + light A/B | Keyword vs cross-encoder notes; DataSync demo scores; on/off compare via `rerank_applied` |
| 2026-07-20 | Memory & context deep-dive + light apply | Server vs client history; separate chat/rag/hybrid stores; condense needs prior turn |
| 2026-07-20 | Agents vs chains deeper + light apply | Math (no tool) / weather (tool) / chat≠RAG; mapped create_agent vs manual loop |

---

## Current focus

> **Done:** Agents vs chains — when to use a tool loop vs plain LLM vs RAG.

**Next learning options**

1. **Draw `/rag-hybrid` mental-model diagram** in your own words (condense → retrieve → rerank → answer)
2. **Prompting patterns** — system vs few-shot vs grounded RAG prompts in this codebase
3. **Cross-encoder (optional later)** — model-based rerank after keyword is clear
4. **Second tool (optional)** — only if you want multi-tool agent practice
