# Backend — AI Chat & Knowledge Assistant API

The Python API powering the assistant: streaming **chat** (LangChain + Groq), live **tool calling**, and document-grounded **RAG Q&A** (LlamaIndex + Supabase pgvector). The Next.js UI in [`../web/`](../web/) talks to this service on port **8000**.

**Stack:** FastAPI · Uvicorn · LangChain · LlamaIndex · Groq · Hugging Face Inference API · Supabase (pgvector)

## Prerequisites

| Tool | Version |
| --- | --- |
| Python | 3.10+ |

| API key | Required for |
| --- | --- |
| `GROQ_API_KEY` | Chat and RAG-generated answers |
| `HUGGINGFACE_API_KEY` | Document embeddings (Ask My Docs) |
| `SUPABASE_DB_URL` | Vector storage (Ask My Docs) |

Chat runs with only a Groq key. Ask My Docs needs all three.

## Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set at least `GROQ_API_KEY`. For Ask My Docs, also set `HUGGINGFACE_API_KEY` and `SUPABASE_DB_URL`.

## Running the server

Always run inside the virtual environment — the system-wide `uvicorn` won't have the required packages installed.

```bash
source .venv/bin/activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

| URL | Purpose |
| --- | --- |
| http://127.0.0.1:8000 | API root |
| http://127.0.0.1:8000/docs | Interactive API documentation (Swagger UI) |
| http://127.0.0.1:8000/health | Configuration / readiness check |

## Configuration reference

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | — | Groq LLM key for chat and RAG |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Model used for chat / RAG |
| `GROQ_TEMPERATURE` | `0.7` | Default sampling temperature (overridable per request) |
| `GROQ_TIMEOUT_SECONDS` | `60` | LLM request timeout (returns HTTP 504 on timeout) |
| `RATE_LIMIT_PER_MINUTE` | `60` | Per-IP limit on chat and RAG routes (`0` disables) |
| `JWT_SECRET` | *(placeholder — change for production)* | Signs admin auth tokens |
| `AUTH_USERNAME` / `AUTH_PASSWORD` | *(sample credentials)* | Login for `POST /auth/login` |
| `JWT_TTL_MINUTES` | `30` | Admin token lifetime |
| `WARM_RAG_ON_STARTUP` | `true` | Preload the RAG index at startup |
| `HUGGINGFACE_API_KEY` | — | Hosted embeddings (no local model download) |
| `HF_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `HF_EMBED_DIM` | `384` | Must match the Supabase collection dimension |
| `SUPABASE_DB_URL` | — | Postgres connection string (`postgresql://…`) |
| `SUPABASE_COLLECTION` | `ai_chat_docs` | Vector collection name |
| `RAG_MAX_UPLOAD_BYTES` | `2097152` (2 MB) | Max document upload size |
| `RAG_SOURCE_SCORE_GAP` | `0.08` | Filters weak source matches from responses |
| `RAG_CHUNK_SIZE` | `512` | Document chunk size for indexing |
| `RAG_CHUNK_OVERLAP` | `64` | Overlap between chunks (re-run `/rag/rebuild` after changing) |
| `RAG_RERANK_ENABLED` | `true` | Enables keyword reranking on `/rag-hybrid` |
| `RAG_RETRIEVE_TOP_K` | `6` | Candidate chunks fetched before reranking |
| `RAG_RERANK_TOP_K` | `3` | Chunks kept after reranking (sent to the LLM) |

**Update `AUTH_USERNAME`, `AUTH_PASSWORD`, and `JWT_SECRET` before any production or client-facing deployment.** Never commit `.env`.

## API reference

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Service info |
| `GET` | `/health` | Configuration and readiness status |
| `POST` | `/chat` | Full (non-streaming) chat reply |
| `POST` | `/chat/stream` | Streaming chat reply (Server-Sent Events) — used by the web UI |
| `DELETE` | `/chat/session/{id}` | Clear a chat session's history |
| `POST` | `/auth/login` | Exchange username/password for a JWT |
| `GET` | `/auth/me` | Verify the current bearer token |
| `POST` | `/rag` | Document Q&A |
| `POST` | `/rag-hybrid` | Document Q&A with LlamaIndex retrieval + LangChain/Groq answer generation |
| `DELETE` | `/rag-hybrid/session/{id}` | Clear a hybrid RAG session |
| `POST` | `/rag/upload` | Upload a `.md` / `.txt` document for incremental indexing |
| `POST` | `/rag/rebuild` | Re-index all documents in `data/` (**requires authentication**) |
| `DELETE` | `/rag/session/{id}` | Clear a RAG conversation (keeps the indexed vectors) |

### Chat

- Supports live tool calling — the assistant can fetch real-time data (e.g. current weather) instead of guessing.
- Conversation history can be supplied by the client or kept server-side per session.
- If a client disconnects mid-stream, the partial response is not saved to history.

### Document Q&A (RAG)

- Documents are chunked, embedded, and stored in **Postgres with pgvector**, so the knowledge base persists across restarts.
- Source documents live in `backend/data/`. After adding or editing files there, call `POST /rag/rebuild` to re-index.
- Uploads accept `.md` and `.txt` files; duplicate filenames are rejected.
- `/rag` handles retrieval and answer generation in one step. `/rag-hybrid` exposes the retrieval query and applies an additional keyword-based reranking pass for improved relevance — useful when you need visibility into what was retrieved.

### Reliability & observability

- Every request is logged with a unique request ID, method, path, status, and latency; the ID is returned in the `X-Request-Id` response header.
- Per-IP rate limiting on chat and RAG routes returns HTTP `429` with a `Retry-After` header when exceeded.
- LLM calls that exceed `GROQ_TIMEOUT_SECONDS` return a clean HTTP `504` instead of hanging.
- A vector index is maintained automatically on the pgvector table to keep retrieval fast as the document set grows.

## Vector store: Supabase + pgvector

1. Create a free project at [supabase.com](https://supabase.com).
2. In the SQL Editor, run:

   ```sql
   create extension if not exists vector;
   ```

3. Go to **Connect** in the dashboard and copy the connection **URI**:
   - Prefer the **Session pooler** host if the direct connection fails.
   - Use the `postgresql://` scheme (not `postgres://`).
   - Replace `[YOUR-PASSWORD]` with your database password.
4. Set `SUPABASE_DB_URL` in `backend/.env`.
5. Restart the server and verify:

   ```bash
   curl -s http://127.0.0.1:8000/health   # expect "supabase_configured": true

   TOKEN=$(curl -s -X POST http://127.0.0.1:8000/auth/login \
     -H "Content-Type: application/json" \
     -d '{"username":"<your-username>","password":"<your-password>"}' \
     | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

   curl -s -X POST http://127.0.0.1:8000/rag/rebuild -H "Authorization: Bearer $TOKEN"

   curl -s -X POST http://127.0.0.1:8000/rag \
     -H "Content-Type: application/json" \
     -d '{"question":"What is this document about?"}'
   ```

`/rag/rebuild` requires authentication. Chat and standard document Q&A remain open for end users.

## Frontend integration

The web UI lives in [`../web/`](../web/) and runs on port **3001**. See [../web/README.md](../web/README.md) for details.

| UI action | Next.js route | FastAPI endpoint |
| --- | --- | --- |
| Chat | `POST /api/chat` | `POST /chat/stream` |
| Ask My Docs | `POST /api/rag` | `POST /rag` |
| Upload document | `POST /api/rag/upload` | `POST /rag/upload` |
| Clear chat | `DELETE /api/chat/session` | `DELETE /chat/session/{id}` |
| Clear documents | `DELETE /api/rag?session_id=` | `DELETE /rag/session/{id}` |

The browser only talks to Next.js — API keys and database credentials never leave the server.

## Project layout

```text
backend/
  main.py               App setup, request/response models, health check
  routers/
    chat.py              Chat endpoints and session management
    rag.py                Document Q&A, upload, rebuild
  rag_hybrid.py          Hybrid retrieval + reranking pipeline
  ops.py                  Logging and rate-limiting middleware
  rag_eval.py             RAG quality evaluation script
  requirements.txt
  .env.example
  data/                   Source documents for the knowledge base
  tests/
```

## Tests

```bash
.venv/bin/python -m unittest tests.test_main -v
```

## RAG quality evaluation

Run a fixed set of test questions against both RAG endpoints, with pass/fail and latency reporting:

```bash
.venv/bin/python rag_eval.py --rebuild
```

Compare chunking strategies in one command:

```bash
.venv/bin/python rag_eval.py --rebuild --chunk-size 256 --chunk-overlap 64
```

Latency profiling:

```bash
.venv/bin/python rag_eval.py --profile
```

## Deploy to the cloud

The service deploys to any standard Python web host with no containerization required.

### Render (recommended)

1. Push the repository to GitHub (do **not** commit `backend/.env`).
2. In the [Render Dashboard](https://dashboard.render.com), create a **New Web Service** from the repo.
3. Configure:
   - **Root Directory:** `backend`
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Set environment variables (same as local `.env`):
   - `GROQ_API_KEY`
   - `HUGGINGFACE_API_KEY` and `SUPABASE_DB_URL` (for Ask My Docs)
   - `AUTH_USERNAME`, `AUTH_PASSWORD`, `JWT_SECRET` (set real production values)
   - `ALLOWED_ORIGINS` (your frontend's URL)
5. Deploy, then verify `https://YOUR-SERVICE.onrender.com/health` returns `{"status":"ok", ...}`.
6. Point the frontend at the deployed API by setting `FASTAPI_URL` (in `web/.env.local` or your hosting provider's environment settings) to your service URL.

Free-tier services may sleep when idle; the first request after inactivity can take 30–60 seconds.

### Railway

1. Create a new project, deploy from GitHub, and set the root directory to `backend`.
2. Add the same environment variables as above.
3. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT` (or use the included `Procfile`).
4. Point `FASTAPI_URL` at the deployed URL as above.

### CORS

If the browser will call the API directly (rather than only through the Next.js proxy), set:

```bash
ALLOWED_ORIGINS=https://your-frontend-domain.com
```

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `No module named 'langchain_groq'` | Activate the virtual environment or run `.venv/bin/uvicorn ...` |
| Ask My Docs returns errors | Set `HUGGINGFACE_API_KEY` and `SUPABASE_DB_URL`, then call `POST /rag/rebuild` |
| Weak or empty RAG answers | Rebuild the index and confirm the `data/` files and Supabase table contain the expected content |

Full stack setup: [../README.md](../README.md)