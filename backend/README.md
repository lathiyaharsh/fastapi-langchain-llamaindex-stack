# Learning backend (FastAPI + LangChain + LlamaIndex)

Python API for the FastAPI learning stack. Serves **chat** (LangChain + Groq) and **Ask My Docs** (LlamaIndex RAG + Supabase pgvector). The Next.js UI in [`../web/`](../web/) proxies to this server on port **8000**.

Stack: **FastAPI** · **Uvicorn** · **LangChain** · **LlamaIndex** · **Groq** · **Hugging Face Inference API** · **Supabase (pgvector)**

## Prerequisites

| Tool | Version |
| --- | --- |
| Python | 3.10+ |

API keys:

| Key | Required for |
| --- | --- |
| `GROQ_API_KEY` | Chat + RAG answers |
| `HUGGINGFACE_API_KEY` | Ask My Docs embeddings |
| `SUPABASE_DB_URL` | Ask My Docs vector storage |

Chat works with only Groq. Ask My Docs needs all three.

## Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set at least `GROQ_API_KEY`. For Ask My Docs, also set `HUGGINGFACE_API_KEY` and `SUPABASE_DB_URL`.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | — | Groq LLM for chat and RAG |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Chat / RAG model name |
| `GROQ_TEMPERATURE` | `0.7` | Default sampling temp; override per `/chat` request |
| `HUGGINGFACE_API_KEY` | — | Cloud embeddings (no local torch) |
| `HF_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `HF_EMBED_DIM` | `384` | Must match Supabase collection dim |
| `SUPABASE_DB_URL` | — | Postgres URI (`postgresql://…`) |
| `SUPABASE_COLLECTION` | `ai_chat_docs` | Vector collection name |
| `RAG_MAX_UPLOAD_BYTES` | `2097152` (2 MB) | Upload size limit |
| `RAG_SOURCE_SCORE_GAP` | `0.08` | Filter weak RAG source chunks |
| `RAG_CHUNK_SIZE` | `512` | SentenceSplitter chunk size (tokens-ish) |
| `RAG_CHUNK_OVERLAP` | `64` | Overlap between chunks; change + `/rag/rebuild` |

Never commit `.env`.

## Run

Always use the venv (or call `.venv/bin/uvicorn` directly). System `uvicorn` outside the venv will miss packages like `langchain_groq`.

```bash
source .venv/bin/activate
which uvicorn   # should be .../backend/.venv/bin/uvicorn
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

| URL | What |
| --- | --- |
| http://127.0.0.1:8000 | Root |
| http://127.0.0.1:8000/docs | Swagger UI |
| http://127.0.0.1:8000/health | Key / config check |

## API endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Hello |
| `GET` | `/health` | Config flags (no secrets) |
| `POST` | `/chat` | Full JSON chat reply |
| `POST` | `/chat/stream` | SSE token stream (used by the web UI) |
| `DELETE` | `/chat/session/{id}` | Clear in-memory chat history |
| `POST` | `/rag` | Document Q&A |
| `POST` | `/rag-hybrid` | LlamaIndex retrieve + LangChain/Groq answer |
| `DELETE` | `/rag-hybrid/session/{id}` | Clear hybrid RAG chat history |
| `POST` | `/rag/upload` | Upload `.md` / `.txt` (incremental insert) |
| `POST` | `/rag/rebuild` | Re-embed everything in `data/` |
| `DELETE` | `/rag/session/{id}` | Clear RAG conversation (keeps vectors) |

### Chat notes

- `/chat` uses LangChain **`create_agent`** (automatic tool loop). `/chat/stream` uses the **manual** `bind_tools` loop (what the web UI calls). Same `get_weather` tool on both. Try: *"What's the weather in London?"*
- Requests may include `history` so context survives reload.
- `/chat/stream` skips `remember()` if the client disconnects (Stop button).

### RAG notes

- Vectors live in **Supabase Postgres (pgvector)**, not only in memory.
- Source files live in `backend/data/`. After editing them, call `POST /rag/rebuild`.
- Upload accepts `.md` / `.txt` only; duplicate filenames return `409`.
- Chunking uses `SentenceSplitter` (`RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP`, default 512/64). After changing either, call `POST /rag/rebuild`.
- `/rag` = LlamaIndex chat engine end-to-end (`CONDENSE_PLUS_CONTEXT`).
- `/rag-hybrid` = LlamaIndex retrieval only, then LangChain/Groq writes the final answer. Response also includes `retrieval_query` so you can inspect what got sent to retrieval.

## Vector store: Supabase + pgvector

1. Create a free project at https://supabase.com
2. SQL Editor — run:

```sql
create extension if not exists vector;
```

3. Dashboard → **Connect** → copy the **URI**
   - Prefer **Session pooler** (`*.pooler.supabase.com`) if direct DB host fails
   - Use `postgresql://` (not `postgres://`)
   - Replace `[YOUR-PASSWORD]` with your DB password
   - Special characters in the password are URL-encoded by the backend
4. Set `SUPABASE_DB_URL=...` in `backend/.env`
5. Restart uvicorn, then:

```bash
curl -s http://127.0.0.1:8000/health   # supabase_configured should be true
curl -s -X POST http://127.0.0.1:8000/rag/rebuild
curl -s -X POST http://127.0.0.1:8000/rag \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the fridge password?"}'
```

## Next.js integration (`web/`)

The learning UI is in **`fastapi-stack/web/`** (port **3001**). See [../web/README.md](../web/README.md).

| UI mode | Next.js | FastAPI |
| --- | --- | --- |
| Chat | `POST /api/chat` | `POST /chat/stream` |
| Ask My Docs | `POST /api/rag` | `POST /rag` |
| Upload doc | `POST /api/rag/upload` | `POST /rag/upload` |
| Clear chat | `DELETE /api/chat/session` | `DELETE /chat/session/{id}` |
| Clear docs | `DELETE /api/rag?session_id=` | `DELETE /rag/session/{id}` |

Browser → Next.js BFF → this FastAPI app. API keys stay in `backend/.env` only.

## Project layout

```text
backend/
  main.py           # App, chat, RAG, tools
  requirements.txt
  .env.example
  data/             # RAG source docs (.md / .txt)
  tests/
    test_main.py
```

## Tests

```bash
.venv/bin/python -m unittest tests.test_main -v
```

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `No module named 'langchain_groq'` | Activate venv or run `.venv/bin/uvicorn ...` |
| `bash: .venv/bin/activate: No such file` | Run `python3 -m venv .venv` inside `backend/` |
| Ask My Docs errors | Set `HUGGINGFACE_API_KEY` + `SUPABASE_DB_URL`, then `POST /rag/rebuild` |
| Empty / weak RAG answers | Rebuild index; check `data/` files and Supabase table `vecs.ai_chat_docs` |

Full stack (both terminals): [../README.md](../README.md). Learning checklist: [../LEARNING_LANGCHAIN_LLAMAINDEX_FASTAPI.md](../LEARNING_LANGCHAIN_LLAMAINDEX_FASTAPI.md).
