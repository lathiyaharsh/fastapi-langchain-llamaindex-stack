# FastAPI learning stack

Separate from the original **AI-Chat** Next.js app. This folder is self-contained:

```text
fastapi-stack/
  backend/     Python FastAPI + LangChain + LlamaIndex (port 8000)
  web/         Next.js UI for chat + Ask My Docs (port 3001)
```

The root `AI-Chat` project at `../` stays the **classic multi-provider** demo (Groq/Gemini/HF via Next.js only).

## Prerequisites

Install these before you start:

| Tool | Version | Check |
| --- | --- | --- |
| Python | 3.10+ | `python3 --version` |
| Node.js | 18+ | `node --version` |
| npm | 9+ | `npm --version` |

You will also need API keys:

- **Groq** (required for chat) — https://console.groq.com
- **Hugging Face** (required for Ask My Docs embeddings) — https://huggingface.co/settings/tokens
- **Supabase** (required for Ask My Docs vector storage) — https://supabase.com

Chat works with only `GROQ_API_KEY`. Ask My Docs needs all three.

## First-time setup

Run these steps once from the `fastapi-stack/` folder.

### 1. Backend (Python)

```bash
cd backend

# Create and activate a virtual environment (do this inside backend/, not AI-Chat/)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
```

Edit `backend/.env` and set at least:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

For **Ask My Docs**, also set `HUGGINGFACE_API_KEY` and `SUPABASE_DB_URL`. See [backend/README.md](./backend/README.md) for Supabase/pgvector setup.

Verify the venv is active — `which uvicorn` should point inside `backend/.venv/`:

```bash
which uvicorn
# expected: .../fastapi-stack/backend/.venv/bin/uvicorn
```

If it shows `~/.local/bin/uvicorn` or `/usr/bin/uvicorn`, the venv is not active. Run `source .venv/bin/activate` again, or use `.venv/bin/uvicorn` directly.

### 2. Web (Next.js)

Open a **new** terminal:

```bash
cd web

cp .env.example .env.local
npm install
```

`web/.env.local` only needs the backend URL (default is fine if you run FastAPI on port 8000):

```bash
FASTAPI_URL=http://127.0.0.1:8000
```

## Run (two terminals)

Start the backend first, then the web UI.

### Terminal 1 — Python backend

```bash
cd backend
source .venv/bin/activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

- API: http://127.0.0.1:8000
- Docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

### Terminal 2 — Next.js UI

```bash
cd web
npm run dev
```

Open **http://localhost:3001**.

## Quick health check

After the backend is running:

```bash
curl -s http://127.0.0.1:8000/health
```

You should see JSON with `status: "ok"`. If you configured Supabase, `supabase_configured` should be `true`.

## What the web app does

- **Chat** → `POST /api/chat` → FastAPI `/chat/stream` (Groq + LangChain)
- **Ask My Docs** → `POST /api/rag` → FastAPI `/rag` (LlamaIndex + Supabase)
- Same visual style as the original chat UI
- Separate localStorage keys and session IDs from the classic app

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `No module named 'langchain_groq'` | Activate the venv (`source .venv/bin/activate`) or run `.venv/bin/uvicorn ...` |
| `bash: .venv/bin/activate: No such file` | Run `python3 -m venv .venv` from inside `backend/` first |
| `pip install` fails on llama-index packages | Make sure you are on the latest `requirements.txt` from this repo |
| Ask My Docs errors | Set `HUGGINGFACE_API_KEY` and `SUPABASE_DB_URL` in `backend/.env`, then `POST /rag/rebuild` |
| Web UI cannot reach API | Confirm backend is on port 8000 and `FASTAPI_URL` in `web/.env.local` matches |

More backend detail (Supabase SQL, RAG rebuild, curl examples): [backend/README.md](./backend/README.md).

## Learning checklist

See [LEARNING_LANGCHAIN_LLAMAINDEX_FASTAPI.md](./LEARNING_LANGCHAIN_LLAMAINDEX_FASTAPI.md).

## Tests

```bash
# Backend
cd backend && .venv/bin/python -m unittest tests.test_main -v

# Web
cd web && npm test
```
