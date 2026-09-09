# AI Chat & Knowledge Assistant

A production-style AI assistant platform combining conversational chat, real-time tool use, and document-grounded question answering (RAG) over your own content.

Built with **Next.js** on the front end and **FastAPI + LangChain + LlamaIndex** on the back end, this stack demonstrates a complete, deployable AI application — not just a single API call to a language model.

## What it does

- **Conversational Chat** — natural, streaming responses powered by Groq's LLMs via LangChain, with per-session memory so the assistant remembers context across turns.
- **Live Tool Use** — the assistant can call live external tools (e.g. real-time weather) when a question needs current, factual data instead of guessing.
- **Ask My Docs (RAG)** — upload your own documents (Markdown / text) and get answers grounded in that content, with retrieval powered by LlamaIndex and a Postgres/pgvector vector store.
- **Secure by default** — authenticated admin actions, per-IP rate limiting, request logging, and sanitized error responses.

## Architecture

```text
 Browser
   │
   ▼
 Next.js (UI + BFF)         →  Streaming chat UI, session storage, API proxy
   │  HTTPS
   ▼
 FastAPI (AI service)       →  Auth, validation, rate limiting, orchestration
   │
   ├── LangChain  → prompts, memory, tool calling      →  Groq LLM
   └── LlamaIndex → chunking, embeddings, retrieval     →  Postgres (pgvector)
```

The browser only ever talks to Next.js. All model provider keys and database credentials stay server-side in the FastAPI service.

## Tech stack

| Layer | Technology |
| --- | --- |
| Frontend | Next.js 15 (App Router), React 19, TypeScript, Tailwind CSS |
| Backend | FastAPI, Uvicorn, Python 3.10+ |
| AI orchestration | LangChain (chat, memory, tools) |
| Retrieval / RAG | LlamaIndex, Hugging Face embeddings |
| LLM provider | Groq |
| Vector store | Supabase Postgres (pgvector) |
| Auth | JWT (admin-only routes) |

## Project structure

```text
├── backend/     FastAPI service — chat, tools, RAG, auth (port 8000)
└── web/         Next.js UI — chat interface + document Q&A (port 3001)
```

Each part has its own detailed README: [backend/README.md](./backend/README.md) · [web/README.md](./web/README.md)

## Getting started

### Prerequisites

| Tool | Version |
| --- | --- |
| Python | 3.10+ |
| Node.js | 18+ |
| npm | 9+ |

You'll need accounts/API keys for:

- **[Groq](https://console.groq.com)** — required for chat
- **[Hugging Face](https://huggingface.co/settings/tokens)** — required for document embeddings (Ask My Docs)
- **[Supabase](https://supabase.com)** — required for vector storage (Ask My Docs)

> Chat works with only a Groq key. Ask My Docs requires all three.

### 1. Backend setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
```

Edit `backend/.env` and set at minimum:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

For Ask My Docs, also set `HUGGINGFACE_API_KEY` and `SUPABASE_DB_URL` — see [backend/README.md](./backend/README.md#vector-store-supabase--pgvector) for the full Supabase setup.

### 2. Frontend setup

```bash
cd web
cp .env.example .env.local
npm install
```

The default `FASTAPI_URL` already points at the local backend, so no changes are needed for local development.

### 3. Run the app

Two terminals — backend first, then the UI:

```bash
# Terminal 1 — backend
cd backend
source .venv/bin/activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

```bash
# Terminal 2 — frontend
cd web
npm run dev
```

Open **http://localhost:3001**.

| Service | URL |
| --- | --- |
| Web app | http://localhost:3001 |
| API | http://127.0.0.1:8000 |
| Interactive API docs | http://127.0.0.1:8000/docs |
| Health check | http://127.0.0.1:8000/health |

### Verify it's running

```bash
curl -s http://127.0.0.1:8000/health
```

A healthy response returns `"status": "ok"`. If Supabase is configured, `supabase_configured` will be `true`.

## Testing

```bash
# Backend
cd backend && .venv/bin/python -m unittest tests.test_main -v

# Frontend
cd web && npm test
```

## Deployment

The backend deploys as a standard Python web service (Render, Railway, Fly.io, etc.) and the frontend deploys as a standard Next.js app (Vercel, Render, etc.). Full deployment instructions, including environment variables and CORS configuration, are in [backend/README.md](./backend/README.md#deploy-to-the-cloud).

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `No module named 'langchain_groq'` | Activate the virtual environment: `source .venv/bin/activate` |
| Ask My Docs returns errors | Set `HUGGINGFACE_API_KEY` and `SUPABASE_DB_URL` in `backend/.env`, then call `POST /rag/rebuild` |
| Web UI can't reach the API | Confirm the backend is running on port 8000 and `FASTAPI_URL` in `web/.env.local` matches |

For deeper troubleshooting and API details, see [backend/README.md](./backend/README.md) and [web/README.md](./web/README.md).