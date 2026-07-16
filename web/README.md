# Learning UI (Next.js)

Next.js front end for the FastAPI learning stack. Runs on **port 3001** and proxies chat / Ask My Docs to the Python backend.

Stack: **Next.js 15** (App Router) · **React 19** · **Tailwind CSS 4** · **TypeScript** · **Vitest**

## Prerequisites

- Node.js 18+
- npm 9+
- FastAPI backend running on port 8000 (see [../backend/README.md](../backend/README.md))

## Setup

```bash
cd web
cp .env.example .env.local
npm install
```

`.env.local` only needs the backend URL (default is fine if FastAPI is on port 8000):

```bash
FASTAPI_URL=http://127.0.0.1:8000
```

API keys live in the backend `.env`, not here.

## Run

```bash
npm run dev
```

Open **http://localhost:3001**.

Other scripts:

| Command | Description |
| --- | --- |
| `npm run build` | Production build |
| `npm start` | Serve production build |
| `npm run lint` | ESLint |
| `npm run format` | Prettier write |
| `npm test` | Vitest unit tests |

## Features

- **Chat** — streaming replies via SSE (Groq + LangChain on the backend)
- **Ask My Docs** — RAG Q&A over uploaded docs (LlamaIndex + Supabase)
- Markdown rendering with syntax highlighting
- Local chat history / session persistence (separate keys from the classic AI-Chat app)
- Export chat

## API routes (Next → FastAPI)

| UI action | Next.js route | FastAPI |
| --- | --- | --- |
| Chat (stream) | `POST /api/chat` | `POST /chat/stream` |
| Clear chat session | `DELETE /api/chat/session` | `DELETE /chat/session/{id}` |
| Ask My Docs | `POST /api/rag` | `POST /rag` |
| Upload doc | `POST /api/rag/upload` | `POST /rag/upload` |
| Clear docs | `DELETE /api/rag?session_id=` | `DELETE /rag/session/{id}` |

The browser talks only to Next.js; server routes forward to `FASTAPI_URL`.

## Project layout

```text
web/
  app/
    page.tsx              # Main UI
    layout.tsx
    api/chat/             # Chat proxy + session
    api/rag/              # RAG query + upload
  components/
    FastApiChat.tsx       # Chat / Ask My Docs UI
    MarkdownContent.tsx
  lib/
    config.ts             # FASTAPI_URL
    api/                  # Client, validation, rate limit
    sse.ts / sse-client.ts
    chat-storage.ts
```

## Troubleshooting

| Problem | Fix |
| --- | --- |
| UI cannot reach API | Confirm backend is on `:8000` and `FASTAPI_URL` in `.env.local` matches |
| Chat / RAG errors | Check backend logs and `backend/.env` keys (`GROQ_API_KEY`, etc.) |
| Port already in use | Dev server is fixed to **3001** in `package.json` |

Full stack setup (both terminals): [../README.md](../README.md).
