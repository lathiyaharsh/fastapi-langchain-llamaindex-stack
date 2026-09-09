# Web — AI Chat & Knowledge Assistant UI

The Next.js front end for the assistant. Provides the chat interface and document Q&A experience, and proxies requests to the FastAPI backend so API keys never reach the browser.

**Stack:** Next.js 15 (App Router) · React 19 · TypeScript · Tailwind CSS 4 · Vitest

## Prerequisites

- Node.js 18+
- npm 9+
- The backend running on port 8000 — see [../backend/README.md](../backend/README.md)

## Setup

```bash
cd web
cp .env.example .env.local
npm install
```

`.env.local` only needs the backend URL (the default already matches a locally running backend):

```bash
FASTAPI_URL=http://127.0.0.1:8000
```

API keys and secrets live only in the backend's `.env` — never in the frontend.

## Running the app

```bash
npm run dev
```

Open **http://localhost:3001**.

| Command | Description |
| --- | --- |
| `npm run dev` | Start the development server (port 3001) |
| `npm run build` | Create a production build |
| `npm start` | Serve the production build |
| `npm run lint` | Run ESLint |
| `npm run format` | Format code with Prettier |
| `npm test` | Run the unit test suite (Vitest) |

## Features

- **Chat** — real-time streaming responses over Server-Sent Events, with per-session history.
- **Ask My Docs** — retrieval-augmented Q&A over uploaded or pre-loaded documents.
- Rendered Markdown responses with syntax-highlighted code blocks.
- Local session persistence, so conversations survive a page reload.
- One-click chat export.

## How it talks to the backend

The browser only calls Next.js API routes; those routes forward requests to FastAPI server-side, keeping the backend URL and any credentials off the client.

| UI action | Next.js route | FastAPI endpoint |
| --- | --- | --- |
| Chat (streaming) | `POST /api/chat` | `POST /chat/stream` |
| Clear chat session | `DELETE /api/chat/session` | `DELETE /chat/session/{id}` |
| Ask My Docs | `POST /api/rag` | `POST /rag` |
| Upload document | `POST /api/rag/upload` | `POST /rag/upload` |
| Clear documents | `DELETE /api/rag?session_id=` | `DELETE /rag/session/{id}` |

## Project layout

```text
web/
  app/
    page.tsx              Main application UI
    layout.tsx
    api/chat/              Chat proxy routes
    api/rag/                Document Q&A proxy routes
  components/
    FastApiChat.tsx        Chat / Ask My Docs interface
    MarkdownContent.tsx    Markdown + code rendering
  lib/
    config.ts               Backend URL configuration
    api/                     API client, validation, rate limiting
    sse.ts / sse-client.ts   Streaming response handling
    chat-storage.ts          Session persistence
```

## Troubleshooting

| Problem | Fix |
| --- | --- |
| UI can't reach the API | Confirm the backend is running on port 8000 and `FASTAPI_URL` in `.env.local` matches |
| Chat or document Q&A errors | Check the backend logs and confirm its `.env` keys are set |
| Port already in use | The dev server is fixed to port **3001** in `package.json` |

Full stack setup: [../README.md](../README.md)