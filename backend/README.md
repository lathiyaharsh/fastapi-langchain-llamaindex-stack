# Learning backend (FastAPI + LangChain + LlamaIndex)

## Setup (same as Phase 0)

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env               # then put your real GROQ_API_KEY in .env
```

## Run

Always activate the venv first (or call `.venv/bin/uvicorn` directly).
If you run system `uvicorn` outside the venv, imports like `langchain_groq` will fail.

```bash
source .venv/bin/activate
which uvicorn   # should be .../backend/.venv/bin/uvicorn
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

- API: http://127.0.0.1:8000  
- Docs: http://127.0.0.1:8000/docs  
- Health: http://127.0.0.1:8000/health  

## RAG embeddings

Uses **Hugging Face Inference API** (`HUGGINGFACE_API_KEY`) — cloud embeddings, no local torch needed for RAG.

## Vector store: Supabase + pgvector

Vectors are stored in **Supabase Postgres** (pgvector), not only in memory.

1. Create a free project at https://supabase.com
2. In the project: open SQL Editor and run:

```sql
create extension if not exists vector;
```

3. Dashboard → **Connect** → copy the **URI** connection string  
   - Prefer **Session pooler** (`*.pooler.supabase.com`) if direct DB host fails  
   - Change `postgres://` → `postgresql://` if needed  
   - Replace `[YOUR-PASSWORD]` with your database password  
   - If the password has `@ # % *` etc., leave it as-is in `.env` — the backend URL-encodes it automatically  
4. Put it in `backend/.env` as `SUPABASE_DB_URL=...`
5. Restart uvicorn, then:

```bash
curl -s http://127.0.0.1:8000/health   # supabase_configured should be true
curl -s -X POST http://127.0.0.1:8000/rag/rebuild
curl -s -X POST http://127.0.0.1:8000/rag \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the fridge password?"}'
```

After editing files in `data/`, call `POST /rag/rebuild` again.

## Next.js integration (Phase 5)

The learning UI lives in **`fastapi-stack/web/`** (port 3001), not in the parent AI-Chat app.

| UI mode | Next route (`web/`) | FastAPI |
| --- | --- | --- |
| Chat | `POST /api/chat` | `POST /chat/stream` (+ optional `history`) |
| Ask My Docs | `POST /api/rag` | `POST /rag` |
| Upload doc | `POST /api/rag/upload` | `POST /rag/upload` (.md / .txt, incremental insert) |
| Clear chat | `DELETE /api/chat/session` | `DELETE /chat/session/{id}` |
| Clear docs | `DELETE /api/rag?session_id=` | `DELETE /rag/session/{id}` |

Chat requests may include `history` (prior turns) so Groq stays in sync when users switch from Gemini/HF.

`/chat/stream` skips `remember()` if the client disconnects (Stop button).

## Tests

```bash
.venv/bin/python -m unittest tests.test_main -v
```

