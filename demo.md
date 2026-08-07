# How Modern AI Applications Work: From User Prompt to AI Response

**Audience:** JS team (FE / BE / Full stack)  
**Length:** ~20–22 minutes + Q&A  
**Goal:** Show that ChatGPT-like apps are normal product architecture — UI, API, orchestration, data — not “magic LLM calls.”

**Subtitle (optional slide):**  
Understanding ChatGPT-like apps with Next.js, FastAPI, LangChain, LlamaIndex — mapped to concepts you already use.

---

## Prep checklist (before the room)

- [ ] Backend running on `:8000`, web on `:3001`
- [ ] Chat streaming works
- [ ] Memory demo session ready (fresh chat)
- [ ] Weather tool works
- [ ] RAG: one short doc uploaded (company FAQ / README excerpt) + rebuild done
- [ ] Browser DevTools → Network open (for SSE)
- [ ] Backup: screenshots / short recording if APIs flake

**JS bridge line to repeat:**  
*“Think of the LLM as a smart dependency. Your app still owns auth, validation, streaming, data, and business rules.”*

---

## Agenda (timebox)

| # | Section | Time |
|---|---------|------|
| 1 | Journey of a prompt | 2 min |
| 2 | Basic architecture + Demo 1 (stream) | 4 min |
| 3 | Memory, Tools, RAG + Demos 2–4 | 8 min |
| 4 | Tech map (JS equivalents) | 2 min |
| 5 | Production AI platform (vision) | 4 min |
| 6 | Takeaways + Q&A | 2–3 min |

If short on time: skip section 4 table details; keep demos + production slide.

---

## 1. The Journey of a Prompt (2 min)

**Open with:**  
“When we press Enter in ChatGPT, what actually happens?”

**Show (start simple):**

```text
User → Frontend → Backend → LLM → Response
```

**Say:**  
“Looks like any app: client → server → external service. Today we replace this simple picture with what production AI actually does — and map every piece to things you already know in JS.”

**Audience hook by role:**
- **FE:** streaming UX, chat state, SSE / ReadableStream  
- **BE:** orchestration, tools, validation, rate limits  
- **FS:** where the boundary sits between Next.js and the AI backend  

---

## 2. Basic AI Application Architecture (4 min)

**Build gradually:**

```text
User → Next.js → FastAPI → LLM → streamed Response
```

### Explain each (keep short)

| Layer | What it is | JS mental model |
|-------|------------|-----------------|
| **Next.js** | Chat UI | React app + Route Handlers as BFF |
| **FastAPI** | AI API / orchestration | Express/Nest service dedicated to AI |
| **LLM** | Text generation API | Like Stripe/Twilio — paid external API |

**Next.js responsibilities**
- UI + chat history in the browser
- Calling backend / proxy routes
- Consuming streamed tokens (SSE)

**FastAPI responsibilities**
- Auth, validation, rate limits
- Building prompts, calling the model
- Memory / tools / RAG
- Streaming the response back

**LLM**
- GPT, Claude, Gemini, Groq, etc.
- Only generates tokens — it does not “know” your product by default

### Why not call the LLM only from Next.js?

Say this once for the JS audience:

> “You *can* call Groq from a Next.js route. We use a dedicated backend when AI logic grows: tools, RAG, shared memory, multiple clients, stronger secrets, and clearer ownership — same reason you’d extract a payments or search service.”

### Demo 1 — Simple chat + streaming (~2 min)

**Ask:** `Explain React Hooks in 3 bullet points`

**Show:**
1. Tokens appear gradually (not one big JSON blob)
2. DevTools → Network → event stream / `text/event-stream` (SSE)

**Explain SSE in one sentence:**  
“Server-Sent Events = server pushes chunks over one HTTP connection — like `EventSource` / streaming `fetch` body. Same idea as progressive rendering.”

**FE takeaway:** Chat UX is mostly stream parsing + optimistic UI, not waiting for a full response.

---

## 3. Modern AI Needs More Than an LLM (8 min)

**Say:**  
“A production AI system isn’t frontend → LLM. The backend orchestrates.”

**Replace architecture with:**

```text
Frontend
   ↓
Backend
   ├── Prompt builder
   ├── Memory
   ├── Tools
   └── RAG
   ↓
LLM
```

### Prompt builder

Creates the final message list sent to the model:

- System prompt (product rules / tone)
- User message
- History
- Retrieved context (RAG)

**JS analogy:** Building the request body for an API — but the “body” is conversation + context.

---

### Memory

Stores conversation so the model has context across turns.

**Without memory:**

```text
You: My name is Harsh
You: What's my name?
AI: I don't know.
```

### Demo 2 — Memory (~1.5 min)

1. New chat  
2. `My name is Harsh and I prefer TypeScript.`  
3. `What language do I prefer, and what's my name?`

**Say:**  
“Memory is app state — session/thread ID + message history — not magic. Same problem as keeping cart state across requests.”

---

### Tool calling

Lets the model request backend functions instead of guessing.

**Examples:** weather, DB lookup, calendar, payments, internal APIs

**JS analogy:**  
“The model returns something like ‘call `getWeather({ city })`’ — your server runs it (like a controller), then sends the result back for the final answer.”

### Demo 3 — Weather tool (~2 min)

**Ask:** `What's the weather in Mumbai right now?`

**Draw / show:**

```text
User → LLM → Tool → Weather API → LLM → Answer
```

**Point out:** Model didn’t invent the temperature; your code fetched it.

**BE takeaway:** Tools are normal APIs with schemas + auth + timeouts — the LLM is only the decision layer for *when* to call them.

---

### RAG (Retrieval-Augmented Generation)

Answers using *your* documents, not only training data.

**Pipeline:**

```text
Document → Chunking → Embeddings → Vector DB
                              ↓
User question → similar chunks → LLM → Answer
```

### Demo 4 — Ask My Docs (~2 min)

**Ask something only the uploaded doc answers** (not general knowledge).

**Show briefly:**
- Upload / indexed doc
- Answer cites or reflects doc content
- Wrong/general question without doc = weak/hallucinated vs with RAG = grounded

**FS takeaway:** RAG is a data pipeline + retrieval API in front of the LLM.

---

## 4. Technologies Used (2 min)

One slide — keep it as a **JS translation table**, not a Python tour.

| Technology | What it is | Closest JS mental model |
|------------|------------|-------------------------|
| **FastAPI** | Python API framework | Express / Nest / Hono |
| **LangChain** | AI orchestration | SDK that composes prompts, memory, tools, model calls |
| **LlamaIndex** | Data/RAG framework | Search + indexing library for docs → LLM |
| **Groq** | LLM provider | OpenAI-compatible chat API |
| **Embeddings (HF)** | Text → vectors | Feature vectors for similarity search |
| **pgvector / Supabase** | Vector store | Postgres + similarity index |
| **SSE** | Token streaming | `EventSource` / streamed `fetch` |

**Say:**  
“You don’t need to memorize the stack. Remember the jobs: orchestrate, remember, retrieve, call tools, stream.”

---

## 5. Production AI Platform (4 min)

Don’t only show what we built — show where it sits in a larger platform.

**Evolve live (or animate slides):**

```text
1) Frontend → FastAPI → LLM
2) + Memory
3) + RAG
4) + Tools
5) Full platform ↓
```

**Full picture:**

```text
Client
  ↓
FastAPI API
  ↓
Auth
  ↓
AI Gateway
  ├── Router
  ├── Prompt manager
  ├── Memory
  ├── RAG
  ├── Context builder
  ├── Tool executor
  ├── Cache
  ├── Reflection
  ├── Observability
  ↓
LLM providers (Groq / Gemini / OpenAI / …)
```

**Describe in product language (one line each):**

| Component | One-liner |
|-----------|-----------|
| **AI Gateway** | Single entry for all AI requests |
| **Router** | Pick model/provider by task, cost, latency |
| **Prompt manager** | Templates + versions (like feature flags for prompts) |
| **Memory** | Conversation / user context |
| **RAG** | Company knowledge retrieval |
| **Context builder** | Merge system prompt + history + docs into one request |
| **Tool executor** | Run side-effecting APIs safely |
| **Cache** | Reuse similar answers / embeddings |
| **Reflection** | Optional second pass for quality |
| **Observability** | Latency, tokens, errors, cost |

**Honest closer:**  
“We implemented the core path: chat, memory, tools, RAG, streaming. Router, cache, reflection, full observability are the next production layers — same as shipping MVP payments then adding fraud, retries, and metrics.”

---

## 6. Key takeaways + Q&A (2–3 min)

**Takeaways (leave on screen):**

1. Modern AI apps are more than one LLM API call.  
2. The backend orchestrates prompts, memory, tools, and document retrieval.  
3. Production platforms add routing, caching, observability, and reliability.  
4. Same building blocks power ChatGPT, Cursor, Claude, and internal enterprise assistants.

**Invite questions by role:**
- FE: streaming / UX / error states  
- BE: tools, auth, rate limits, prompt injection  
- FS: Next.js BFF vs dedicated AI service  

---

## Demo script card (print / second screen)

| Demo | Exact prompt | Prove |
|------|--------------|--------|
| 1 Stream | `Explain React Hooks in 3 bullet points` | Tokens stream; Network shows SSE |
| 2 Memory | Name + TS pref → ask name + language | Remembers across turns |
| 3 Tools | `What's the weather in Mumbai right now?` | Live tool path, not hallucinated guess |
| 4 RAG | Doc-specific question | Answer grounded in uploaded doc |

---

## Likely questions (short answers)

**Q: Why FastAPI if we’re a JS team?**  
A: AI ecosystem (LangChain/LlamaIndex) is mature in Python; Next.js stays the product UI/BFF. Same split as Python ML services beside Node apps.

**Q: Can we do this all in Node?**  
A: Yes — LangChain.js, LlamaIndex.TS, Vercel AI SDK. Concepts transfer; this repo teaches the architecture clearly.

**Q: Is RAG just embeddings?**  
A: Embeddings + store + retrieval + prompt assembly (+ optional rerank). Vectors alone aren’t an answer.

**Q: How do we stop hallucinations?**  
A: Ground with RAG/tools, constrain prompts, cite sources, evaluate; never trust the model alone for facts.

**Q: What about cost / latency?**  
A: Streaming improves UX; caching, smaller models for easy tasks, retrieve less context, observe token usage.

---

## Timing cuts (if running long)

1. Drop tech table deep-dive (keep one sentence).  
2. Production platform: show final diagram only, skip step-by-step.  
3. Keep all four demos — they teach more than slides.
