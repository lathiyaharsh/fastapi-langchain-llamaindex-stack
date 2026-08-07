---
marp: true
theme: default
paginate: true
title: How Modern AI Applications Work
description: From User Prompt to AI Response
---

<!-- _class: lead -->

# How Modern AI Applications Work

## From User Prompt to AI Response

Understanding ChatGPT-like apps with Next.js, FastAPI, LangChain & LlamaIndex

---

# Agenda

1. Journey of a prompt
2. Basic architecture + streaming demo
3. Memory, Tools, RAG + demos
4. Tech map (JS equivalents)
5. Production AI platform
6. Takeaways + Q&A

---

# Opening question

## When we press Enter in ChatGPT…

### what actually happens?

---

# The simple picture

```text
User
  ↓
Frontend
  ↓
Backend
  ↓
LLM
  ↓
Response
```

Looks like any app: **client → server → external service**

---

# Basic architecture

```text
User → Next.js → FastAPI → LLM → streamed Response
```

| Layer | Job | JS mental model |
|-------|-----|-----------------|
| **Next.js** | Chat UI | React + Route Handlers (BFF) |
| **FastAPI** | AI coordination | Express |
| **LLM** | Generate tokens | Stripe / Twilio-style API |

---

# Who does what?

**Next.js**
- UI + local chat history
- Proxy / BFF routes
- Consume streamed tokens

**FastAPI**
- Auth, validation, rate limits
- Prompts, memory, tools, RAG
- Stream response back

**LLM**
- GPT · Claude · Gemini · Groq
- Generates text — doesn’t know your product by default

---

# Why a dedicated AI backend?

You *can* call the LLM from Next.js.

Extract a backend when AI grows:

- Tools & RAG
- Shared memory
- Multiple clients
- Secrets & ownership

Same reason you’d extract **payments** or **search**.

---

<!-- _class: lead -->

# Demo 1 — Streaming chat

Ask: `Explain React Hooks in 3 bullet points`

Watch Network → `text/event-stream` (SSE)

---

# SSE in one line

**Server-Sent Events** = server pushes chunks over one HTTP connection

JS: `EventSource` / streamed `fetch` body

Chat UX ≈ **stream parsing** + progressive UI  
— not waiting for one big JSON blob

---

# More than an LLM

A production AI system is **not** frontend → LLM.

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

---

# Prompt builder

Assembles the final request to the model:

- System prompt (rules / tone)
- User message
- Conversation history
- Retrieved context (RAG)

**JS analogy:** building the API request body —  
but the body is conversation + context

---

# Memory

Without memory:

```text
You: My name is Harsh
You: What's my name?
AI:  I don't know.
```

Memory = **session / thread ID + message history**  
Same idea as cart state across requests

---

<!-- _class: lead -->

# Demo 2 — Memory

1. `My name is Harsh and I prefer TypeScript.`
2. `What language do I prefer, and what's my name?`

---

# Tool calling

The model can ask your backend to run functions.

Weather · Database · Calendar · Payments · Internal APIs

**JS analogy:** model says `getWeather({ city })`  
→ your controller runs it  
→ result goes back to the model for the final answer

---

# Tool calling flow

```text
User
  ↓
LLM
  ↓
Tool
  ↓
Weather API
  ↓
LLM
  ↓
Answer
```

Model didn’t invent the temperature — **your code fetched it**

---

<!-- _class: lead -->

# Demo 3 — Weather tool

Ask: `What's the weather in Mumbai right now?`

---

# RAG

**Retrieval-Augmented Generation**

Answer from *your* documents — not only training data

```text
Document → Chunk → Embed → Vector DB
                              ↓
Question → similar chunks → LLM → Answer
```

---

<!-- _class: lead -->

# Demo 4 — Ask My Docs

Ask something **only the uploaded doc** can answer

Grounded answer vs hallucination

---

# Tech map (JS equivalents)

| Tech | Role | Closest JS idea |
|------|------|-----------------|
| FastAPI | API framework | Express / Nest / Hono |
| LangChain | Orchestration | Compose prompts, memory, tools |
| LlamaIndex | RAG / data | Index docs → retrieve → LLM |
| Groq | LLM provider | OpenAI-compatible chat API |
| Embeddings | Text → vectors | Similarity features |
| pgvector | Vector store | Postgres + similarity |
| SSE | Streaming | EventSource / streamed fetch |

Remember the **jobs**: orchestrate · remember · retrieve · tools · stream

---

# Evolving the backend

```text
1) Frontend → FastAPI → LLM

2) + Memory

3) + RAG

4) + Tools

5) Production AI platform →
```

---

# Production AI platform

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
LLM providers
```

---

# Platform components

| Component | One-liner |
|-----------|-----------|
| AI Gateway | Single entry for AI requests |
| Router | Pick model by task / cost / latency |
| Prompt manager | Templates + versions |
| Memory | Conversation context |
| RAG | Company knowledge retrieval |
| Context builder | Merge prompt + history + docs |
| Tool executor | Run APIs safely |
| Cache | Reuse similar work |
| Reflection | Optional quality pass |
| Observability | Latency, tokens, errors, cost |

---

# Key takeaways

1. AI apps are more than one LLM API call
2. Backend orchestrates prompts, memory, tools, retrieval
3. Production adds routing, caching, observability, reliability
4. Same blocks power ChatGPT, Cursor, Claude, enterprise assistants

---

<!-- _class: lead -->

# Q&A

**FE** — streaming / UX / errors  
**BE** — tools / auth / rate limits / prompt injection  

---

# Demo cheat sheet

| # | Prompt | Prove |
|---|--------|--------|
| 1 | Explain React Hooks in 3 bullets | Tokens stream · SSE |
| 2 | Name + TS → ask name + language | Remembers turns |
| 3 | Weather in Mumbai right now? | Live tool, not guess |
| 4 | Doc-specific question | Grounded in upload |
