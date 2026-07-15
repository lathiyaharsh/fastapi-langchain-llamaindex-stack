# Project notes — FastAPI learning backend

This project teaches building an AI API with FastAPI.

## Stack

- FastAPI serves HTTP endpoints
- LangChain handles chat, prompts, and memory
- LlamaIndex handles document Q&A (RAG)
- Groq is the LLM provider

## Endpoints you built

- GET /health — server status
- POST /chat — full reply with session memory
- POST /chat/stream — streaming reply via SSE
- DELETE /chat/session/{id} — clear memory
