"""
Chat routes (LangChain + Groq).

Learning split:
  POST /chat        → create_agent (automatic tool loop)
  POST /chat/stream → manual bind_tools + astream loop (UI uses this)

Helpers and Pydantic models stay in main.py; this module only owns HTTP wiring.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

# SSE wire format for /chat/stream (Next.js BFF translates to JSON events).
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # nginx: don't buffer SSE
}


def create_chat_router() -> APIRouter:
    """
    Build chat routes.

    Import main inside the factory so helpers exist when main.py calls this
    after defining ChatRequest / invoke_* (avoids circular-import issues).
    """
    import main

    router = APIRouter(tags=["chat"])

    @router.post("/chat", response_model=main.ChatResponse)
    def chat(request: main.ChatRequest):
        """
        Full reply in one JSON response via LangChain create_agent.

        Compare with /chat/stream, which uses the manual tool-calling loop instead.
        Try in Swagger /docs: POST /chat {"message": "What's the weather in London?"}
        """
        message = request.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")

        try:
            main.sync_session_history(request.session_id, request.history)
            reply = main.invoke_chat_with_agent(
                request.session_id,
                message,
                request.reply_mode,
                temperature=request.temperature,
            )
            main.remember(request.session_id, message, reply)
        except HTTPException:
            raise
        except Exception as exc:
            raise main._http_error_for_llm(exc, kind="LLM") from exc

        return main.ChatResponse(reply=reply)

    @router.post("/chat/stream")
    async def chat_stream(request: Request, body: main.ChatRequest):
        """
        Stream tokens over SSE via the *manual* tool loop (not create_agent).

        Wire format (one event per line):
          data: Hello\\n\\n
          data: [DONE]\\n\\n

        Newlines inside a token are escaped as \\n so each SSE line stays intact.
        The Next.js BFF must NOT .trim() payloads — spaces are real tokens.

        If the user hits Stop, we detect disconnect and skip remember() so a
        half-finished reply is not saved to session memory.
        """
        message = body.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")

        try:
            main.sync_session_history(body.session_id, body.history)
            messages = main.build_messages(body.session_id, message, body.reply_mode)
        except HTTPException:
            raise

        async def event_generator():
            parts: list[str] = []
            disconnected = False
            try:
                async for text in main.astream_chat_with_tools(
                    messages, temperature=body.temperature
                ):
                    if await request.is_disconnected():
                        disconnected = True
                        break
                    if text:
                        parts.append(text)
                        # Escape newlines — SSE spec uses \\n\\n as event delimiter.
                        safe = text.replace("\n", "\\n")
                        yield f"data: {safe}\n\n"
                if disconnected:
                    return
                # Only persist full replies — not partial streams after Stop.
                main.remember(body.session_id, message, "".join(parts))
                yield "data: [DONE]\n\n"
            except Exception as exc:
                if main._is_llm_timeout(exc):
                    yield (
                        f"data: [ERROR] LLM request timed out after "
                        f"{main.GROQ_TIMEOUT_SECONDS:.0f}s\n\n"
                    )
                else:
                    yield "data: [ERROR] LLM request failed\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    @router.delete("/chat/session/{session_id}")
    def clear_session(session_id: str):
        """Clear in-memory chat history for one session."""
        main.chat_sessions.pop(session_id, None)
        return {"cleared": session_id}

    return router
