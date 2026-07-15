/**
 * API route: Ask My Docs (RAG) via FastAPI.
 *
 * POST /api/rag  — question → FastAPI /rag → synthetic JSON SSE (meta/chunk/done)
 * DELETE /api/rag?session_id=... — clear FastAPI RAG session memory
 */
import { NextRequest } from "next/server";
import {
  askRagFromFastApi,
  clearFastApiSession,
} from "@/lib/api/fastapi-client";
import { parseSessionId } from "@/lib/api/chat-validation";
import { validateRagRequest } from "@/lib/api/rag-validation";
import { checkRateLimit, getClientIp } from "@/lib/api/rate-limit";
import { createSseErrorResponse, createSseResponse } from "@/lib/sse";

export const runtime = "nodejs";

const GENERIC_ERROR = "Failed to answer from documents";
const RATE_LIMIT_ERROR = "Too many requests. Please try again later.";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const clientIp = getClientIp(request.headers);
    if (!checkRateLimit(clientIp)) {
      return createSseErrorResponse(RATE_LIMIT_ERROR, 429);
    }

    const validation = validateRagRequest(body);
    if (!validation.ok) {
      return createSseErrorResponse(validation.error, validation.status);
    }

    const result = await askRagFromFastApi({
      question: validation.data.question,
      sessionId: validation.data.sessionId,
      history: validation.data.history,
      signal: request.signal,
    });

    if (!result.ok) {
      return createSseErrorResponse(result.error, result.status);
    }

    // RAG is non-streaming on FastAPI — emit one chunk so Chat.tsx can reuse SSE.
    return createSseResponse(
      [
        { type: "meta", provider: "groq" },
        {
          type: "chunk",
          content: result.data.answer,
          sources: result.data.sources,
        },
        { type: "done", provider: "groq", sources: result.data.sources },
      ],
      200
    );
  } catch (error) {
    console.error("RAG API error:", error);
    return createSseErrorResponse(GENERIC_ERROR, 500);
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const sessionParam =
      request.nextUrl.searchParams.get("session_id") ??
      request.nextUrl.searchParams.get("sessionId");
    const sessionResult = parseSessionId(sessionParam);
    if (sessionResult.error) {
      return Response.json({ error: sessionResult.error }, { status: 400 });
    }

    const result = await clearFastApiSession("rag", sessionResult.sessionId);
    if (!result.ok) {
      return Response.json({ error: result.error }, { status: result.status });
    }

    return Response.json({ cleared: sessionResult.sessionId });
  } catch (error) {
    console.error("RAG session clear error:", error);
    return Response.json({ error: GENERIC_ERROR }, { status: 500 });
  }
}
