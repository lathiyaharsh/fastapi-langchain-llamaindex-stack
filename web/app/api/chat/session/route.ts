/**
 * API route: clear FastAPI chat session memory.
 * DELETE /api/chat/session?session_id=...
 */
import { NextRequest } from "next/server";
import { clearFastApiSession } from "@/lib/api/fastapi-client";
import { parseSessionId } from "@/lib/api/chat-validation";

export const runtime = "nodejs";

export async function DELETE(request: NextRequest) {
  try {
    const sessionParam =
      request.nextUrl.searchParams.get("session_id") ??
      request.nextUrl.searchParams.get("sessionId");
    const sessionResult = parseSessionId(sessionParam);
    if (sessionResult.error) {
      return Response.json({ error: sessionResult.error }, { status: 400 });
    }

    const result = await clearFastApiSession("chat", sessionResult.sessionId);
    if (!result.ok) {
      return Response.json({ error: result.error }, { status: result.status });
    }

    return Response.json({ cleared: sessionResult.sessionId });
  } catch (error) {
    console.error("Chat session clear error:", error);
    return Response.json(
      { error: "Failed to clear chat session" },
      { status: 500 }
    );
  }
}
