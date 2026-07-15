/**
 * POST /api/chat — always proxies to FastAPI /chat/stream (Groq + LangChain).
 */
import { NextRequest } from "next/server";
import {
  parseStreamFlag,
  validateChatRequest,
} from "@/lib/api/chat-validation";
import {
  generateChatFromFastApi,
  streamChatFromFastApi,
} from "@/lib/api/fastapi-client";
import { checkRateLimit, getClientIp } from "@/lib/api/rate-limit";
import { createSseErrorResponse, SSE_HEADERS } from "@/lib/sse";

export const runtime = "nodejs";

const GENERIC_ERROR = "Failed to generate response";
const RATE_LIMIT_ERROR = "Too many requests. Please try again later.";

function errorResponse(
  error: string,
  status: number,
  stream: boolean
): Response {
  if (stream) {
    return createSseErrorResponse(error, status);
  }
  return Response.json({ error }, { status });
}

function getStreamFlagFromBody(body: unknown): boolean {
  if (typeof body !== "object" || body === null) return true;
  return parseStreamFlag((body as Record<string, unknown>).stream);
}

export async function POST(request: NextRequest) {
  let stream = true;

  try {
    const body = await request.json();
    stream = getStreamFlagFromBody(body);

    const clientIp = getClientIp(request.headers);
    if (!checkRateLimit(clientIp)) {
      return errorResponse(RATE_LIMIT_ERROR, 429, stream);
    }

    const validation = validateChatRequest(body, false);
    if (!validation.ok) {
      return errorResponse(validation.error, validation.status, stream);
    }

    const {
      messages,
      concise,
      stream: useStream,
      sessionId,
    } = validation.data;
    stream = useStream;

    if (useStream) {
      const fastApiResponse = await streamChatFromFastApi({
        messages,
        sessionId,
        concise,
        signal: request.signal,
      });

      if (!fastApiResponse.ok) {
        let error = GENERIC_ERROR;
        try {
          const data = (await fastApiResponse.json()) as { error?: string };
          if (data.error) error = data.error;
        } catch {
          // ignore
        }
        return errorResponse(error, fastApiResponse.status, true);
      }

      return fastApiResponse;
    }

    const result = await generateChatFromFastApi({
      messages,
      sessionId,
      concise,
      signal: request.signal,
    });

    if ("error" in result) {
      return errorResponse(result.error, result.status, false);
    }

    return Response.json(result);
  } catch (error) {
    console.error("Chat API error:", error);
    return errorResponse(GENERIC_ERROR, 500, stream);
  }
}

void SSE_HEADERS;
