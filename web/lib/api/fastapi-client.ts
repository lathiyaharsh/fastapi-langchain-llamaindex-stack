/**
 * Server-side adapter: Next.js ↔ FastAPI learning backend.
 *
 * Responsibilities:
 * - Map Next.js chat payloads to FastAPI /chat and /chat/stream
 * - Translate FastAPI plain-text SSE into JSON SSE (meta/chunk/done/error)
 * - Proxy RAG questions and session clears
 * - Keep API keys / FastAPI URL server-side only
 */
import type { ChatMessage } from "@/lib/ai/types";
import { getFastApiBaseUrl } from "@/lib/config";
import { encodeSSE } from "@/lib/sse";

const GENERIC_UPSTREAM_ERROR = "Failed to reach the FastAPI backend";

export type FastApiChatPayload = {
  message: string;
  session_id: string;
  reply_mode: "concise" | "detailed";
  history: Array<{ role: "user" | "assistant"; content: string }>;
};

export type FastApiRagResponse = {
  answer: string;
  sources: string[];
};

/** Trim trailing slash from base URL. */
export function normalizeFastApiBaseUrl(raw: string): string {
  return raw.trim().replace(/\/+$/, "");
}

/**
 * Split Next.js messages into FastAPI fields:
 * - message = last user turn
 * - history = everything before that (for provider-switch sync)
 */
export function mapMessagesToFastApiChat(
  messages: ChatMessage[],
  sessionId: string,
  concise: boolean
): FastApiChatPayload {
  if (messages.length === 0) {
    throw new Error("Messages array is required");
  }

  const last = messages[messages.length - 1];
  if (last.role !== "user" || !last.content.trim()) {
    throw new Error("Last message must be a non-empty user message");
  }

  return {
    message: last.content.trim(),
    session_id: sessionId,
    reply_mode: concise ? "concise" : "detailed",
    history: messages.slice(0, -1).map((m) => ({
      role: m.role,
      content: m.content,
    })),
  };
}

/**
 * Parse one FastAPI SSE data payload into a Next.js-compatible event object,
 * or null if the line should be skipped.
 */
export function translateFastApiSsePayload(
  payload: string
): object | null {
  if (!payload || payload === "[DONE]") {
    return payload === "[DONE]" ? { type: "done", provider: "groq" } : null;
  }

  if (payload.startsWith("[ERROR]")) {
    const msg = payload.slice("[ERROR]".length).trim() || GENERIC_UPSTREAM_ERROR;
    return { type: "error", error: msg };
  }

  // FastAPI escapes real newlines as "\\n" so each SSE event stays one line.
  const content = payload.replace(/\\n/g, "\n");
  return { type: "chunk", content };
}

/**
 * Convert a FastAPI text/event-stream body into Next.js JSON-SSE bytes.
 * Emits meta first, then translated chunks, then done (if FastAPI sent [DONE]).
 */
export function createFastApiTranslationStream(
  upstream: ReadableStream<Uint8Array>,
  signal?: AbortSignal
): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();

  return new ReadableStream({
    async start(controller) {
      const enqueue = (event: object) => {
        controller.enqueue(encoder.encode(encodeSSE(event)));
      };

      enqueue({ type: "meta", provider: "groq" });

      const reader = upstream.getReader();
      let buffer = "";
      let sawDone = false;

      const abortReader = () => {
        reader.cancel().catch(() => {});
      };
      signal?.addEventListener("abort", abortReader);

      try {
        while (true) {
          if (signal?.aborted) break;

          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith("data:")) continue;

            const payload = trimmed.slice(5).trim();
            const event = translateFastApiSsePayload(payload);
            if (!event) continue;

            if ((event as { type?: string }).type === "done") {
              sawDone = true;
            }
            enqueue(event);

            if ((event as { type?: string }).type === "error") {
              controller.close();
              return;
            }
          }
        }

        if (!sawDone && !signal?.aborted) {
          enqueue({ type: "done", provider: "groq" });
        }
        controller.close();
      } catch (error) {
        if (!signal?.aborted) {
          enqueue({
            type: "error",
            error:
              error instanceof Error
                ? error.message
                : GENERIC_UPSTREAM_ERROR,
          });
        }
        controller.close();
      } finally {
        signal?.removeEventListener("abort", abortReader);
        reader.releaseLock();
      }
    },
    cancel() {
      // Browser / Next abort — FastAPI disconnect handling picks this up.
    },
  });
}

async function parseFastApiError(response: Response): Promise<string> {
  try {
    const data = (await response.json()) as {
      detail?: string | Array<{ msg?: string }>;
    };
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail) && data.detail[0]?.msg) {
      return data.detail[0].msg;
    }
  } catch {
    // ignore
  }
  return GENERIC_UPSTREAM_ERROR;
}

/** Stream Groq chat from FastAPI, translated to JSON SSE. */
export async function streamChatFromFastApi(options: {
  messages: ChatMessage[];
  sessionId: string;
  concise: boolean;
  signal?: AbortSignal;
}): Promise<Response> {
  const base = getFastApiBaseUrl();
  if (!base) {
    return Response.json(
      { error: "FASTAPI_URL is not configured" },
      { status: 500 }
    );
  }

  const payload = mapMessagesToFastApiChat(
    options.messages,
    options.sessionId,
    options.concise
  );

  let upstream: Response;
  try {
    upstream = await fetch(`${base}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: options.signal,
    });
  } catch (error) {
    if (options.signal?.aborted) {
      return new Response(null, { status: 499 });
    }
    console.error("FastAPI chat stream fetch failed:", error);
    return Response.json({ error: GENERIC_UPSTREAM_ERROR }, { status: 502 });
  }

  if (!upstream.ok) {
    const error = await parseFastApiError(upstream);
    return Response.json({ error }, { status: upstream.status });
  }

  if (!upstream.body) {
    return Response.json({ error: "Empty stream from FastAPI" }, { status: 502 });
  }

  return new Response(
    createFastApiTranslationStream(upstream.body, options.signal),
    {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
        "X-Accel-Buffering": "no",
      },
    }
  );
}

/** Non-streaming Groq chat via FastAPI. */
export async function generateChatFromFastApi(options: {
  messages: ChatMessage[];
  sessionId: string;
  concise: boolean;
  signal?: AbortSignal;
}): Promise<{ message: string; provider: "groq" } | { error: string; status: number }> {
  const base = getFastApiBaseUrl();
  if (!base) {
    return { error: "FASTAPI_URL is not configured", status: 500 };
  }

  const payload = mapMessagesToFastApiChat(
    options.messages,
    options.sessionId,
    options.concise
  );

  let upstream: Response;
  try {
    upstream = await fetch(`${base}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: options.signal,
    });
  } catch (error) {
    console.error("FastAPI chat fetch failed:", error);
    return { error: GENERIC_UPSTREAM_ERROR, status: 502 };
  }

  if (!upstream.ok) {
    return {
      error: await parseFastApiError(upstream),
      status: upstream.status,
    };
  }

  const data = (await upstream.json()) as { reply?: string };
  if (typeof data.reply !== "string") {
    return { error: "Invalid response from FastAPI", status: 502 };
  }

  return { message: data.reply, provider: "groq" };
}

/** Ask docs via FastAPI /rag. */
export async function askRagFromFastApi(options: {
  question: string;
  sessionId: string;
  history?: Array<{ role: "user" | "assistant"; content: string }> | null;
  signal?: AbortSignal;
}): Promise<
  | { ok: true; data: FastApiRagResponse }
  | { ok: false; error: string; status: number }
> {
  const base = getFastApiBaseUrl();
  if (!base) {
    return { ok: false, error: "FASTAPI_URL is not configured", status: 500 };
  }

  let upstream: Response;
  try {
    upstream = await fetch(`${base}/rag`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: options.question,
        session_id: options.sessionId,
        ...(options.history != null ? { history: options.history } : {}),
      }),
      signal: options.signal,
    });
  } catch (error) {
    console.error("FastAPI RAG fetch failed:", error);
    return { ok: false, error: GENERIC_UPSTREAM_ERROR, status: 502 };
  }

  if (!upstream.ok) {
    return {
      ok: false,
      error: await parseFastApiError(upstream),
      status: upstream.status,
    };
  }

  const data = (await upstream.json()) as FastApiRagResponse;
  if (typeof data.answer !== "string" || !Array.isArray(data.sources)) {
    return { ok: false, error: "Invalid RAG response from FastAPI", status: 502 };
  }

  return { ok: true, data };
}

export type FastApiRagUploadResponse = {
  status: string;
  filename: string;
  files_seen: number;
  data_dir: string;
};

/** Upload and incrementally insert one doc into the RAG index. */
export async function uploadRagDocumentFromFastApi(options: {
  file: Blob;
  filename: string;
  signal?: AbortSignal;
}): Promise<
  | { ok: true; data: FastApiRagUploadResponse }
  | { ok: false; error: string; status: number }
> {
  const base = getFastApiBaseUrl();
  if (!base) {
    return { ok: false, error: "FASTAPI_URL is not configured", status: 500 };
  }

  const formData = new FormData();
  formData.append("file", options.file, options.filename);

  let upstream: Response;
  try {
    upstream = await fetch(`${base}/rag/upload`, {
      method: "POST",
      body: formData,
      signal: options.signal,
    });
  } catch (error) {
    console.error("FastAPI RAG upload failed:", error);
    return { ok: false, error: GENERIC_UPSTREAM_ERROR, status: 502 };
  }

  if (!upstream.ok) {
    return {
      ok: false,
      error: await parseFastApiError(upstream),
      status: upstream.status,
    };
  }

  const data = (await upstream.json()) as FastApiRagUploadResponse;
  if (
    typeof data.filename !== "string" ||
    typeof data.files_seen !== "number" ||
    typeof data.status !== "string"
  ) {
    return { ok: false, error: "Invalid upload response from FastAPI", status: 502 };
  }

  return { ok: true, data };
}

/** Clear FastAPI chat or RAG session memory. */
export async function clearFastApiSession(
  kind: "chat" | "rag",
  sessionId: string
): Promise<{ ok: true } | { ok: false; error: string; status: number }> {
  const base = getFastApiBaseUrl();
  if (!base) {
    return { ok: false, error: "FASTAPI_URL is not configured", status: 500 };
  }

  const path =
    kind === "chat"
      ? `/chat/session/${encodeURIComponent(sessionId)}`
      : `/rag/session/${encodeURIComponent(sessionId)}`;

  try {
    const upstream = await fetch(`${base}${path}`, { method: "DELETE" });
    if (!upstream.ok) {
      return {
        ok: false,
        error: await parseFastApiError(upstream),
        status: upstream.status,
      };
    }
    return { ok: true };
  } catch (error) {
    console.error("FastAPI session clear failed:", error);
    return { ok: false, error: GENERIC_UPSTREAM_ERROR, status: 502 };
  }
}
