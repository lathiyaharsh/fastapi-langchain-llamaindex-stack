/**
 * Client-side parser for Server-Sent Events (SSE) from /api/chat and /api/rag.
 *
 * The server sends lines like:
 * data: {"type":"chunk","content":"Hello"}
 */
import type { AIProvider } from "@/lib/ai/types";

export type ChatStreamEvent =
  | { type: "meta"; provider: AIProvider }
  | { type: "chunk"; content: string; sources?: string[] }
  | { type: "done"; provider: AIProvider; sources?: string[] }
  | { type: "error"; error: string };

/** Read the fetch response body and yield parsed SSE events one by one. */
export async function* readChatStream(
  response: Response,
  signal?: AbortSignal
): AsyncGenerator<ChatStreamEvent> {
  if (!response.body) {
    throw new Error("No response body");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const abortReader = () => {
    reader.cancel().catch(() => {
      // Ignore cancel errors when the stream is already closed.
    });
  };

  signal?.addEventListener("abort", abortReader);

  try {
    while (true) {
      if (signal?.aborted) break;

      const { done, value } = await reader.read();
      if (done) break;

      // SSE chunks may arrive split across multiple network packets.
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;

        const payload = trimmed.slice(5).trim();
        if (!payload) continue;

        try {
          yield JSON.parse(payload) as ChatStreamEvent;
        } catch {
          // Skip malformed events.
        }
      }
    }
  } finally {
    signal?.removeEventListener("abort", abortReader);
    reader.releaseLock();
  }
}
