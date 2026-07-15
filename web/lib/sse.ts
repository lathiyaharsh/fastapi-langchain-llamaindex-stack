/** Format one Server-Sent Events (SSE) data line. */
export function encodeSSE(data: object): string {
  return `data: ${JSON.stringify(data)}\n\n`;
}

export const SSE_HEADERS: Record<string, string> = {
  "Content-Type": "text/event-stream",
  "Cache-Control": "no-cache",
  Connection: "keep-alive",
  "X-Accel-Buffering": "no",
};

/** Build an HTTP response that streams one or more SSE events. */
export function createSseResponse(events: object[], status = 200): Response {
  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    start(controller) {
      for (const event of events) {
        controller.enqueue(encoder.encode(encodeSSE(event)));
      }
      controller.close();
    },
  });

  return new Response(stream, { status, headers: SSE_HEADERS });
}

/** Build an SSE response containing a single error event. */
export function createSseErrorResponse(error: string, status = 500): Response {
  return createSseResponse([{ type: "error", error }], status);
}
