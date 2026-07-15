import { describe, expect, it } from "vitest";
import { createSseErrorResponse, encodeSSE } from "./sse";

describe("encodeSSE", () => {
  it("formats SSE data lines", () => {
    expect(encodeSSE({ type: "error", error: "Bad request" })).toBe(
      'data: {"type":"error","error":"Bad request"}\n\n'
    );
  });
});

describe("createSseErrorResponse", () => {
  it("returns an SSE response with the expected headers", async () => {
    const response = createSseErrorResponse("Invalid request", 400);

    expect(response.status).toBe(400);
    expect(response.headers.get("Content-Type")).toBe("text/event-stream");
    expect(response.headers.get("X-Accel-Buffering")).toBe("no");

    const body = await response.text();
    expect(body).toContain('"type":"error"');
    expect(body).toContain("Invalid request");
  });
});
