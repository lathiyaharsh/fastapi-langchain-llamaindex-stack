import { describe, expect, it } from "vitest";
import {
  mapMessagesToFastApiChat,
  translateFastApiSsePayload,
} from "@/lib/api/fastapi-client";
import type { ChatMessage } from "@/lib/ai/types";

describe("mapMessagesToFastApiChat", () => {
  it("maps last user message and prior history", () => {
    const messages: ChatMessage[] = [
      { role: "user", content: "Hi" },
      { role: "assistant", content: "Hello" },
      { role: "user", content: "How are you?" },
    ];

    expect(mapMessagesToFastApiChat(messages, "sess-1", true)).toEqual({
      message: "How are you?",
      session_id: "sess-1",
      reply_mode: "concise",
      history: [
        { role: "user", content: "Hi" },
        { role: "assistant", content: "Hello" },
      ],
    });
  });

  it("uses detailed reply mode when concise is false", () => {
    const messages: ChatMessage[] = [{ role: "user", content: "Hello" }];
    expect(mapMessagesToFastApiChat(messages, "s", false).reply_mode).toBe(
      "detailed"
    );
  });

  it("rejects empty messages", () => {
    expect(() => mapMessagesToFastApiChat([], "s", true)).toThrow(
      /Messages array is required/
    );
  });
});

describe("translateFastApiSsePayload", () => {
  it("translates plain text chunk and restores newlines", () => {
    expect(translateFastApiSsePayload("Hello\\nworld")).toEqual({
      type: "chunk",
      content: "Hello\nworld",
    });
  });

  it("translates DONE marker", () => {
    expect(translateFastApiSsePayload("[DONE]")).toEqual({
      type: "done",
      provider: "groq",
    });
  });

  it("translates ERROR marker", () => {
    expect(translateFastApiSsePayload("[ERROR] boom")).toEqual({
      type: "error",
      error: "boom",
    });
  });

  it("returns null for empty payload", () => {
    expect(translateFastApiSsePayload("")).toBeNull();
  });
});
