import { describe, expect, it } from "vitest";
import {
  MAX_MESSAGE_LENGTH,
  MAX_MESSAGES,
  parseConciseFlag,
  parseStreamFlag,
  validateChatRequest,
  validateMessages,
} from "./chat-validation";

describe("parseStreamFlag", () => {
  it("defaults to true", () => {
    expect(parseStreamFlag(undefined)).toBe(true);
    expect(parseStreamFlag(null)).toBe(true);
    expect(parseStreamFlag("true")).toBe(true);
  });

  it("treats false and string false as disabled", () => {
    expect(parseStreamFlag(false)).toBe(false);
    expect(parseStreamFlag("false")).toBe(false);
  });
});

describe("parseConciseFlag", () => {
  it("only enables on true boolean or string", () => {
    expect(parseConciseFlag(true)).toBe(true);
    expect(parseConciseFlag("true")).toBe(true);
    expect(parseConciseFlag(false)).toBe(false);
    expect(parseConciseFlag("false")).toBe(false);
    expect(parseConciseFlag(undefined)).toBe(false);
  });
});

describe("validateMessages", () => {
  it("requires a non-empty array", () => {
    expect(validateMessages(undefined)).toBe("Messages array is required");
    expect(validateMessages([])).toBe("Messages array is required");
  });

  it("requires alternating roles ending with a user message", () => {
    expect(validateMessages([{ role: "assistant", content: "Hi" }])).toBe(
      "Invalid message order at index 0"
    );

    expect(
      validateMessages([
        { role: "user", content: "Hello" },
        { role: "user", content: "Again" },
      ])
    ).toBe("Invalid message order at index 1");
  });

  it("rejects empty last user message", () => {
    expect(validateMessages([{ role: "user", content: "   " }])).toBe(
      "Last message must be a non-empty user message"
    );
  });

  it("rejects messages that are too long", () => {
    const longContent = "a".repeat(MAX_MESSAGE_LENGTH + 1);
    expect(validateMessages([{ role: "user", content: longContent }])).toBe(
      "Invalid message at index 0"
    );
  });

  it("rejects too many messages", () => {
    const messages = Array.from({ length: MAX_MESSAGES + 1 }, (_, i) => ({
      role: i % 2 === 0 ? "user" : "assistant",
      content: "hello",
    })) as Array<{ role: "user" | "assistant"; content: string }>;

    expect(validateMessages(messages)).toBe(
      `Too many messages (max ${MAX_MESSAGES})`
    );
  });

  it("accepts valid alternating history", () => {
    expect(
      validateMessages([
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi there" },
        { role: "user", content: "Thanks" },
      ])
    ).toBeNull();
  });
});

describe("validateChatRequest", () => {
  it("rejects invalid provider", () => {
    const result = validateChatRequest(
      {
        messages: [{ role: "user", content: "Hello" }],
        provider: "openai",
      },
      true
    );

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.error).toBe("Invalid AI provider");
      expect(result.status).toBe(400);
    }
  });

  it("rejects provider switching when disabled", () => {
    const result = validateChatRequest(
      {
        messages: [{ role: "user", content: "Hello" }],
        provider: "groq",
      },
      false
    );

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.error).toBe("AI provider switching is disabled");
      expect(result.status).toBe(403);
    }
  });

  it("accepts a valid streaming request", () => {
    const result = validateChatRequest(
      {
        messages: [{ role: "user", content: "Hello" }],
        stream: true,
        concise: true,
        provider: "gemini",
      },
      true
    );

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.data.stream).toBe(true);
      expect(result.data.concise).toBe(true);
      expect(result.data.provider).toBe("gemini");
    }
  });

  it("parses stream false from string", () => {
    const result = validateChatRequest(
      {
        messages: [{ role: "user", content: "Hello" }],
        stream: "false",
      },
      false
    );

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.data.stream).toBe(false);
    }
  });
});
