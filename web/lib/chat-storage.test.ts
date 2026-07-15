import { describe, expect, it, beforeEach, afterEach } from "vitest";
import {
  createSessionId,
  loadStoredChat,
  saveStoredChat,
  clearStoredChat,
} from "@/lib/chat-storage";

function installLocalStorageMock() {
  const store = new Map<string, string>();
  const localStorageMock = {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => {
      store.set(key, value);
    },
    removeItem: (key: string) => {
      store.delete(key);
    },
    clear: () => store.clear(),
  };
  Object.defineProperty(globalThis, "window", {
    value: { localStorage: localStorageMock },
    configurable: true,
  });
  Object.defineProperty(globalThis, "localStorage", {
    value: localStorageMock,
    configurable: true,
  });
  return store;
}

describe("chat-storage", () => {
  beforeEach(() => {
    installLocalStorageMock();
    clearStoredChat();
  });

  afterEach(() => {
    clearStoredChat();
  });

  it("creates session ids with prefix", () => {
    expect(createSessionId("chat")).toMatch(/^chat_/);
    expect(createSessionId("docs").length).toBeLessThanOrEqual(64);
  });

  it("migrates legacy flat storage shape", () => {
    localStorage.setItem(
      "ai-chat-storage",
      JSON.stringify({
        messages: [{ role: "user", content: "Hi" }],
        selectedProvider: "groq",
        conciseMode: true,
        activeProvider: "groq",
      })
    );

    const stored = loadStoredChat();
    expect(stored).not.toBeNull();
    expect(stored?.mode).toBe("chat");
    expect(stored?.chat.messages).toEqual([{ role: "user", content: "Hi" }]);
    expect(stored?.docs.messages).toEqual([]);
    expect(stored?.chat.sessionId).toMatch(/^chat_/);
    expect(stored?.docs.sessionId).toMatch(/^docs_/);
  });

  it("round-trips new storage shape", () => {
    saveStoredChat({
      mode: "docs",
      chat: { messages: [], sessionId: "chat_abc" },
      docs: {
        messages: [{ role: "user", content: "password?" }],
        sessionId: "docs_xyz",
      },
      selectedProvider: "gemini",
      conciseMode: false,
      activeProvider: null,
    });

    const stored = loadStoredChat();
    expect(stored?.mode).toBe("docs");
    expect(stored?.docs.sessionId).toBe("docs_xyz");
    expect(stored?.docs.messages[0]?.content).toBe("password?");
  });
});
