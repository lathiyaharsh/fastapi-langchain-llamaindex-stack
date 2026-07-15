/**
 * Saves chat state in the browser so history survives page refresh.
 * This runs only on the client (localStorage is not available on the server).
 *
 * Stores separate histories for Chat vs Ask My Docs, plus FastAPI session IDs.
 */
import type { AIProvider, ChatMessage, ChatMode } from "@/lib/ai/types";

const STORAGE_KEY = "ai-chat-storage";

export type ModeThread = {
  messages: ChatMessage[];
  sessionId: string;
};

export type StoredChat = {
  mode: ChatMode;
  chat: ModeThread;
  docs: ModeThread;
  selectedProvider: AIProvider;
  conciseMode: boolean;
  activeProvider: AIProvider | null;
};

/** Create a stable random session id for FastAPI memory. */
export function createSessionId(prefix: string): string {
  const rand =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID().replace(/-/g, "").slice(0, 16)
      : `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`;
  return `${prefix}_${rand}`.slice(0, 64);
}

function isChatMode(value: unknown): value is ChatMode {
  return value === "chat" || value === "docs";
}

function normalizeMessages(value: unknown): ChatMessage[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is ChatMessage => {
    if (typeof item !== "object" || item === null) return false;
    const msg = item as ChatMessage;
    return (
      (msg.role === "user" || msg.role === "assistant") &&
      typeof msg.content === "string"
    );
  });
}

function normalizeThread(
  value: unknown,
  prefix: string,
  legacyMessages?: ChatMessage[]
): ModeThread {
  if (value && typeof value === "object") {
    const thread = value as Partial<ModeThread>;
    return {
      messages: normalizeMessages(thread.messages),
      sessionId:
        typeof thread.sessionId === "string" && thread.sessionId.trim()
          ? thread.sessionId.trim().slice(0, 64)
          : createSessionId(prefix),
    };
  }

  return {
    messages: legacyMessages ?? [],
    sessionId: createSessionId(prefix),
  };
}

/** Load previous chat from localStorage (supports legacy flat shape). */
export function loadStoredChat(): StoredChat | null {
  if (typeof window === "undefined") return null;

  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;

    const parsed = JSON.parse(raw) as Record<string, unknown>;

    // Legacy: { messages, selectedProvider, conciseMode, activeProvider }
    const legacyMessages = Array.isArray(parsed.messages)
      ? normalizeMessages(parsed.messages)
      : undefined;

    const chat = normalizeThread(parsed.chat, "chat", legacyMessages);
    const docs = normalizeThread(parsed.docs, "docs");

    return {
      mode: isChatMode(parsed.mode) ? parsed.mode : "chat",
      chat,
      docs,
      selectedProvider: (parsed.selectedProvider as AIProvider) ?? "groq",
      conciseMode: Boolean(parsed.conciseMode),
      activeProvider: (parsed.activeProvider as AIProvider | null) ?? null,
    };
  } catch {
    return null;
  }
}

/** Save current chat to localStorage. */
export function saveStoredChat(data: StoredChat): void {
  if (typeof window === "undefined") return;

  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  } catch {
    // Ignore quota or serialization errors.
  }
}

/** Remove saved chat from localStorage. */
export function clearStoredChat(): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem(STORAGE_KEY);
}
