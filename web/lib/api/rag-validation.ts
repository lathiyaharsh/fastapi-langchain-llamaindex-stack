/**
 * Validate POST /api/rag bodies and DELETE session params.
 */
import {
  MAX_MESSAGE_LENGTH,
  MAX_MESSAGES,
  parseSessionId,
} from "@/lib/api/chat-validation";
import type { ChatMessage } from "@/lib/ai/types";

export type ValidatedRagRequest = {
  question: string;
  sessionId: string;
  /** Prior turns only (excluding the current question). */
  history: ChatMessage[] | null;
};

export type RagValidationResult =
  | { ok: true; data: ValidatedRagRequest }
  | { ok: false; error: string; status: number };

function parseOptionalHistory(value: unknown): {
  history: ChatMessage[] | null;
  error?: string;
} {
  if (value === undefined || value === null) {
    return { history: null };
  }

  if (!Array.isArray(value)) {
    return { history: null, error: "history must be an array" };
  }

  if (value.length > MAX_MESSAGES) {
    return {
      history: null,
      error: `Too many history messages (max ${MAX_MESSAGES})`,
    };
  }

  const history: ChatMessage[] = [];
  for (let i = 0; i < value.length; i++) {
    const item = value[i];
    if (typeof item !== "object" || item === null) {
      return { history: null, error: `Invalid history item at index ${i}` };
    }
    const msg = item as Record<string, unknown>;
    if (msg.role !== "user" && msg.role !== "assistant") {
      return { history: null, error: `Invalid history role at index ${i}` };
    }
    if (typeof msg.content !== "string" || !msg.content.trim()) {
      return { history: null, error: `Invalid history content at index ${i}` };
    }
    if (msg.content.length > MAX_MESSAGE_LENGTH) {
      return {
        history: null,
        error: `History message too long at index ${i}`,
      };
    }
    history.push({ role: msg.role, content: msg.content.trim() });
  }

  return { history };
}

export function validateRagRequest(body: unknown): RagValidationResult {
  if (typeof body !== "object" || body === null) {
    return { ok: false, error: "Invalid request body", status: 400 };
  }

  const payload = body as Record<string, unknown>;
  const question =
    typeof payload.question === "string"
      ? payload.question.trim()
      : typeof payload.message === "string"
        ? payload.message.trim()
        : "";

  if (!question) {
    return { ok: false, error: "Question cannot be empty", status: 400 };
  }

  if (question.length > MAX_MESSAGE_LENGTH) {
    return {
      ok: false,
      error: `Question too long (max ${MAX_MESSAGE_LENGTH})`,
      status: 400,
    };
  }

  const sessionResult = parseSessionId(
    payload.session_id ?? payload.sessionId
  );
  if (sessionResult.error) {
    return { ok: false, error: sessionResult.error, status: 400 };
  }

  const historyResult = parseOptionalHistory(payload.history);
  if (historyResult.error) {
    return { ok: false, error: historyResult.error, status: 400 };
  }

  return {
    ok: true,
    data: {
      question,
      sessionId: sessionResult.sessionId,
      history: historyResult.history,
    },
  };
}
