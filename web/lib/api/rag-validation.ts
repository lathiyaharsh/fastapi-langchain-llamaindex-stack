/**
 * Validate POST /api/rag bodies and DELETE session params.
 */
import { MAX_MESSAGE_LENGTH, parseSessionId } from "@/lib/api/chat-validation";

export type ValidatedRagRequest = {
  question: string;
  sessionId: string;
};

export type RagValidationResult =
  | { ok: true; data: ValidatedRagRequest }
  | { ok: false; error: string; status: number };

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

  return {
    ok: true,
    data: {
      question,
      sessionId: sessionResult.sessionId,
    },
  };
}
