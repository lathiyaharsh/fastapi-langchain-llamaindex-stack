import { describe, expect, it } from "vitest";
import { validateRagRequest } from "@/lib/api/rag-validation";

describe("validateRagRequest", () => {
  it("accepts question + session_id", () => {
    const result = validateRagRequest({
      question: "What is the fridge password?",
      session_id: "docs_1",
    });
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.data.question).toBe("What is the fridge password?");
      expect(result.data.sessionId).toBe("docs_1");
    }
  });

  it("rejects empty question", () => {
    const result = validateRagRequest({ question: "   " });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(400);
    }
  });

  it("rejects invalid session id", () => {
    const result = validateRagRequest({
      question: "Hello",
      session_id: "bad session!",
    });
    expect(result.ok).toBe(false);
  });
});
