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

  it("accepts optional history", () => {
    const result = validateRagRequest({
      question: "Who set it?",
      session_id: "docs_1",
      history: [
        { role: "user", content: "What is the fridge password?" },
        { role: "assistant", content: "BANANA-42" },
      ],
    });
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.data.history).toHaveLength(2);
      expect(result.data.history?.[0].content).toBe(
        "What is the fridge password?"
      );
    }
  });

  it("rejects invalid history role", () => {
    const result = validateRagRequest({
      question: "Hello",
      history: [{ role: "system", content: "nope" }],
    });
    expect(result.ok).toBe(false);
  });
});
