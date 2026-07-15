import { describe, expect, it, beforeEach } from "vitest";
import { checkRateLimit, resetRateLimitsForTests } from "./rate-limit";

describe("checkRateLimit", () => {
  beforeEach(() => {
    resetRateLimitsForTests();
  });

  it("allows requests under the limit", () => {
    expect(checkRateLimit("client-a")).toBe(true);
    expect(checkRateLimit("client-a")).toBe(true);
  });

  it("blocks requests after the limit is exceeded", () => {
    for (let i = 0; i < 30; i++) {
      expect(checkRateLimit("client-b")).toBe(true);
    }

    expect(checkRateLimit("client-b")).toBe(false);
  });

  it("tracks clients independently", () => {
    for (let i = 0; i < 30; i++) {
      checkRateLimit("client-c");
    }

    expect(checkRateLimit("client-c")).toBe(false);
    expect(checkRateLimit("client-d")).toBe(true);
  });
});
