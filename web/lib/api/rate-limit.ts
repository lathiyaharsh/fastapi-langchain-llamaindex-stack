const WINDOW_MS = 60_000;
const MAX_REQUESTS = 30;

type RateLimitEntry = {
  count: number;
  resetAt: number;
};

const requestCounts = new Map<string, RateLimitEntry>();

/** Simple in-memory per-IP rate limiter for the chat API. */
export function checkRateLimit(clientId: string): boolean {
  const now = Date.now();
  const entry = requestCounts.get(clientId);

  if (!entry || now >= entry.resetAt) {
    requestCounts.set(clientId, { count: 1, resetAt: now + WINDOW_MS });
    return true;
  }

  if (entry.count >= MAX_REQUESTS) {
    return false;
  }

  entry.count += 1;
  return true;
}

/** Resolve a stable client identifier from proxy-aware request headers. */
export function getClientIp(headers: Headers): string {
  const forwarded = headers.get("x-forwarded-for");
  if (forwarded) {
    return forwarded.split(",")[0]?.trim() || "unknown";
  }

  return headers.get("x-real-ip")?.trim() || "unknown";
}

/** Test helper — reset in-memory counters. */
export function resetRateLimitsForTests(): void {
  requestCounts.clear();
}
