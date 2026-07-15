/**
 * FastAPI learning UI — server-side config only.
 */

/** Base URL for the Python backend. Example: http://127.0.0.1:8000 */
export function getFastApiBaseUrl(): string | null {
  const raw = process.env.FASTAPI_URL?.replace(/['"]/g, "").trim();
  if (!raw) return null;
  return raw.replace(/\/+$/, "");
}

export function requireFastApiBaseUrl(): string {
  const url = getFastApiBaseUrl();
  if (!url) {
    throw new Error("FASTAPI_URL is not configured");
  }
  return url;
}
