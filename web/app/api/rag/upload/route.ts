/**
 * POST /api/rag/upload — multipart file → FastAPI incremental RAG insert
 */
import { NextRequest } from "next/server";
import { uploadRagDocumentFromFastApi } from "@/lib/api/fastapi-client";
import { checkRateLimit, getClientIp } from "@/lib/api/rate-limit";

export const runtime = "nodejs";

const GENERIC_ERROR = "Failed to upload document";
const RATE_LIMIT_ERROR = "Too many requests. Please try again later.";
const ALLOWED_SUFFIXES = [".md", ".txt"];

export async function POST(request: NextRequest) {
  try {
    const clientIp = getClientIp(request.headers);
    if (!checkRateLimit(clientIp)) {
      return Response.json({ error: RATE_LIMIT_ERROR }, { status: 429 });
    }

    const formData = await request.formData();
    const file = formData.get("file");
    if (!(file instanceof File)) {
      return Response.json({ error: "file is required" }, { status: 400 });
    }

    const lowerName = file.name.toLowerCase();
    if (!ALLOWED_SUFFIXES.some((suffix) => lowerName.endsWith(suffix))) {
      return Response.json(
        { error: "Only .md and .txt files are allowed" },
        { status: 400 }
      );
    }

    const result = await uploadRagDocumentFromFastApi({
      file,
      filename: file.name,
      signal: request.signal,
    });

    if (!result.ok) {
      return Response.json({ error: result.error }, { status: result.status });
    }

    return Response.json(result.data);
  } catch (error) {
    console.error("RAG upload API error:", error);
    return Response.json({ error: GENERIC_ERROR }, { status: 500 });
  }
}
