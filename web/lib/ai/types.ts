/**
 * Shared TypeScript types used by the frontend and backend.
 */
export type ChatRole = "user" | "assistant";

/** App mode: general chat vs Ask My Docs (RAG). */
export type ChatMode = "chat" | "docs";

/** One message in the conversation history. */
export type ChatMessage = {
  role: ChatRole;
  content: string;
  /** Retrieved chunk previews — only set for RAG assistant replies. */
  sources?: string[];
};

/** Supported AI backends. */
export type AIProvider = "groq" | "gemini" | "huggingface";

/** Provider order also affects fallback order after the primary provider. */
export const AI_PROVIDERS: AIProvider[] = [
  "groq",
  "huggingface",
  "gemini",
] as const;

/** Successful non-streaming API response shape. */
export type ChatResponse = {
  message: string;
  provider: AIProvider;
};

/** Error response shape. */
export type ChatError = {
  error: string;
  provider?: AIProvider;
};

/** Human-readable provider name for the UI. */
export function getProviderLabel(provider: AIProvider): string {
  switch (provider) {
    case "groq":
      return "Groq";
    case "gemini":
      return "Gemini";
    case "huggingface":
      return "Hugging Face";
  }
}
