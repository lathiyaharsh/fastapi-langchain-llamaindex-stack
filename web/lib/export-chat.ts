import type { ChatMessage } from "@/lib/ai/types";

/** Format chat history as plain text. */
export function formatChatAsText(messages: ChatMessage[]): string {
  return messages
    .map((message) => {
      const label = message.role === "user" ? "You" : "Assistant";
      return `${label}:\n${message.content}`;
    })
    .join("\n\n");
}

/** Format chat history as markdown. */
export function formatChatAsMarkdown(messages: ChatMessage[]): string {
  return messages
    .map((message) => {
      const label = message.role === "user" ? "## You" : "## Assistant";
      return `${label}\n\n${message.content}`;
    })
    .join("\n\n---\n\n");
}

/** Trigger a browser download of the chat history. */
export function downloadChat(
  messages: ChatMessage[],
  format: "txt" | "md"
): void {
  const content =
    format === "md"
      ? formatChatAsMarkdown(messages)
      : formatChatAsText(messages);
  const mimeType = format === "md" ? "text/markdown" : "text/plain";

  const blob = new Blob([content], { type: `${mimeType};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  const date = new Date().toISOString().slice(0, 10);

  anchor.href = url;
  anchor.download = `chat-export-${date}.${format}`;
  anchor.click();
  URL.revokeObjectURL(url);
}
