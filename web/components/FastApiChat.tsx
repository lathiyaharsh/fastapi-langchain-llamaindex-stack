/**
 * Main chat UI (client component).
 *
 * Modes:
 * - Chat → POST /api/chat (Groq via FastAPI when configured; Gemini/HF stay on Next)
 * - Ask My Docs → POST /api/rag (FastAPI LlamaIndex RAG)
 * - Upload → POST /api/rag/upload (.md / .txt into backend/data)
 *
 * Flow:
 * 1. User types a message or clicks a suggestion
 * 2. Frontend streams SSE events back
 * 3. UI updates the assistant bubble while streaming
 * 4. Histories + session IDs are saved to localStorage (per mode)
 */
"use client";

import Image from "next/image";
import { useEffect, useRef, useState, type ChangeEvent } from "react";
import MarkdownContent from "@/components/MarkdownContent";
import type { AIProvider, ChatMessage, ChatMode } from "@/lib/ai/types";
import { getProviderLabel } from "@/lib/ai/types";
import {
  createSessionId,
  loadStoredChat,
  saveStoredChat,
  type ModeThread,
} from "@/lib/chat-storage";
import { downloadChat } from "@/lib/export-chat";
import { readChatStream } from "@/lib/sse-client";

const ASSISTANT_AVATAR = "/assistant-avatar.svg";

/** Small avatar shown next to assistant messages. */
function AssistantAvatar({ size = 32 }: { size?: number }) {
  return (
    <Image
      src={ASSISTANT_AVATAR}
      alt="Assistant"
      width={size}
      height={size}
      className="shrink-0 rounded-full border border-[var(--color-border)] bg-[var(--color-surface-elevated)]"
    />
  );
}

/** Animated dots shown before the first stream chunk arrives. */
function TypingIndicator() {
  return (
    <div className="flex items-start gap-3">
      <AssistantAvatar />
      <div
        className="flex items-center gap-1 rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-assistant-bubble)] px-4 py-3"
        aria-label="Assistant is typing"
      >
        <span className="typing-dot h-2 w-2 rounded-full bg-[var(--color-text-muted)]" />
        <span className="typing-dot h-2 w-2 rounded-full bg-[var(--color-text-muted)]" />
        <span className="typing-dot h-2 w-2 rounded-full bg-[var(--color-text-muted)]" />
      </div>
    </div>
  );
}

/** Small text action button used under assistant messages. */
function MessageActionButton({
  label,
  onClick,
  disabled,
}: {
  label: string;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="rounded-[var(--radius-sm)] px-2 py-1 text-xs text-[var(--color-text-muted)] transition-colors hover:bg-[var(--color-surface-elevated)] hover:text-[var(--color-text)] disabled:cursor-not-allowed disabled:opacity-50"
    >
      {label}
    </button>
  );
}

/** Copies assistant message text to the clipboard. */
function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard may be unavailable in some browsers.
    }
  };

  return (
    <button
      type="button"
      onClick={handleCopy}
      aria-label="Copy message"
      className="rounded-[var(--radius-sm)] px-2 py-1 text-xs text-[var(--color-text-muted)] transition-colors hover:bg-[var(--color-surface-elevated)] hover:text-[var(--color-text)]"
    >
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

function MessageContent({
  content,
  isUser,
  isStreaming,
}: {
  content: string;
  isUser: boolean;
  isStreaming?: boolean;
}) {
  if (isUser) {
    return <p className="m-0 whitespace-pre-wrap break-words">{content}</p>;
  }

  // While streaming, render plain text for smooth updates.
  // After streaming finishes, render markdown formatting.
  if (isStreaming) {
    return (
      <p className="m-0 whitespace-pre-wrap break-words">
        {content}
        <span className="stream-cursor" aria-hidden="true" />
      </p>
    );
  }

  return <MarkdownContent content={content} />;
}

/** Renders one chat bubble (user or assistant). */
function MessageBubble({
  message,
  isStreaming,
  showRegenerate,
  onRegenerate,
  actionsDisabled,
}: {
  message: ChatMessage;
  isStreaming?: boolean;
  showRegenerate?: boolean;
  onRegenerate?: () => void;
  actionsDisabled?: boolean;
}) {
  const isUser = message.role === "user";

  return (
    <div
      className={`flex items-end gap-3 ${isUser ? "justify-end" : "justify-start"}`}
    >
      {!isUser && <AssistantAvatar />}
      <div className="flex max-w-[85%] flex-col gap-1 sm:max-w-[75%]">
        <div
          className={`rounded-[var(--radius-lg)] px-4 py-3 text-[15px] leading-relaxed break-words ${
            isUser
              ? "bg-[var(--color-user-bubble)] text-white"
              : "bg-[var(--color-assistant-bubble)] text-[var(--color-text)] border border-[var(--color-border)]"
          }`}
        >
          <MessageContent
            content={message.content}
            isUser={isUser}
            isStreaming={isStreaming}
          />
        </div>
        {!isUser && message.content && !isStreaming && (
          <div className="flex justify-start gap-1 pl-1">
            <CopyButton text={message.content} />
            {showRegenerate && onRegenerate && (
              <MessageActionButton
                label="Regenerate"
                onClick={onRegenerate}
                disabled={actionsDisabled}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/** Welcome screen with quick-start suggestion chips. */
function EmptyState({
  mode,
  allowProviderSwitch,
  onSuggestion,
  disabled,
}: {
  mode: ChatMode;
  allowProviderSwitch: boolean;
  onSuggestion: (text: string) => void;
  disabled?: boolean;
}) {
  const isDocs = mode === "docs";

  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-4 px-6 text-center">
      <AssistantAvatar size={56} />
      <div>
        <h2 className="text-lg font-semibold text-[var(--color-text)]">
          {isDocs ? "Ask my docs" : "Start a conversation"}
        </h2>
        <p className="mt-1 text-sm text-[var(--color-text-muted)]">
          {isDocs
            ? "Questions are answered from files in backend/data. Upload .md or .txt to add more."
            : allowProviderSwitch
              ? "Choose an AI provider above and ask anything."
              : "Ask anything. Groq goes through FastAPI when configured; other providers stay on Next.js."}
        </p>
      </div>
      <div className="flex flex-wrap justify-center gap-2">
        {(isDocs
          ? [
              "What is the fridge password?",
              "Summarize the project notes",
              "Who set the fridge password?",
            ]
          : [
              "Explain quantum computing simply",
              "Write a haiku about coding",
              "What can you help me with?",
            ]
        ).map((suggestion) => (
          <SuggestionChip
            key={suggestion}
            text={suggestion}
            disabled={disabled}
            onClick={() => onSuggestion(suggestion)}
          />
        ))}
      </div>
    </div>
  );
}

function SuggestionChip({
  text,
  onClick,
  disabled,
}: {
  text: string;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="rounded-[var(--radius-full)] border border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-2 text-sm text-[var(--color-text-muted)] transition-colors hover:border-[var(--color-primary)] hover:text-[var(--color-text)] disabled:cursor-not-allowed disabled:opacity-50"
    >
      {text}
    </button>
  );
}

/** Download chat history as .md or .txt. */
function ExportMenu({
  messages,
  disabled,
}: {
  messages: ChatMessage[];
  disabled?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;

    const handleClickOutside = (event: MouseEvent) => {
      if (!menuRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [open]);

  const handleExport = (format: "md" | "txt") => {
    downloadChat(messages, format);
    setOpen(false);
  };

  return (
    <div className="relative" ref={menuRef}>
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        disabled={disabled}
        aria-expanded={open}
        aria-haspopup="menu"
        className="rounded-[var(--radius-sm)] px-3 py-1.5 text-sm text-[var(--color-text-muted)] transition-colors hover:bg-[var(--color-surface-elevated)] hover:text-[var(--color-text)] disabled:cursor-not-allowed disabled:opacity-50"
      >
        Export
      </button>
      {open && (
        <div
          role="menu"
          className="absolute right-0 top-full z-10 mt-1 min-w-[10rem] rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-surface-elevated)] py-1 shadow-lg"
        >
          <button
            type="button"
            role="menuitem"
            onClick={() => handleExport("md")}
            className="block w-full px-3 py-2 text-left text-sm text-[var(--color-text)] hover:bg-[var(--color-surface)]"
          >
            Markdown (.md)
          </button>
          <button
            type="button"
            role="menuitem"
            onClick={() => handleExport("txt")}
            className="block w-full px-3 py-2 text-left text-sm text-[var(--color-text)] hover:bg-[var(--color-surface)]"
          >
            Plain text (.txt)
          </button>
        </div>
      )}
    </div>
  );
}

/** Header dropdown to pick Groq / Gemini / Hugging Face (when enabled). */
function ProviderSelect({
  providers,
  value,
  onChange,
  disabled,
}: {
  providers: readonly AIProvider[];
  value: AIProvider;
  onChange: (provider: AIProvider) => void;
  disabled?: boolean;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value as AIProvider)}
      disabled={disabled}
      aria-label="AI provider"
      className="rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-surface-elevated)] px-3 py-1.5 text-sm text-[var(--color-text)] transition-colors hover:border-[var(--color-primary)] focus:border-[var(--color-primary)] disabled:cursor-not-allowed disabled:opacity-50"
    >
      {providers.map((provider) => (
        <option key={provider} value={provider}>
          {getProviderLabel(provider)}
        </option>
      ))}
    </select>
  );
}

/** Toggles shorter vs longer assistant replies (sent to the API). */
function ReplyModeToggle({
  concise,
  onChange,
  disabled,
}: {
  concise: boolean;
  onChange: (concise: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <div
      className="flex rounded-[var(--radius-full)] border border-[var(--color-border)] bg-[var(--color-surface-elevated)] p-1"
      role="group"
      aria-label="Reply mode"
    >
      <button
        type="button"
        disabled={disabled}
        onClick={() => onChange(false)}
        className={`rounded-[var(--radius-full)] px-3 py-1 text-xs transition-colors disabled:opacity-50 ${
          !concise
            ? "bg-[var(--color-primary)] text-white"
            : "text-[var(--color-text-muted)] hover:text-[var(--color-text)]"
        }`}
      >
        Detailed
      </button>
      <button
        type="button"
        disabled={disabled}
        onClick={() => onChange(true)}
        className={`rounded-[var(--radius-full)] px-3 py-1 text-xs transition-colors disabled:opacity-50 ${
          concise
            ? "bg-[var(--color-primary)] text-white"
            : "text-[var(--color-text-muted)] hover:text-[var(--color-text)]"
        }`}
      >
        Concise
      </button>
    </div>
  );
}

/** Upload a markdown/text doc and incrementally insert it into the RAG index. */
function DocUploadButton({
  disabled,
  onUploaded,
  onError,
}: {
  disabled?: boolean;
  onUploaded: (filename: string, filesSeen: number) => void;
  onError: (message: string) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);

  const handleChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file || uploading) return;

    const lower = file.name.toLowerCase();
    if (!lower.endsWith(".md") && !lower.endsWith(".txt")) {
      onError("Only .md and .txt files are allowed");
      return;
    }

    setUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch("/api/rag/upload", {
        method: "POST",
        body: formData,
      });
      const data = (await response.json().catch(() => ({}))) as {
        error?: string;
        filename?: string;
        files_seen?: number;
      };
      if (!response.ok) {
        throw new Error(data.error || "Upload failed");
      }
      onUploaded(data.filename ?? file.name, data.files_seen ?? 0);
    } catch (err) {
      onError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept=".md,.txt,text/markdown,text/plain"
        className="hidden"
        onChange={handleChange}
        disabled={disabled || uploading}
      />
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        disabled={disabled || uploading}
        className="rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-surface-elevated)] px-3 py-1.5 text-sm text-[var(--color-text-muted)] transition-colors hover:border-[var(--color-primary)] hover:text-[var(--color-text)] disabled:cursor-not-allowed disabled:opacity-50"
      >
        {uploading ? "Uploading..." : "Upload doc"}
      </button>
    </>
  );
}

/** Switch between general chat and Ask My Docs. */
function ModeToggle({
  mode,
  onChange,
  disabled,
}: {
  mode: ChatMode;
  onChange: (mode: ChatMode) => void;
  disabled?: boolean;
}) {
  return (
    <div
      className="flex rounded-[var(--radius-full)] border border-[var(--color-border)] bg-[var(--color-surface-elevated)] p-1"
      role="group"
      aria-label="Chat mode"
    >
      <button
        type="button"
        disabled={disabled}
        onClick={() => onChange("chat")}
        className={`rounded-[var(--radius-full)] px-3 py-1 text-xs transition-colors disabled:opacity-50 ${
          mode === "chat"
            ? "bg-[var(--color-primary)] text-white"
            : "text-[var(--color-text-muted)] hover:text-[var(--color-text)]"
        }`}
      >
        Chat
      </button>
      <button
        type="button"
        disabled={disabled}
        onClick={() => onChange("docs")}
        className={`rounded-[var(--radius-full)] px-3 py-1 text-xs transition-colors disabled:opacity-50 ${
          mode === "docs"
            ? "bg-[var(--color-primary)] text-white"
            : "text-[var(--color-text-muted)] hover:text-[var(--color-text)]"
        }`}
      >
        Ask My Docs
      </button>
    </div>
  );
}

export default function FastApiChat() {
  const allowProviderSwitch = false;
  const defaultProvider = "groq" as const;
  const providers = ["groq"] as const;
  const [mode, setMode] = useState<ChatMode>("docs");
  const [chatThread, setChatThread] = useState<ModeThread>({
    messages: [],
    sessionId: "chat_pending",
  });
  const [docsThread, setDocsThread] = useState<ModeThread>({
    messages: [],
    sessionId: "docs_pending",
  });
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeProvider, setActiveProvider] = useState<AIProvider | null>(null);
  const [selectedProvider, setSelectedProvider] =
    useState<AIProvider>("groq");
  const [conciseMode, setConciseMode] = useState(false);
  const [hydrated, setHydrated] = useState(false);
  const [canRetry, setCanRetry] = useState(false);
  const [uploadNotice, setUploadNotice] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const pendingStreamRef = useRef("");
  const flushStreamRef = useRef<number | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const retryMessagesRef = useRef<ChatMessage[] | null>(null);

  const activeThread = mode === "chat" ? chatThread : docsThread;
  const messages = activeThread.messages;
  const sessionId = activeThread.sessionId;

  const setActiveMessages = (
    updater: ChatMessage[] | ((prev: ChatMessage[]) => ChatMessage[])
  ) => {
    const apply = (prev: ModeThread): ModeThread => ({
      ...prev,
      messages:
        typeof updater === "function" ? updater(prev.messages) : updater,
    });
    if (mode === "chat") {
      setChatThread(apply);
    } else {
      setDocsThread(apply);
    }
  };

  // Load saved chat from localStorage after the component mounts in the browser.
  useEffect(() => {
    const stored = loadStoredChat();
    if (stored) {
      setMode(stored.mode);
      setChatThread(stored.chat);
      setDocsThread(stored.docs);
      setSelectedProvider(stored.selectedProvider ?? "groq");
      setConciseMode(stored.conciseMode ?? false);
      setActiveProvider(stored.activeProvider ?? null);
    } else {
      setChatThread({ messages: [], sessionId: createSessionId("chat") });
      setDocsThread({ messages: [], sessionId: createSessionId("docs") });
    }
    setHydrated(true);
  }, []);

  // Save chat to localStorage, but not on every streaming token.
  useEffect(() => {
    if (!hydrated || isStreaming) return;

    saveStoredChat({
      mode,
      chat: chatThread,
      docs: docsThread,
      selectedProvider,
      conciseMode,
      activeProvider,
    });
  }, [
    mode,
    chatThread,
    docsThread,
    selectedProvider,
    conciseMode,
    activeProvider,
    hydrated,
    isStreaming,
  ]);

  // Keep the latest message visible while chatting.
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: isStreaming ? "auto" : "smooth",
      block: "end",
    });
  }, [messages, isLoading, isStreaming]);

  const flushStreamBuffer = () => {
    const chunk = pendingStreamRef.current;
    if (!chunk) return;

    pendingStreamRef.current = "";
    setActiveMessages((prev) => {
      const next = [...prev];
      const last = next[next.length - 1];
      if (last?.role === "assistant") {
        next[next.length - 1] = {
          ...last,
          content: last.content + chunk,
        };
      }
      return next;
    });
  };

  const appendStreamChunk = (chunk: string) => {
    pendingStreamRef.current += chunk;

    if (flushStreamRef.current !== null) return;

    flushStreamRef.current = window.requestAnimationFrame(() => {
      flushStreamRef.current = null;
      flushStreamBuffer();
    });
  };

  const cancelStreamAnimation = () => {
    if (flushStreamRef.current !== null) {
      window.cancelAnimationFrame(flushStreamRef.current);
      flushStreamRef.current = null;
    }
  };

  const removeEmptyAssistantMessage = (prev: ChatMessage[]) => {
    const last = prev[prev.length - 1];
    if (last?.role === "assistant" && !last.content.trim()) {
      return prev.slice(0, -1);
    }
    return prev;
  };

  /** Core streaming request — used by send, retry, and regenerate. */
  const executeChatRequest = async (chatMessages: ChatMessage[]) => {
    if (isLoading) return;

    setError(null);
    setUploadNotice(null);
    setCanRetry(false);
    setIsLoading(true);
    setIsStreaming(false);
    retryMessagesRef.current = chatMessages;

    const controller = new AbortController();
    abortControllerRef.current = controller;

    const endpoint = mode === "docs" ? "/api/rag" : "/api/chat";
    const body =
      mode === "docs"
        ? {
            question: chatMessages[chatMessages.length - 1]?.content ?? "",
            session_id: sessionId,
            // Prior turns only — restores FastAPI RAG memory after reload
            history: chatMessages.slice(0, -1).map((m) => ({
              role: m.role,
              content: m.content,
            })),
          }
        : {
            messages: chatMessages,
            stream: true,
            concise: conciseMode,
            session_id: sessionId,
            ...(allowProviderSwitch ? { provider: selectedProvider } : {}),
          };

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const contentType = response.headers.get("content-type") ?? "";
        if (!contentType.includes("text/event-stream")) {
          const data = await response.json().catch(() => ({}));
          throw new Error(data.error || "Something went wrong");
        }
      }

      setActiveMessages((prev) => [
        ...prev,
        { role: "assistant", content: "" },
      ]);
      setIsStreaming(true);

      let streamError: string | null = null;

      for await (const event of readChatStream(response, controller.signal)) {
        if (controller.signal.aborted) break;

        if (event.type === "meta") {
          setActiveProvider(event.provider);
        }

        if (event.type === "chunk") {
          appendStreamChunk(event.content);
        }

        if (event.type === "done") {
          setActiveProvider(event.provider);
        }

        if (event.type === "error") {
          streamError = event.error;
        }
      }

      cancelStreamAnimation();

      if (controller.signal.aborted) {
        flushStreamBuffer();
        setActiveMessages(removeEmptyAssistantMessage);
        return;
      }

      flushStreamBuffer();

      if (streamError) {
        setActiveMessages((prev) => {
          const next = [...prev];
          const last = next[next.length - 1];
          if (last?.role === "assistant" && !last.content) {
            return next.slice(0, -1);
          }
          return next;
        });
        throw new Error(streamError);
      }

      setActiveMessages(removeEmptyAssistantMessage);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        return;
      }

      const message =
        err instanceof Error ? err.message : "Failed to get a response";
      setError(message);
      setCanRetry(true);
      setActiveMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last?.role === "assistant" && !last.content) {
          return prev.slice(0, -1);
        }
        return prev;
      });
    } finally {
      abortControllerRef.current = null;
      setIsLoading(false);
      setIsStreaming(false);
      inputRef.current?.focus();
    }
  };

  /** Send a user message to the backend and stream the assistant reply. */
  const sendMessage = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || isLoading) return;

    const userMessage: ChatMessage = { role: "user", content: trimmed };
    const updatedMessages = [...messages, userMessage];

    setActiveMessages(updatedMessages);
    setInput("");
    await executeChatRequest(updatedMessages);
  };

  const stopGeneration = () => {
    abortControllerRef.current?.abort();
    cancelStreamAnimation();
    flushStreamBuffer();
    setIsLoading(false);
    setIsStreaming(false);
    setError(null);
    setCanRetry(false);
  };

  const handleRetry = () => {
    if (!retryMessagesRef.current || isLoading) return;

    setActiveMessages((prev) => {
      const last = prev[prev.length - 1];
      if (last?.role === "assistant") {
        return prev.slice(0, -1);
      }
      return prev;
    });
    executeChatRequest(retryMessagesRef.current);
  };

  const handleRegenerate = () => {
    if (isLoading || messages.length === 0) return;

    let lastAssistantIndex = -1;
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "assistant") {
        lastAssistantIndex = i;
        break;
      }
    }
    if (lastAssistantIndex === -1) return;

    const chatMessages = messages.slice(0, lastAssistantIndex);
    const last = chatMessages[chatMessages.length - 1];
    if (!last || last.role !== "user") return;

    setActiveMessages(chatMessages);
    executeChatRequest(chatMessages);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage(input);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  const handleModeChange = (nextMode: ChatMode) => {
    if (nextMode === mode || isLoading) return;
    setError(null);
    setCanRetry(false);
    setMode(nextMode);
    inputRef.current?.focus();
  };

  const handleClear = async () => {
    if (isLoading) stopGeneration();

    const clearedSessionId = sessionId;
    const clearPath =
      mode === "docs"
        ? `/api/rag?session_id=${encodeURIComponent(clearedSessionId)}`
        : `/api/chat/session?session_id=${encodeURIComponent(clearedSessionId)}`;

    try {
      await fetch(clearPath, { method: "DELETE" });
    } catch {
      // Clearing server memory is best-effort; local state still resets.
    }

    const freshThread: ModeThread = {
      messages: [],
      sessionId: createSessionId(mode === "docs" ? "docs" : "chat"),
    };

    if (mode === "docs") {
      setDocsThread(freshThread);
    } else {
      setChatThread(freshThread);
    }

    setError(null);
    setCanRetry(false);
    setActiveProvider(null);
    retryMessagesRef.current = null;
    inputRef.current?.focus();
  };

  const lastAssistantIndex = (() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "assistant") return i;
    }
    return -1;
  })();

  if (!hydrated) {
    return (
      <div className="flex h-dvh items-center justify-center bg-[var(--color-bg)] text-sm text-[var(--color-text-muted)]">
        Loading chat...
      </div>
    );
  }

  const showTypingIndicator =
    isLoading &&
    !isStreaming &&
    messages[messages.length - 1]?.role !== "assistant";

  return (
    <div className="flex h-dvh flex-col bg-[var(--color-bg)]">
      <header className="flex shrink-0 flex-wrap items-center justify-between gap-3 border-b border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-3 sm:px-6">
        <div className="flex items-center gap-3">
          <AssistantAvatar size={36} />
          <div>
            <h1 className="text-base font-semibold text-[var(--color-text)]">
              {mode === "docs" ? "Ask My Docs" : "AI Chat"}
            </h1>
            {mode === "chat" && activeProvider && (
              <p className="text-xs text-[var(--color-success)]">
                via {getProviderLabel(activeProvider)}
              </p>
            )}
            {mode === "docs" && (
              <p className="text-xs text-[var(--color-text-muted)]">
                FastAPI RAG · backend/data
              </p>
            )}
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2 sm:gap-3">
          <ModeToggle
            mode={mode}
            onChange={handleModeChange}
            disabled={isLoading}
          />
          {mode === "docs" && (
            <DocUploadButton
              disabled={isLoading}
              onUploaded={(filename, filesSeen) => {
                setUploadNotice(
                  `Uploaded ${filename} — ${filesSeen} file(s) indexed. You can ask about it now.`
                );
                setError(null);
              }}
              onError={(message) => {
                setUploadNotice(null);
                setError(message);
              }}
            />
          )}
          {mode === "chat" && (
            <ReplyModeToggle
              concise={conciseMode}
              onChange={setConciseMode}
              disabled={isLoading}
            />
          )}
          {mode === "chat" && allowProviderSwitch && (
            <ProviderSelect
              providers={providers}
              value={selectedProvider}
              onChange={setSelectedProvider}
              disabled={isLoading}
            />
          )}
          {messages.length > 0 && (
            <>
              <ExportMenu messages={messages} disabled={isLoading} />
              <button
                type="button"
                onClick={handleClear}
                className="rounded-[var(--radius-sm)] px-3 py-1.5 text-sm text-[var(--color-text-muted)] transition-colors hover:bg-[var(--color-surface-elevated)] hover:text-[var(--color-text)]"
              >
                Clear {mode === "docs" ? "docs" : "chat"}
              </button>
            </>
          )}
        </div>
      </header>

      <main className="flex flex-1 flex-col overflow-hidden">
        {messages.length === 0 && !isLoading ? (
          <EmptyState
            mode={mode}
            allowProviderSwitch={allowProviderSwitch}
            onSuggestion={sendMessage}
            disabled={isLoading}
          />
        ) : (
          <div className="flex-1 overflow-y-auto px-4 py-6 sm:px-6">
            <div className="mx-auto flex max-w-3xl flex-col gap-4">
              {messages.map((msg, i) => (
                <MessageBubble
                  key={i}
                  message={msg}
                  isStreaming={
                    isStreaming &&
                    i === messages.length - 1 &&
                    msg.role === "assistant"
                  }
                  showRegenerate={
                    i === lastAssistantIndex && msg.role === "assistant"
                  }
                  onRegenerate={handleRegenerate}
                  actionsDisabled={isLoading}
                />
              ))}
              {showTypingIndicator && <TypingIndicator />}
              <div ref={messagesEndRef} />
            </div>
          </div>
        )}

        {uploadNotice && mode === "docs" && (
          <div
            role="status"
            className="mx-4 mb-2 rounded-[var(--radius-md)] border border-[var(--color-success)]/30 bg-[var(--color-success)]/10 px-4 py-3 text-sm text-[var(--color-success)] sm:mx-6"
          >
            {uploadNotice}
          </div>
        )}

        {error && (
          <div
            role="alert"
            className="mx-4 mb-2 flex flex-wrap items-center justify-between gap-3 rounded-[var(--radius-md)] border border-[var(--color-error)]/30 bg-[var(--color-error)]/10 px-4 py-3 text-sm text-[var(--color-error)] sm:mx-6"
          >
            <span>{error}</span>
            {canRetry && (
              <button
                type="button"
                onClick={handleRetry}
                disabled={isLoading}
                className="shrink-0 rounded-[var(--radius-sm)] border border-[var(--color-error)]/40 px-3 py-1 text-xs font-medium text-[var(--color-error)] transition-colors hover:bg-[var(--color-error)]/10 disabled:cursor-not-allowed disabled:opacity-50"
              >
                Try again
              </button>
            )}
          </div>
        )}

        <form
          onSubmit={handleSubmit}
          className="shrink-0 border-t border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-4 sm:px-6"
        >
          <div className="mx-auto flex max-w-3xl items-end gap-3">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                mode === "docs"
                  ? "Ask a question about your docs..."
                  : "Type a message..."
              }
              rows={1}
              disabled={isLoading}
              aria-label={mode === "docs" ? "Docs question input" : "Message input"}
              className="max-h-32 min-h-[44px] flex-1 resize-none rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface-elevated)] px-4 py-3 text-[15px] text-[var(--color-text)] placeholder:text-[var(--color-text-muted)] transition-colors focus:border-[var(--color-primary)] disabled:opacity-50"
            />
            {isLoading ? (
              <button
                type="button"
                onClick={stopGeneration}
                aria-label="Stop generation"
                className="flex h-11 w-11 shrink-0 items-center justify-center rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface-elevated)] text-[var(--color-text)] transition-colors hover:bg-[var(--color-surface)]"
              >
                <svg
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="currentColor"
                  aria-hidden="true"
                >
                  <rect x="6" y="6" width="12" height="12" rx="1" />
                </svg>
              </button>
            ) : (
              <button
                type="submit"
                disabled={!input.trim()}
                aria-label="Send message"
                className="flex h-11 w-11 shrink-0 items-center justify-center rounded-[var(--radius-md)] bg-[var(--color-primary)] text-white transition-colors hover:bg-[var(--color-primary-hover)] disabled:cursor-not-allowed disabled:opacity-40"
              >
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  aria-hidden="true"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                  />
                </svg>
              </button>
            )}
          </div>
          <p className="mx-auto mt-2 max-w-3xl text-center text-xs text-[var(--color-text-muted)]">
            {isLoading
              ? "Click stop to cancel generation"
              : "Press Enter to send, Shift+Enter for new line"}
          </p>
        </form>
      </main>
    </div>
  );
}
