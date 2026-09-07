import { MessageSquare } from "lucide-react";

import { EmptyPanel, ErrorPanel, LoadingPanel } from "@/components/ui/StatePanel";
import type { Message } from "@/types/api";
import { cn } from "@/lib/cn";

export function MessageTimeline({
  messages,
  isLoading,
  isError,
  onRetry,
  selectedMessageId,
  onSelectMessage,
}: {
  messages: Message[] | undefined;
  isLoading: boolean;
  isError: boolean;
  onRetry: () => void;
  /** Which message the intelligence panels (emotion/NLP/incongruence) are
   * currently analyzing - highlighted so the selection is visible in the
   * timeline, not just implied by a sibling panel's state. */
  selectedMessageId?: string | null;
  onSelectMessage?: (messageId: string) => void;
}) {
  if (isLoading) {
    return <LoadingPanel label="Loading messages…" />;
  }

  if (isError) {
    return (
      <ErrorPanel description="We couldn't load this conversation's messages." onRetry={onRetry} />
    );
  }

  if (!messages || messages.length === 0) {
    return (
      <EmptyPanel
        icon={<MessageSquare className="text-[var(--color-text-tertiary)]" size={24} />}
        title="No messages yet"
        description="Speak or type below to begin this conversation."
      />
    );
  }

  return (
    <ol className="flex flex-1 flex-col gap-3 overflow-y-auto p-4">
      {messages.map((message) => (
        <li
          key={message.id}
          className={cn(
            "max-w-[75%] rounded-2xl px-4 py-2.5 text-sm",
            message.role === "user"
              ? "self-end bg-[var(--color-accent)] text-[#0b0b10]"
              : "self-start bg-[var(--color-surface-raised)] text-[var(--color-text-primary)]",
            onSelectMessage && "cursor-pointer",
            selectedMessageId === message.id && "ring-2 ring-[var(--color-accent-strong)]",
          )}
          onClick={() => onSelectMessage?.(message.id)}
        >
          {message.content}
        </li>
      ))}
    </ol>
  );
}
