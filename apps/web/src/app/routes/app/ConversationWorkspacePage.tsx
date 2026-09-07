import { AudioLines, BookOpen, MessageSquare, Sparkles } from "lucide-react";
import { useState } from "react";
import { Link, useParams } from "react-router-dom";

import { AudioPanel } from "@/components/conversation/AudioPanel";
import { IntelligencePanel } from "@/components/conversation/IntelligencePanel";
import { KnowledgePanel } from "@/components/conversation/KnowledgePanel";
import { MessageComposer } from "@/components/conversation/MessageComposer";
import { MessageTimeline } from "@/components/conversation/MessageTimeline";
import { Card } from "@/components/ui/Card";
import { ErrorPanel, LoadingPanel } from "@/components/ui/StatePanel";
import { VoiceSessionControl } from "@/components/voice/VoiceSessionControl";
import {
  useConversation,
  useMessages,
  useSendMessage,
} from "@/features/conversations/hooks";
import { cn } from "@/lib/cn";

type MobileTab = "chat" | "audio" | "intelligence" | "knowledge";

const MOBILE_TABS: { key: MobileTab; label: string; icon: typeof MessageSquare }[] = [
  { key: "chat", label: "Chat", icon: MessageSquare },
  { key: "audio", label: "Audio", icon: AudioLines },
  { key: "intelligence", label: "Intelligence", icon: Sparkles },
  { key: "knowledge", label: "Knowledge", icon: BookOpen },
];

export function ConversationWorkspacePage() {
  const { id } = useParams<{ id: string }>();
  const {
    data: conversation,
    isLoading: isConversationLoading,
    isError: isConversationError,
    refetch: refetchConversation,
  } = useConversation(id);
  const {
    data: messages,
    isLoading: isMessagesLoading,
    isError: isMessagesError,
    refetch: refetchMessages,
  } = useMessages(id);
  const sendMessage = useSendMessage(id ?? "");
  const [selectedMessageId, setSelectedMessageId] = useState<string | null>(null);
  const [mobileTab, setMobileTab] = useState<MobileTab>("chat");

  if (isConversationLoading) {
    return <LoadingPanel label="Loading conversation…" />;
  }

  if (isConversationError || !conversation) {
    return (
      <ErrorPanel
        title="Conversation not found"
        description="This conversation may have been deleted, or doesn't belong to your account."
        onRetry={() => refetchConversation()}
      />
    );
  }

  return (
    <div className="flex h-full flex-col lg:grid lg:grid-cols-[minmax(0,1fr)_22rem]">
      <nav
        aria-label="Workspace panels"
        className="flex shrink-0 items-center gap-1 overflow-x-auto border-b border-[var(--color-border)] px-2 py-1.5 lg:hidden"
      >
        {MOBILE_TABS.map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            type="button"
            aria-pressed={mobileTab === key}
            onClick={() => setMobileTab(key)}
            className={cn(
              "flex shrink-0 items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium transition-colors",
              mobileTab === key
                ? "bg-[var(--color-accent-muted)] text-[var(--color-accent-strong)]"
                : "text-[var(--color-text-tertiary)] hover:text-[var(--color-text-secondary)]",
            )}
          >
            <Icon size={13} /> {label}
          </button>
        ))}
      </nav>

      <div
        className={cn(
          "min-h-0 min-w-0 flex-col border-r border-[var(--color-border)]",
          mobileTab === "chat" ? "flex" : "hidden",
          "lg:flex",
        )}
      >
        <header className="flex items-center justify-between gap-2 border-b border-[var(--color-border)] px-4 py-3">
          <div className="min-w-0">
            <h1 className="truncate text-sm font-semibold">{conversation.title ?? "Untitled session"}</h1>
            <p className="text-xs text-[var(--color-text-tertiary)]">
              Session status:{" "}
              <span
                className={
                  conversation.status === "active"
                    ? "text-[var(--color-signal-positive)]"
                    : "text-[var(--color-text-tertiary)]"
                }
              >
                {conversation.status}
              </span>
            </p>
          </div>
          <Link
            to={`/app/conversations/${id}/insights`}
            className="flex items-center gap-1.5 rounded-full border border-[var(--color-border-strong)] px-3 py-1.5 text-xs text-[var(--color-text-secondary)] transition-colors hover:border-[var(--color-accent)] hover:text-[var(--color-text-primary)]"
          >
            <Sparkles size={12} /> Insights
          </Link>
        </header>

        <VoiceSessionControl conversationId={id ?? ""} />

        <MessageTimeline
          messages={messages}
          isLoading={isMessagesLoading}
          isError={isMessagesError}
          onRetry={() => refetchMessages()}
          selectedMessageId={selectedMessageId}
          onSelectMessage={setSelectedMessageId}
        />

        <MessageComposer
          isSending={sendMessage.isPending}
          onSend={(content) => sendMessage.mutateAsync(content).then(() => undefined)}
        />
      </div>

      <div
        className={cn(
          "min-h-0 flex-col divide-y divide-[var(--color-border)]",
          mobileTab === "chat" ? "hidden" : "flex",
          "lg:flex",
        )}
      >
        <Card
          className={cn(
            "min-h-0 rounded-none border-0 border-b border-[var(--color-border)]",
            mobileTab === "audio" ? "flex h-full flex-col" : "hidden",
            "lg:flex lg:h-1/2",
          )}
        >
          <AudioPanel conversationId={id ?? ""} onSelectMessage={setSelectedMessageId} />
        </Card>
        <Card
          className={cn(
            "min-h-0 rounded-none border-0 border-b border-[var(--color-border)]",
            mobileTab === "intelligence" ? "flex h-full flex-col" : "hidden",
            "lg:flex lg:h-1/4",
          )}
        >
          <IntelligencePanel conversationId={id ?? ""} selectedMessageId={selectedMessageId} />
        </Card>
        <Card
          className={cn(
            "min-h-0 rounded-none border-0",
            mobileTab === "knowledge" ? "flex h-full flex-col" : "hidden",
            "lg:flex lg:h-1/4",
          )}
        >
          <KnowledgePanel conversationId={id ?? ""} />
        </Card>
      </div>
    </div>
  );
}
