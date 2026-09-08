import { MessageSquare, Plus } from "lucide-react";
import { useNavigate } from "react-router-dom";

import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { EmptyPanel, ErrorPanel, LoadingPanel } from "@/components/ui/StatePanel";
import { useConversations, useCreateConversation } from "@/features/conversations/hooks";

export function ConversationsListPage() {
  const { data: conversations, isLoading, isError, refetch } = useConversations();
  const createConversation = useCreateConversation();
  const navigate = useNavigate();

  async function handleCreate() {
    const conversation = await createConversation.mutateAsync(undefined);
    navigate(`/app/conversations/${conversation.id}`);
  }

  if (isLoading) {
    return <LoadingPanel label="Loading your conversations…" />;
  }

  if (isError) {
    return (
      <ErrorPanel
        description="We couldn't load your conversations. Check your connection and try again."
        onRetry={() => refetch()}
      />
    );
  }

  if (!conversations || conversations.length === 0) {
    return (
      <EmptyPanel
        icon={<MessageSquare className="text-[var(--color-text-tertiary)]" size={28} />}
        title="No conversations yet"
        description="Start a new session to begin speaking with VoxMind."
        // axe-core `page-has-heading-one`: this is the entire content of
        // the page for a first-time user (no conversations yet) - the
        // most common state a brand-new user actually sees - and needs a
        // real <h1>, not the component's usual <h3> default.
        headingLevel="h1"
        action={
          <Button onClick={handleCreate} isLoading={createConversation.isPending}>
            <Plus size={15} /> New conversation
          </Button>
        }
      />
    );
  }

  return (
    <div className="mx-auto max-w-2xl p-6">
      <div className="mb-4 flex items-center justify-between">
        <h1 className="text-lg font-semibold">Conversations</h1>
        <Button onClick={handleCreate} isLoading={createConversation.isPending} variant="secondary">
          <Plus size={15} /> New
        </Button>
      </div>
      <div className="flex flex-col gap-2">
        {conversations.map((conversation) => (
          <Card
            key={conversation.id}
            className="cursor-pointer p-4 hover:border-[var(--color-accent)]"
            onClick={() => navigate(`/app/conversations/${conversation.id}`)}
          >
            <p className="font-medium">{conversation.title ?? "Untitled session"}</p>
            <p className="mt-1 text-xs text-[var(--color-text-tertiary)]">
              {new Date(conversation.created_at).toLocaleString()} · {conversation.status}
            </p>
          </Card>
        ))}
      </div>
    </div>
  );
}
