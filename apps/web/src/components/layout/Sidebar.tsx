import { BarChart3, FlaskConical, MessageSquare, Plus, Settings } from "lucide-react";
import { Link, NavLink, useNavigate } from "react-router-dom";

import { Button } from "@/components/ui/Button";
import { UserMenu } from "@/components/layout/UserMenu";
import { useConversations, useCreateConversation } from "@/features/conversations/hooks";
import { cn } from "@/lib/cn";

export function Sidebar() {
  const { data: conversations } = useConversations();
  const createConversation = useCreateConversation();
  const navigate = useNavigate();

  // Real bug found via a real browser walkthrough: this used to call
  // `.mutate()` (fire-and-forget) with no follow-up navigation, unlike the
  // identical action on ConversationsListPage.tsx (which correctly awaits
  // `.mutateAsync()` then navigates to the new conversation). The real
  // conversation was always genuinely created on the backend - the sidebar
  // button just silently left the user wherever they already were,
  // requiring them to notice and click the new entry themselves.
  async function handleCreate() {
    const conversation = await createConversation.mutateAsync(undefined);
    navigate(`/app/conversations/${conversation.id}`);
  }

  return (
    <aside className="flex h-full w-72 flex-col border-r border-[var(--color-border)] bg-[var(--color-surface)]">
      <div className="flex items-center gap-2 px-4 py-5">
        <span className="h-2.5 w-2.5 rounded-full bg-[var(--color-accent)]" />
        <Link to="/app/conversations" className="text-sm font-semibold tracking-wide">
          VoxMind
        </Link>
      </div>

      <div className="px-3">
        <Button
          variant="secondary"
          className="w-full justify-start"
          onClick={handleCreate}
          isLoading={createConversation.isPending}
        >
          <Plus size={15} /> New conversation
        </Button>
      </div>

      <nav className="mt-4 flex-1 overflow-y-auto px-2" aria-label="Conversations">
        {(conversations ?? []).map((conversation) => (
          <NavLink
            key={conversation.id}
            to={`/app/conversations/${conversation.id}`}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-2 rounded-lg px-3 py-2 text-sm text-[var(--color-text-secondary)]",
                "hover:bg-[var(--color-surface-raised)] hover:text-[var(--color-text-primary)]",
                isActive && "bg-[var(--color-surface-raised)] text-[var(--color-text-primary)]",
              )
            }
          >
            <MessageSquare size={14} className="shrink-0" />
            <span className="truncate">{conversation.title ?? "Untitled session"}</span>
          </NavLink>
        ))}
      </nav>

      <div className="border-t border-[var(--color-border)] p-2">
        <NavLink
          to="/app/analytics"
          className={({ isActive }) =>
            cn(
              "mb-1 flex items-center gap-2 rounded-lg px-3 py-2 text-sm text-[var(--color-text-secondary)] hover:bg-[var(--color-surface-raised)]",
              isActive && "bg-[var(--color-surface-raised)] text-[var(--color-text-primary)]",
            )
          }
        >
          <BarChart3 size={14} /> Analytics
        </NavLink>
        <NavLink
          to="/app/evaluation"
          className={({ isActive }) =>
            cn(
              "mb-1 flex items-center gap-2 rounded-lg px-3 py-2 text-sm text-[var(--color-text-secondary)] hover:bg-[var(--color-surface-raised)]",
              isActive && "bg-[var(--color-surface-raised)] text-[var(--color-text-primary)]",
            )
          }
        >
          <FlaskConical size={14} /> Evaluation
        </NavLink>
        <NavLink
          to="/app/settings"
          className={({ isActive }) =>
            cn(
              "mb-1 flex items-center gap-2 rounded-lg px-3 py-2 text-sm text-[var(--color-text-secondary)] hover:bg-[var(--color-surface-raised)]",
              isActive && "bg-[var(--color-surface-raised)] text-[var(--color-text-primary)]",
            )
          }
        >
          <Settings size={14} /> Settings
        </NavLink>
        <UserMenu />
      </div>
    </aside>
  );
}
