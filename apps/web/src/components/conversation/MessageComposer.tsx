import { Send } from "lucide-react";
import { type FormEvent, useState } from "react";

import { Button } from "@/components/ui/Button";

export function MessageComposer({
  onSend,
  isSending,
}: {
  onSend: (content: string) => Promise<void>;
  isSending: boolean;
}) {
  const [content, setContent] = useState("");

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const trimmed = content.trim();
    if (!trimmed) return;
    await onSend(trimmed);
    setContent("");
  }

  return (
    <form onSubmit={handleSubmit} className="flex items-center gap-2 border-t border-[var(--color-border)] p-3">
      <label htmlFor="message-composer" className="sr-only">
        Type a message
      </label>
      <input
        id="message-composer"
        value={content}
        onChange={(e) => setContent(e.target.value)}
        placeholder="Type a message…"
        className="flex-1 rounded-lg border border-[var(--color-border-strong)] bg-[var(--color-surface)] px-3 py-2 text-sm outline-none focus:border-[var(--color-accent)]"
      />
      <Button type="submit" isLoading={isSending} disabled={!content.trim()}>
        <Send size={15} />
        <span className="sr-only">Send</span>
      </Button>
    </form>
  );
}
