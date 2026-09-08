import { EmotionPanel } from "@/components/conversation/EmotionPanel";
import { IncongruencePanel } from "@/components/conversation/IncongruencePanel";
import { NlpPanel } from "@/components/conversation/NlpPanel";

export function IntelligencePanel({
  conversationId,
  selectedMessageId,
}: {
  conversationId: string;
  selectedMessageId: string | null;
}) {
  return (
    // axe-core `scrollable-region-focusable`: a scrollable container with
    // no way to receive keyboard focus is unreachable by keyboard-only
    // scrolling (WCAG 2.1.1).
    <div
      className="flex h-full flex-col divide-y divide-[var(--color-border)] overflow-y-auto"
      tabIndex={0}
    >
      <div className="min-h-40 flex-1">
        <EmotionPanel conversationId={conversationId} messageId={selectedMessageId} />
      </div>
      <div className="min-h-40 flex-1">
        <NlpPanel conversationId={conversationId} messageId={selectedMessageId} />
      </div>
      <div className="min-h-40 flex-1">
        <IncongruencePanel conversationId={conversationId} messageId={selectedMessageId} />
      </div>
    </div>
  );
}
