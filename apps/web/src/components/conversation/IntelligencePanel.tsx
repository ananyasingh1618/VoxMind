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
    <div className="flex h-full flex-col divide-y divide-[var(--color-border)] overflow-y-auto">
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
