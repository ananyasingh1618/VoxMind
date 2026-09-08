import { Brain } from "lucide-react";

import { ApiError } from "@/api/client";
import { PipelineNotAvailable } from "@/components/conversation/PipelineNotAvailable";
import { Button } from "@/components/ui/Button";
import { ScoreBar } from "@/components/ui/ScoreBar";
import { Spinner } from "@/components/ui/Spinner";

const SENTIMENT_COLOR: Record<string, string> = {
  positive: "var(--color-signal-positive)",
  negative: "var(--color-signal-negative)",
  neutral: "var(--color-text-tertiary)",
};
import { useNlpAnnotation, useProcessNlp } from "@/features/nlp/hooks";

export function NlpPanel({
  conversationId,
  messageId,
}: {
  conversationId: string;
  messageId: string | null;
}) {
  const annotationQuery = useNlpAnnotation(conversationId, messageId);
  const processNlp = useProcessNlp(conversationId);

  if (!messageId) {
    return (
      <PipelineNotAvailable
        icon={<Brain className="text-[var(--color-text-tertiary)]" size={18} />}
        title="Language understanding"
        phaseLabel="Select a message"
        description="Select a message to analyze real sentiment, intent, topics, and entities."
      />
    );
  }

  const annotation = annotationQuery.data;

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="flex items-center justify-between gap-2 border-b border-[var(--color-border)] p-2">
        <span className="flex items-center gap-1.5 text-xs font-medium text-[var(--color-text-secondary)]">
          <Brain size={13} /> Language understanding
        </span>
        <Button
          variant="ghost"
          className="px-2 py-1 text-xs"
          isLoading={processNlp.isPending}
          onClick={() => processNlp.mutateAsync(messageId)}
        >
          {annotation ? "Re-analyze" : "Analyze"}
        </Button>
      </div>

      {processNlp.isError && (
        <p role="alert" className="p-2 text-xs text-[var(--color-signal-negative)]">
          {processNlp.error instanceof ApiError ? processNlp.error.message : "Analysis failed."}
        </p>
      )}

      {/* axe-core `scrollable-region-focusable`: WCAG 2.1.1 keyboard access. */}
      <div className="min-h-0 flex-1 overflow-y-auto p-2 text-xs" tabIndex={0}>
        {annotationQuery.isLoading && (
          <div className="flex justify-center py-3">
            <Spinner />
          </div>
        )}
        {!annotationQuery.isLoading && !annotation && (
          <p className="text-[var(--color-text-tertiary)]">Not yet analyzed.</p>
        )}
        {annotation && (
          <div className="flex flex-col gap-2">
            <div className="flex flex-col gap-1.5">
              <p>
                Sentiment:{" "}
                <span className="font-medium text-[var(--color-text-primary)]">
                  {annotation.sentiment_label}
                </span>{" "}
                <span className="text-[var(--color-text-tertiary)]">
                  ({Math.round(annotation.sentiment_score * 100)}%)
                </span>
              </p>
              <ScoreBar
                label="Sentiment confidence"
                value={annotation.sentiment_score}
                color={SENTIMENT_COLOR[annotation.sentiment_label] ?? "var(--color-accent)"}
              />
            </div>
            <p>
              Intent:{" "}
              <span className="font-medium text-[var(--color-text-primary)]">{annotation.intent_label}</span>
            </p>
            {annotation.topics.length > 0 && (
              <p>
                Topics:{" "}
                <span className="text-[var(--color-text-secondary)]">{annotation.topics.join(", ")}</span>
              </p>
            )}
            {annotation.entities.length > 0 && (
              <p>
                Entities:{" "}
                <span className="text-[var(--color-text-secondary)]">
                  {annotation.entities.map((e) => `${e.text} (${e.label})`).join(", ")}
                </span>
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
