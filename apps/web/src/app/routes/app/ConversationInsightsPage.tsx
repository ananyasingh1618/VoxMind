import { ArrowLeft, Lightbulb, Sparkles } from "lucide-react";
import { Link, useParams } from "react-router-dom";

import { Card } from "@/components/ui/Card";
import { EmptyPanel, ErrorPanel, LoadingPanel } from "@/components/ui/StatePanel";
import { useConversationInsights } from "@/features/insights/hooks";
import { cn } from "@/lib/cn";

export function ConversationInsightsPage() {
  const { id } = useParams<{ id: string }>();
  const { data, isLoading, isError, refetch } = useConversationInsights(id);

  if (isLoading) return <LoadingPanel label="Loading insights…" />;
  if (isError || !data) {
    return <ErrorPanel description="Couldn't load conversation insights." onRetry={() => refetch()} />;
  }

  return (
    <div className="flex h-full flex-col overflow-y-auto p-6">
      <div className="mb-4 flex items-center gap-3">
        <Link
          to={`/app/conversations/${id}`}
          className="flex items-center gap-1 text-xs text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)]"
        >
          <ArrowLeft size={13} /> Back to conversation
        </Link>
      </div>
      <h1 className="mb-1 text-lg font-semibold text-[var(--color-text-primary)]">Conversation insights</h1>
      <p className="mb-6 text-sm text-[var(--color-text-tertiary)]">
        What VoxMind detected in this conversation, and how it was computed.
      </p>

      {data.message_count === 0 ? (
        <EmptyPanel
          icon={<Sparkles className="text-[var(--color-text-tertiary)]" size={28} />}
          title="Nothing to show yet"
          description="Send a message in this conversation to see real sentiment, emotion, and incongruence signals here."
        />
      ) : (
        <div className="flex flex-col gap-6">
          {data.observations.length > 0 && (
            <Card className="p-4">
              <h2 className="mb-2 flex items-center gap-1.5 text-sm font-medium text-[var(--color-text-primary)]">
                <Sparkles size={14} /> Observations
              </h2>
              <ul className="flex flex-col gap-2">
                {data.observations.map((o, i) => (
                  <li key={i} className="text-sm text-[var(--color-text-secondary)]">
                    {o.text}
                    <span className="ml-1 text-xs text-[var(--color-text-tertiary)]">— {o.basis}</span>
                  </li>
                ))}
              </ul>
            </Card>
          )}

          {data.interpretations.length > 0 && (
            <Card className="p-4">
              <h2 className="mb-2 flex items-center gap-1.5 text-sm font-medium text-[var(--color-text-primary)]">
                <Lightbulb size={14} /> Interpretations
              </h2>
              <ul className="flex flex-col gap-3">
                {data.interpretations.map((interpretation, i) => (
                  <li key={i} className="text-sm text-[var(--color-text-secondary)]">
                    <p>{interpretation.text}</p>
                    <p className="mt-0.5 text-xs italic text-[var(--color-text-tertiary)]">
                      {interpretation.caveat}
                    </p>
                  </li>
                ))}
              </ul>
            </Card>
          )}

          <Card className="p-4">
            <h2 className="mb-3 text-sm font-medium text-[var(--color-text-primary)]">Intelligence timeline</h2>
            <ol className="flex flex-col gap-3">
              {data.timeline.map((entry) => (
                <li
                  key={entry.message_id}
                  className={cn(
                    "rounded-lg border border-[var(--color-border)] p-3 text-sm",
                    entry.role === "assistant" && "bg-[var(--color-surface-raised)]",
                  )}
                >
                  <div className="mb-1 flex items-center justify-between text-xs text-[var(--color-text-tertiary)]">
                    <span className="font-medium uppercase">{entry.role}</span>
                    <span>{new Date(entry.created_at).toLocaleString()}</span>
                  </div>
                  <p className="text-[var(--color-text-primary)]">{entry.content}</p>
                  <div className="mt-2 flex flex-wrap gap-2 text-xs text-[var(--color-text-tertiary)]">
                    {entry.sentiment_label && (
                      <span className="rounded-full border border-[var(--color-border)] px-2 py-0.5">
                        sentiment: {entry.sentiment_label} ({Math.round((entry.sentiment_score ?? 0) * 100)}%)
                      </span>
                    )}
                    {entry.intent_label && (
                      <span className="rounded-full border border-[var(--color-border)] px-2 py-0.5">
                        intent: {entry.intent_label}
                      </span>
                    )}
                    {entry.topics.length > 0 && (
                      <span className="rounded-full border border-[var(--color-border)] px-2 py-0.5">
                        topics: {entry.topics.join(", ")}
                      </span>
                    )}
                    {entry.emotion.map((e, i) => (
                      <span key={i} className="rounded-full border border-[var(--color-border)] px-2 py-0.5">
                        emotion: {e.predicted_label} ({Math.round(e.confidence * 100)}%)
                      </span>
                    ))}
                  </div>
                  {entry.incongruence.length > 0 && (
                    <div className="mt-2 flex flex-col gap-1">
                      {entry.incongruence.map((signal, i) => (
                        <p key={i} className="text-xs text-[var(--color-text-tertiary)]">
                          Incongruence: {Math.round(signal.incongruence_score * 100)}% — {signal.explanation}
                        </p>
                      ))}
                    </div>
                  )}
                </li>
              ))}
            </ol>
          </Card>
        </div>
      )}
    </div>
  );
}
