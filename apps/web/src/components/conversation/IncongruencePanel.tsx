import { Activity } from "lucide-react";

import { ApiError } from "@/api/client";
import { PipelineNotAvailable } from "@/components/conversation/PipelineNotAvailable";
import { Button } from "@/components/ui/Button";
import { ScoreBar } from "@/components/ui/ScoreBar";
import { Spinner } from "@/components/ui/Spinner";
import { useAnalyzeIncongruence, useIncongruenceSignals } from "@/features/nlp/hooks";

export function IncongruencePanel({
  conversationId,
  messageId,
}: {
  conversationId: string;
  messageId: string | null;
}) {
  const signalsQuery = useIncongruenceSignals(conversationId, messageId);
  const analyze = useAnalyzeIncongruence(conversationId);

  if (!messageId) {
    return (
      <PipelineNotAvailable
        icon={<Activity className="text-[var(--color-text-tertiary)]" size={18} />}
        title="Tone & word alignment"
        phaseLabel="Select a message"
        description="An analytical signal comparing what's said to how it's said - useful for spotting sarcasm, stress, or hesitation. It is not a lie detector and never makes a truthfulness claim."
      />
    );
  }

  const signals = signalsQuery.data ?? [];

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="flex items-center justify-between gap-2 border-b border-[var(--color-border)] p-2">
        <span className="flex items-center gap-1.5 text-xs font-medium text-[var(--color-text-secondary)]">
          <Activity size={13} /> Tone & word alignment
        </span>
        <Button
          variant="ghost"
          className="px-2 py-1 text-xs"
          isLoading={analyze.isPending}
          onClick={() => analyze.mutateAsync(messageId)}
        >
          {signals.length > 0 ? "Re-analyze" : "Analyze"}
        </Button>
      </div>

      {analyze.isError && (
        <p role="alert" className="p-2 text-xs text-[var(--color-signal-negative)]">
          {analyze.error instanceof ApiError ? analyze.error.message : "Analysis failed."}
        </p>
      )}

      <div className="min-h-0 flex-1 overflow-y-auto p-2 text-xs">
        {signalsQuery.isLoading && (
          <div className="flex justify-center py-3">
            <Spinner />
          </div>
        )}
        {!signalsQuery.isLoading && signals.length === 0 && (
          <p className="text-[var(--color-text-tertiary)]">
            Not yet analyzed. Requires both a transcribed turn and a real emotion prediction for it.
          </p>
        )}
        <ol className="flex flex-col gap-2">
          {signals.map((signal) => (
            <li key={signal.id} className="flex flex-col gap-1.5 rounded-lg border border-[var(--color-border)] p-2">
              <p className="font-medium text-[var(--color-text-primary)]">
                Incongruence: {Math.round(signal.incongruence_score * 100)}%{" "}
                <span className="font-normal text-[var(--color-text-tertiary)]">
                  (confidence {Math.round(signal.confidence * 100)}%)
                </span>
              </p>
              <ScoreBar
                label="Divergence"
                value={signal.incongruence_score}
                color="var(--color-accent)"
                hint="A descriptive comparison signal, not a certainty or diagnostic score."
              />
              <p className="text-[var(--color-text-secondary)]">{signal.explanation}</p>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
