import { Waves } from "lucide-react";
import { useState } from "react";

import { ApiError } from "@/api/client";
import { PipelineNotAvailable } from "@/components/conversation/PipelineNotAvailable";
import { Button } from "@/components/ui/Button";
import { ScoreBar } from "@/components/ui/ScoreBar";
import { Spinner } from "@/components/ui/Spinner";
import { useEmotionPredictions, useProcessEmotion } from "@/features/emotion/hooks";
import { useTranscript } from "@/features/audio/hooks";
import type { EmotionProcessingJob } from "@/types/api";

function formatTimestamp(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

export function EmotionPanel({
  conversationId,
  messageId,
}: {
  conversationId: string;
  messageId: string | null;
}) {
  const [lastJob, setLastJob] = useState<EmotionProcessingJob | null>(null);
  const transcriptQuery = useTranscript(conversationId, messageId);
  const predictionsQuery = useEmotionPredictions(conversationId, messageId);
  const processEmotion = useProcessEmotion(conversationId);

  if (!messageId) {
    return (
      <PipelineNotAvailable
        icon={<Waves className="text-[var(--color-text-tertiary)]" size={18} />}
        title="Speaker & emotion"
        phaseLabel="Select a transcript"
        description="Process an audio upload above, then select it to analyze emotion per speaker turn."
      />
    );
  }

  async function handleAnalyze() {
    if (!messageId) return;
    const job = await processEmotion.mutateAsync(messageId);
    setLastJob(job);
  }

  const turns = transcriptQuery.data?.aligned_turns ?? [];
  const predictions = predictionsQuery.data ?? [];
  const hasPredictions = predictions.length > 0;

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="flex items-center justify-between gap-2 border-b border-[var(--color-border)] p-2">
        <span className="text-xs font-medium text-[var(--color-text-secondary)]">Speaker & emotion</span>
        <Button
          variant="ghost"
          className="px-2 py-1 text-xs"
          isLoading={processEmotion.isPending}
          onClick={handleAnalyze}
        >
          {hasPredictions ? "Re-analyze" : "Analyze emotion"}
        </Button>
      </div>

      {lastJob?.status === "unavailable" && (
        <p className="p-2 text-xs text-[var(--color-text-tertiary)]">{lastJob.error_message}</p>
      )}
      {lastJob?.status === "failed" && (
        <p role="alert" className="p-2 text-xs text-[var(--color-signal-negative)]">
          {lastJob.error_message ?? "Emotion analysis failed."}
        </p>
      )}
      {processEmotion.isError && (
        <p role="alert" className="p-2 text-xs text-[var(--color-signal-negative)]">
          {processEmotion.error instanceof ApiError
            ? processEmotion.error.message
            : "Emotion analysis failed."}
        </p>
      )}

      <div className="min-h-0 flex-1 overflow-y-auto">
        {(transcriptQuery.isLoading || predictionsQuery.isLoading) && (
          <div className="flex justify-center py-4">
            <Spinner />
          </div>
        )}
        {!transcriptQuery.isLoading && turns.length === 0 && (
          <p className="p-3 text-xs text-[var(--color-text-tertiary)]">No speaker turns to analyze yet.</p>
        )}
        <ol className="flex flex-col gap-3 p-3 text-sm">
          {turns.map((turn) => {
            const prediction = predictions.find((p) => p.aligned_turn_id === turn.id);
            return (
              <li key={turn.id} className="flex flex-col gap-1">
                <div className="flex items-center gap-2 text-xs text-[var(--color-text-tertiary)]">
                  <span className="font-medium text-[var(--color-accent-strong)]">
                    {turn.speaker_label ?? "Unknown speaker"}
                  </span>
                  <span>{formatTimestamp(turn.start_ms)}</span>
                </div>
                <p className="text-[var(--color-text-primary)]">&ldquo;{turn.text}&rdquo;</p>
                {prediction ? (
                  <div className="flex flex-col gap-1.5">
                    <p className="text-xs text-[var(--color-signal-positive)]">
                      Emotion: {prediction.predicted_label} · {Math.round(prediction.confidence * 100)}%
                      <span className="ml-1 text-[var(--color-text-tertiary)]">(model output, not a verified fact)</span>
                    </p>
                    <ScoreBar
                      label="Confidence"
                      value={prediction.confidence}
                      color="var(--color-signal-positive)"
                    />
                  </div>
                ) : (
                  <p className="text-xs text-[var(--color-text-tertiary)]">Not yet analyzed.</p>
                )}
              </li>
            );
          })}
        </ol>
      </div>
    </div>
  );
}
