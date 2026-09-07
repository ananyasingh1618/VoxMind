import { FileText } from "lucide-react";

import { PipelineNotAvailable } from "@/components/conversation/PipelineNotAvailable";
import type { AlignedTurn, TranscriptSegment } from "@/types/api";

function formatTimestamp(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

export function TranscriptPanel({
  segments,
  turns,
}: {
  segments?: TranscriptSegment[];
  turns?: AlignedTurn[];
}) {
  if (!segments) {
    return (
      <PipelineNotAvailable
        icon={<FileText className="text-[var(--color-text-tertiary)]" size={20} />}
        title="No transcript yet"
        phaseLabel="Upload audio"
        description="Upload an audio file above and run processing to see a timestamped, speaker-attributed transcript here."
      />
    );
  }

  // Prefer the derived, speaker-attributed turns when available - they read
  // more like a real conversation than raw per-segment Whisper output.
  if (turns && turns.length > 0) {
    return (
      <ol className="flex flex-col gap-3 overflow-y-auto p-4 text-sm">
        {turns.map((turn, index) => (
          <li key={index} className="flex flex-col gap-0.5">
            <div className="flex items-center gap-2 text-xs text-[var(--color-text-tertiary)]">
              <span className="font-medium text-[var(--color-accent-strong)]">
                {turn.speaker_label ?? "Unknown speaker"}
              </span>
              <span>
                {formatTimestamp(turn.start_ms)}–{formatTimestamp(turn.end_ms)}
              </span>
            </div>
            <p className="text-[var(--color-text-primary)]">{turn.text}</p>
          </li>
        ))}
      </ol>
    );
  }

  return (
    <ol className="flex flex-col gap-2 overflow-y-auto p-4 text-sm">
      {segments.map((segment, index) => (
        <li key={index} className="flex flex-col gap-0.5">
          <span className="text-xs text-[var(--color-text-tertiary)]">
            {formatTimestamp(segment.start_ms)}–{formatTimestamp(segment.end_ms)}
            {segment.confidence !== null && ` · ${Math.round(segment.confidence * 100)}%`}
          </span>
          <p className="text-[var(--color-text-primary)]">{segment.text}</p>
        </li>
      ))}
    </ol>
  );
}
