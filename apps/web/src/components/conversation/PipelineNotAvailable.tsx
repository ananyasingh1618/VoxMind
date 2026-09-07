import type { ReactNode } from "react";

/**
 * Shared "nothing to show for the current selection yet" empty state, used
 * by the transcript/emotion/NLP/incongruence panels. The real backend
 * pipelines behind all of these exist and are wired up (Phases 2-4); this
 * component renders only when there's genuinely no selected message/turn/
 * transcript to analyze yet (e.g. no audio uploaded, or nothing selected in
 * the timeline) - never as a placeholder for a missing pipeline. There is
 * no fallback path here that fabricates transcript text, emotion scores, or
 * citations - a real "no data selected" state instead.
 */
export function PipelineNotAvailable({
  title,
  phaseLabel,
  description,
  icon,
}: {
  title: string;
  phaseLabel: string;
  description: string;
  icon?: ReactNode;
}) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 p-6 text-center">
      {icon}
      <p className="text-sm font-medium text-[var(--color-text-primary)]">{title}</p>
      <p className="max-w-xs text-xs text-[var(--color-text-tertiary)]">{description}</p>
      <span className="mt-1 rounded-full border border-[var(--color-border-strong)] px-2 py-0.5 text-[10px] uppercase tracking-wide text-[var(--color-text-tertiary)]">
        {phaseLabel}
      </span>
    </div>
  );
}
