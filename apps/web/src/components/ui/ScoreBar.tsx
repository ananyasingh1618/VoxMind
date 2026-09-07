/**
 * A small inline score/confidence visualization - a label, the exact
 * percentage as real text (never bar-only, so the value stays readable to
 * screen readers and at a glance), and a thin proportional bar reinforcing
 * it visually. Purely a presentation layer over a value the caller already
 * fetched; it introduces no new data or interaction, only clearer visual
 * hierarchy for the transcript/emotion/incongruence panels (Phase 8).
 * `aria-hidden` on the bar itself avoids a redundant, valueless
 * `progressbar` announcement - the adjacent text already conveys the value
 * accessibly.
 */
export function ScoreBar({
  label,
  value,
  color = "var(--color-accent)",
  hint,
}: {
  label: string;
  value: number;
  color?: string;
  hint?: string;
}) {
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)));
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between gap-2 text-xs">
        <span className="text-[var(--color-text-secondary)]">{label}</span>
        <span className="font-medium tabular-nums text-[var(--color-text-primary)]">{pct}%</span>
      </div>
      <div
        aria-hidden="true"
        className="h-1 w-full overflow-hidden rounded-full bg-[var(--color-surface-raised)]"
      >
        <div
          className="h-full rounded-full transition-[width]"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      {hint && <p className="text-[10px] text-[var(--color-text-tertiary)]">{hint}</p>}
    </div>
  );
}
