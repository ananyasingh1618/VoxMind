import { CheckCircle2, ChevronDown, ClipboardList, FlaskConical, MinusCircle } from "lucide-react";
import { useState } from "react";

import { Card } from "@/components/ui/Card";
import { ErrorPanel, LoadingPanel } from "@/components/ui/StatePanel";
import { useEvaluationDashboard, useEvaluationRuns } from "@/features/evaluation/hooks";
import { cn } from "@/lib/cn";
import type { EvaluationRunSummary, EvaluationType, EvaluationTypeSummary } from "@/types/api";

const TYPE_LABELS: Record<EvaluationType, string> = {
  stt: "Speech-to-text",
  emotion: "Emotion classifier",
  retrieval: "Retrieval",
  grounding: "Grounding",
  system: "System latency & reliability",
};

const TYPE_DESCRIPTIONS: Record<EvaluationType, string> = {
  stt: "Word/character error rate against a versioned reference-transcript dataset.",
  emotion: "Accuracy/F1 against the real, unmodified RAVDESS held-out test split.",
  retrieval: "Recall@K / Precision@K / MRR / nDCG against hand-labeled relevance judgments.",
  grounding: "Citation validity and grounding-status distribution over real generations.",
  system: "p50/p95 latency, completion rate, and failure rate over real execution data.",
};

function formatNumber(value: number): string {
  if (Number.isInteger(value)) return String(value);
  return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

function isUnavailableMarker(value: unknown): value is { status: "unavailable"; reason?: string } {
  return (
    typeof value === "object" &&
    value !== null &&
    "status" in value &&
    (value as { status?: unknown }).status === "unavailable"
  );
}

function MetricValue({ value }: { value: unknown }) {
  if (value === null || value === undefined) {
    return <span className="text-[var(--color-text-tertiary)]">—</span>;
  }
  if (isUnavailableMarker(value)) {
    return (
      <span className="inline-flex items-center gap-1 text-[var(--color-text-tertiary)]">
        <MinusCircle size={12} /> unavailable
        {value.reason && <span className="italic">— {value.reason}</span>}
      </span>
    );
  }
  if (typeof value === "number") return <span>{formatNumber(value)}</span>;
  if (typeof value === "boolean") return <span>{value ? "yes" : "no"}</span>;
  if (typeof value === "string") return <span>{value}</span>;
  if (Array.isArray(value)) {
    if (value.length === 0) return <span className="text-[var(--color-text-tertiary)]">none</span>;
    if (value.every((v) => typeof v !== "object")) {
      return <span>{value.join(", ")}</span>;
    }
    return <span className="text-[var(--color-text-tertiary)]">{value.length} item(s)</span>;
  }
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    return (
      <dl className="ml-3 flex flex-col gap-1 border-l border-[var(--color-border)] pl-3">
        {entries.map(([k, v]) => (
          <div key={k} className="flex flex-wrap items-baseline gap-1.5">
            <dt className="text-xs text-[var(--color-text-tertiary)]">{k}:</dt>
            <dd className="text-xs text-[var(--color-text-secondary)]">
              <MetricValue value={v} />
            </dd>
          </div>
        ))}
      </dl>
    );
  }
  return null;
}

function HeadlineStat({ evaluationType, run }: { evaluationType: EvaluationType; run: EvaluationRunSummary }) {
  const m = run.metrics as Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
  let label = "Samples";
  let value: string = String(run.sample_count);

  switch (evaluationType) {
    case "stt":
      if (typeof m.avg_wer === "number") {
        label = "Avg. WER";
        value = formatNumber(m.avg_wer);
      }
      break;
    case "emotion":
      if (typeof m.macro_f1 === "number") {
        label = "Macro F1";
        value = formatNumber(m.macro_f1);
      }
      break;
    case "retrieval":
      if (typeof m.mrr === "number") {
        label = "MRR";
        value = formatNumber(m.mrr);
      }
      break;
    case "grounding":
      if (m.citation_validity_rate && typeof m.citation_validity_rate.value === "number") {
        label = "Citation validity";
        value = formatNumber(m.citation_validity_rate.value);
      } else {
        label = "Citation validity";
        value = "unavailable";
      }
      break;
    case "system":
      if (typeof m.voice_turn_pipeline?.completion_rate === "number") {
        label = "Voice turn completion rate";
        value = formatNumber(m.voice_turn_pipeline.completion_rate);
      }
      break;
  }

  return (
    <div>
      <p className="text-2xl font-semibold text-[var(--color-text-primary)]">{value}</p>
      <p className="text-xs text-[var(--color-text-tertiary)]">{label}</p>
    </div>
  );
}

function RunHistory({ evaluationType }: { evaluationType: EvaluationType }) {
  const { data, isLoading } = useEvaluationRuns(evaluationType, true);
  if (isLoading) return <p className="py-2 text-xs text-[var(--color-text-tertiary)]">Loading history…</p>;
  if (!data || data.length <= 1) {
    return <p className="py-2 text-xs text-[var(--color-text-tertiary)]">No earlier runs to compare yet.</p>;
  }
  return (
    <ul className="mt-2 flex flex-col gap-1.5 border-t border-[var(--color-border)] pt-2">
      {data.map((run) => (
        <li key={run.id} className="flex items-center justify-between text-xs text-[var(--color-text-secondary)]">
          <span>{new Date(run.created_at).toLocaleString()}</span>
          <span className="text-[var(--color-text-tertiary)]">
            {run.status} · n={run.sample_count}
            {run.model_version ? ` · ${run.model_version}` : ""}
          </span>
        </li>
      ))}
    </ul>
  );
}

function EvaluationTypeCard({ summary }: { summary: EvaluationTypeSummary }) {
  const [expanded, setExpanded] = useState(false);
  const { evaluation_type: evaluationType, status, latest } = summary;

  return (
    <Card className="flex flex-col gap-4 p-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-medium text-[var(--color-text-primary)]">{TYPE_LABELS[evaluationType]}</h3>
          <p className="mt-0.5 text-xs text-[var(--color-text-tertiary)]">{TYPE_DESCRIPTIONS[evaluationType]}</p>
        </div>
        <span
          className={cn(
            "inline-flex shrink-0 items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium",
            status === "evaluated"
              ? "bg-[var(--color-signal-positive)]/15 text-[var(--color-signal-positive)]"
              : "bg-[var(--color-surface-raised)] text-[var(--color-text-tertiary)]",
          )}
        >
          {status === "evaluated" ? <CheckCircle2 size={12} /> : <MinusCircle size={12} />}
          {status === "evaluated" ? "Evaluated" : "Never run"}
        </span>
      </div>

      {!latest ? (
        <p className="text-xs text-[var(--color-text-tertiary)]">
          No real evaluation has been run yet in this environment. See docs/evaluation.md for the CLI command.
        </p>
      ) : (
        <>
          <div className="flex items-center gap-6">
            <HeadlineStat evaluationType={evaluationType} run={latest} />
            <dl className="flex flex-col gap-0.5 text-xs text-[var(--color-text-tertiary)]">
              {latest.dataset_version && <div>dataset: {latest.dataset_version}</div>}
              {latest.model_version && <div>model: {latest.model_version}</div>}
              <div>n={latest.sample_count} · {new Date(latest.created_at).toLocaleDateString()}</div>
            </dl>
          </div>

          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className="flex items-center gap-1 self-start text-xs text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]"
            aria-expanded={expanded}
          >
            <ChevronDown size={12} className={cn("transition-transform", expanded && "rotate-180")} />
            {expanded ? "Hide details" : "Show full metrics"}
          </button>

          {expanded && (
            <div className="flex flex-col gap-3">
              <MetricValue value={latest.metrics} />
              {latest.notes && (
                <p className="rounded-md bg-[var(--color-surface-raised)] p-2 text-xs text-[var(--color-text-tertiary)]">
                  {latest.notes}
                </p>
              )}
              <RunHistory evaluationType={evaluationType} />
            </div>
          )}
        </>
      )}
    </Card>
  );
}

export function EvaluationDashboardPage() {
  const { data, isLoading, isError, refetch } = useEvaluationDashboard();

  if (isLoading) return <LoadingPanel label="Loading evaluation results…" />;
  if (isError || !data) {
    return <ErrorPanel description="Couldn't load the evaluation dashboard." onRetry={() => refetch()} />;
  }

  return (
    <div className="flex h-full flex-col gap-6 overflow-y-auto p-6">
      <div>
        <h1 className="flex items-center gap-2 text-lg font-semibold text-[var(--color-text-primary)]">
          <FlaskConical size={20} /> Evaluation
        </h1>
        <p className="text-sm text-[var(--color-text-tertiary)]">
          Every number below comes from a real, executed evaluation run - never a placeholder. A type
          marked "Never run" has an implemented, executable evaluation with no result yet, not a fabricated one.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {data.types.map((summary) => (
          <EvaluationTypeCard key={summary.evaluation_type} summary={summary} />
        ))}
      </div>

      <Card className="flex items-start gap-2 p-4 text-xs text-[var(--color-text-tertiary)]">
        <ClipboardList size={14} className="mt-0.5 shrink-0" />
        <p>
          Evaluations run out-of-band via CLI (<code>ml/evaluation/evaluate_*.py</code>), not triggered from this
          page - see docs/evaluation.md for exact commands, dataset provenance, and what each metric does and does
          not measure.
        </p>
      </Card>
    </div>
  );
}
