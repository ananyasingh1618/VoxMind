import { Activity, AlertTriangle, BarChart3, Gauge } from "lucide-react";
import type { ReactElement } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Card } from "@/components/ui/Card";
import { EmptyPanel, ErrorPanel, LoadingPanel } from "@/components/ui/StatePanel";
import { useAnalyticsDashboard } from "@/features/analytics/hooks";
import type { LabelCount, LatencyOverview } from "@/types/api";

function StatCard({ label, value, hint }: { label: string; value: string | number; hint?: string }) {
  return (
    <Card className="p-4">
      <p className="text-xs text-[var(--color-text-tertiary)]">{label}</p>
      <p className="mt-1 text-2xl font-semibold text-[var(--color-text-primary)]">{value}</p>
      {hint && <p className="mt-0.5 text-xs text-[var(--color-text-tertiary)]">{hint}</p>}
    </Card>
  );
}

function ChartCard({
  title,
  data,
  isEmpty,
  emptyHint,
  children,
}: {
  title: string;
  data: unknown[];
  isEmpty: boolean;
  emptyHint: string;
  children: ReactElement;
}) {
  return (
    <Card className="p-4">
      {/* axe-core `heading-order`: every card on this page sits directly
          under the page's own <h1> with no intervening <h2> - was <h3>
          before, incorrectly skipping a level. */}
      <h2 className="mb-3 text-sm font-medium text-[var(--color-text-primary)]">{title}</h2>
      {isEmpty || data.length === 0 ? (
        <p className="py-8 text-center text-xs text-[var(--color-text-tertiary)]">{emptyHint}</p>
      ) : (
        <div className="h-52 w-full">
          <ResponsiveContainer width="100%" height="100%">
            {children}
          </ResponsiveContainer>
        </div>
      )}
    </Card>
  );
}

function LabelBarChart({ data }: { data: LabelCount[] }) {
  return (
    <BarChart data={data}>
      <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
      <XAxis dataKey="label" tick={{ fontSize: 11 }} stroke="var(--color-text-tertiary)" />
      <YAxis allowDecimals={false} tick={{ fontSize: 11 }} stroke="var(--color-text-tertiary)" />
      <Tooltip />
      <Bar dataKey="count" fill="var(--color-accent)" radius={[4, 4, 0, 0]} />
    </BarChart>
  );
}

const LATENCY_LABELS: [key: keyof LatencyOverview, label: string][] = [
  ["stt", "STT"],
  ["analysis", "Analysis"],
  ["retrieval", "Retrieval"],
  ["llm", "LLM"],
  ["tts", "TTS"],
  ["end_to_end", "End-to-end"],
];

export function AnalyticsDashboardPage() {
  const { data, isLoading, isError, refetch } = useAnalyticsDashboard();

  if (isLoading) return <LoadingPanel label="Loading analytics…" />;
  if (isError || !data) {
    return (
      <ErrorPanel description="Couldn't load the analytics dashboard." onRetry={() => refetch()} />
    );
  }

  const hasAnyConversations = data.conversations.total > 0;
  const latencyData = LATENCY_LABELS.map(([key, label]) => ({
    label,
    avg_ms: data.latency[key].avg_ms,
    sample_count: data.latency[key].sample_count,
  })).filter((d) => d.sample_count > 0);

  return (
    // axe-core `scrollable-region-focusable`: WCAG 2.1.1 keyboard access -
    // this page's content can genuinely overflow (many charts/cards).
    <div className="flex h-full flex-col gap-6 overflow-y-auto p-6" tabIndex={0}>
      <div>
        <h1 className="flex items-center gap-2 text-lg font-semibold text-[var(--color-text-primary)]">
          <BarChart3 size={20} /> Analytics
        </h1>
        <p className="text-sm text-[var(--color-text-tertiary)]">
          Real measured data from your own conversations - never simulated.
        </p>
      </div>

      {!hasAnyConversations ? (
        <EmptyPanel
          icon={<Gauge className="text-[var(--color-text-tertiary)]" size={28} />}
          title="No data yet"
          description="Start a conversation and ask a question to see real usage, latency, and intelligence analytics here."
        />
      ) : (
        <>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <StatCard label="Conversations" value={data.conversations.total} hint={`${data.conversations.active} active`} />
            <StatCard label="Messages" value={data.conversations.total_messages} />
            <StatCard label="Documents" value={data.retrieval.total_documents} hint={`${data.retrieval.total_chunks} chunks`} />
            <StatCard label="Retrieval queries" value={data.retrieval.total_queries} />
          </div>

          <ChartCard
            title="Average latency by stage (ms)"
            data={latencyData}
            isEmpty={latencyData.length === 0}
            emptyHint="No real timing measurements yet."
          >
            <LabelBarChart data={latencyData.map((d) => ({ label: d.label, count: d.avg_ms ?? 0 }))} />
          </ChartCard>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <ChartCard
              title="Emotion distribution"
              data={data.emotion.by_label}
              isEmpty={data.emotion.total_predictions === 0}
              emptyHint="No emotion predictions recorded yet - process audio with a trained model active."
            >
              <LabelBarChart data={data.emotion.by_label} />
            </ChartCard>

            <ChartCard
              title="Sentiment distribution"
              data={data.intelligence.sentiment_distribution}
              isEmpty={data.intelligence.sentiment_distribution.length === 0}
              emptyHint="No sentiment analysis recorded yet."
            >
              <LabelBarChart data={data.intelligence.sentiment_distribution} />
            </ChartCard>

            <ChartCard
              title="Intent distribution"
              data={data.intelligence.intent_distribution}
              isEmpty={data.intelligence.intent_distribution.length === 0}
              emptyHint="No intent classifications recorded yet."
            >
              <LabelBarChart data={data.intelligence.intent_distribution} />
            </ChartCard>

            <ChartCard
              title="Grounding status"
              data={data.grounding.by_status}
              isEmpty={data.grounding.total_generations === 0}
              emptyHint="No LLM generations recorded yet."
            >
              <LabelBarChart data={data.grounding.by_status} />
            </ChartCard>
          </div>

          {data.intelligence.sentiment_trend.length > 1 && (
            <Card className="p-4">
              <h2 className="mb-3 text-sm font-medium text-[var(--color-text-primary)]">Sentiment trend</h2>
              <div className="h-52 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={data.intelligence.sentiment_trend}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                    <XAxis dataKey="date" tick={{ fontSize: 11 }} stroke="var(--color-text-tertiary)" />
                    <YAxis allowDecimals={false} tick={{ fontSize: 11 }} stroke="var(--color-text-tertiary)" />
                    <Tooltip />
                    <Line type="monotone" dataKey="positive" stroke="var(--color-signal-positive)" />
                    <Line type="monotone" dataKey="neutral" stroke="var(--color-text-tertiary)" />
                    <Line type="monotone" dataKey="negative" stroke="var(--color-signal-negative)" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </Card>
          )}

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <Card className="p-4">
              <h2 className="mb-2 flex items-center gap-1.5 text-sm font-medium text-[var(--color-text-primary)]">
                <Activity size={14} /> Semantic-vocal incongruence
              </h2>
              {data.intelligence.mismatch_event_count === 0 ? (
                <p className="text-xs text-[var(--color-text-tertiary)]">No incongruence signals recorded yet.</p>
              ) : (
                <div className="text-sm text-[var(--color-text-secondary)]">
                  <p>{data.intelligence.mismatch_event_count} event(s) computed</p>
                  <p>Average score: {data.intelligence.mismatch_average_score?.toFixed(2)}</p>
                  <p>{data.intelligence.high_mismatch_event_count} scored above the 0.6 "notable divergence" threshold</p>
                  <p className="mt-1 text-xs text-[var(--color-text-tertiary)]">
                    An analytical signal only - never a deception or diagnostic indicator.
                  </p>
                </div>
              )}
            </Card>

            <Card className="p-4">
              <h2 className="mb-2 flex items-center gap-1.5 text-sm font-medium text-[var(--color-text-primary)]">
                <AlertTriangle size={14} /> Failures & guardrail actions
              </h2>
              <ul className="flex flex-col gap-1 text-sm text-[var(--color-text-secondary)]">
                <li>Audio processing failed: {data.failures.audio_processing_failed}</li>
                <li>Emotion processing failed: {data.failures.emotion_processing_failed}</li>
                <li>Voice turns failed: {data.failures.voice_turns_failed}</li>
                <li>Voice turns interrupted: {data.failures.voice_turns_interrupted}</li>
                <li>LLM unavailable: {data.failures.llm_unavailable}</li>
                <li>Guardrail blocked: {data.failures.guardrail_blocked}</li>
                <li>Guardrail modified: {data.failures.guardrail_modified}</li>
              </ul>
            </Card>
          </div>
        </>
      )}

      <Card className="p-4">
        <h2 className="mb-2 text-sm font-medium text-[var(--color-text-primary)]">Pipeline health</h2>
        <ul className="flex flex-col gap-1 text-sm text-[var(--color-text-secondary)]">
          <li>
            LLM provider: <strong>{data.pipeline_health.llm_provider}</strong> —{" "}
            {data.pipeline_health.llm_provider_configured ? "configured" : "not configured"}
          </li>
          <li>
            TTS provider: <strong>{data.pipeline_health.tts_provider}</strong> —{" "}
            {data.pipeline_health.tts_provider_configured ? "configured" : "not configured"}
          </li>
          <li>Diarization: {data.pipeline_health.diarization_available ? "available" : "unavailable (no HF token)"}</li>
          <li>Moderation: {data.pipeline_health.moderation_provider}</li>
        </ul>
      </Card>

      <Card className="p-4">
        <h2 className="mb-2 text-sm font-medium text-[var(--color-text-primary)]">Model versions</h2>
        {data.model_versions.length === 0 ? (
          <p className="text-xs text-[var(--color-text-tertiary)]">No models registered yet.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[24rem] text-left text-xs">
              <thead className="text-[var(--color-text-tertiary)]">
                <tr>
                  <th className="pb-1 pr-3">Component</th>
                  <th className="pb-1 pr-3">Version</th>
                  <th className="pb-1 pr-3">Active</th>
                  <th className="pb-1">Trained</th>
                </tr>
              </thead>
              <tbody className="text-[var(--color-text-secondary)]">
                {data.model_versions.map((mv) => (
                  <tr key={`${mv.component}-${mv.version_tag}`} className="border-t border-[var(--color-border)]">
                    <td className="py-1 pr-3">{mv.component}</td>
                    <td className="py-1 pr-3">{mv.version_tag}</td>
                    <td className="py-1 pr-3">{mv.is_active ? "yes" : "no"}</td>
                    <td className="py-1">{mv.trained ? "yes" : "no"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
