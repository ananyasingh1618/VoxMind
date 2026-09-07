import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AnalyticsDashboardPage } from "@/app/routes/app/AnalyticsDashboardPage";
import * as analyticsHooks from "@/features/analytics/hooks";
import type { AnalyticsDashboard } from "@/types/api";

import { renderWithProviders } from "./testUtils";

vi.mock("@/features/analytics/hooks");

const EMPTY_LATENCY = { avg_ms: null, p50_ms: null, p95_ms: null, sample_count: 0 };

function baseDashboard(overrides: Partial<AnalyticsDashboard> = {}): AnalyticsDashboard {
  return {
    conversations: { total: 0, active: 0, ended: 0, total_messages: 0 },
    latency: {
      stt: EMPTY_LATENCY,
      analysis: EMPTY_LATENCY,
      retrieval: EMPTY_LATENCY,
      llm: EMPTY_LATENCY,
      tts: EMPTY_LATENCY,
      end_to_end: EMPTY_LATENCY,
    },
    emotion: { total_predictions: 0, by_label: [] },
    intelligence: {
      sentiment_distribution: [],
      sentiment_trend: [],
      intent_distribution: [],
      mismatch_event_count: 0,
      mismatch_average_score: null,
      high_mismatch_event_count: 0,
    },
    retrieval: {
      total_queries: 0,
      queries_with_results: 0,
      average_chunks_retrieved: null,
      total_documents: 0,
      total_chunks: 0,
      rerank_usage_rate: null,
    },
    grounding: { by_status: [], total_generations: 0 },
    model_versions: [],
    failures: {
      audio_processing_failed: 0,
      emotion_processing_failed: 0,
      voice_turns_failed: 0,
      voice_turns_interrupted: 0,
      llm_unavailable: 0,
      guardrail_blocked: 0,
      guardrail_modified: 0,
    },
    pipeline_health: {
      llm_provider_configured: false,
      llm_provider: "local_dev",
      tts_provider_configured: true,
      tts_provider: "local_hf",
      diarization_available: false,
      moderation_provider: "keyword_fallback",
    },
    ...overrides,
  };
}

function mockDashboard(data: AnalyticsDashboard) {
  vi.mocked(analyticsHooks.useAnalyticsDashboard).mockReturnValue({
    data,
    isLoading: false,
    isError: false,
    refetch: vi.fn(),
  } as unknown as ReturnType<typeof analyticsHooks.useAnalyticsDashboard>);
}

describe("AnalyticsDashboardPage", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("shows a genuine empty state for a fresh account, never a fabricated chart", () => {
    mockDashboard(baseDashboard());
    renderWithProviders(<AnalyticsDashboardPage />);

    expect(screen.getByText(/no data yet/i)).toBeInTheDocument();
  });

  it("renders real stat cards and pipeline health once there is real data", () => {
    mockDashboard(
      baseDashboard({
        conversations: { total: 3, active: 2, ended: 1, total_messages: 12 },
        retrieval: {
          total_queries: 2,
          queries_with_results: 1,
          average_chunks_retrieved: 1.5,
          total_documents: 1,
          total_chunks: 4,
          rerank_usage_rate: 0.5,
        },
      }),
    );
    renderWithProviders(<AnalyticsDashboardPage />);

    expect(screen.getByText("3")).toBeInTheDocument(); // conversations total
    expect(screen.getByText("12")).toBeInTheDocument(); // total messages
    expect(screen.getByText(/local_dev/)).toBeInTheDocument();
    expect(screen.getByText(/not configured/i)).toBeInTheDocument();
  });
});
