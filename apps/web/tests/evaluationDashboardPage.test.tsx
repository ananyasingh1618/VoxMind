import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { EvaluationDashboardPage } from "@/app/routes/app/EvaluationDashboardPage";
import * as evaluationHooks from "@/features/evaluation/hooks";
import type { EvaluationDashboard, EvaluationRunSummary, EvaluationType } from "@/types/api";

import { renderWithProviders } from "./testUtils";

vi.mock("@/features/evaluation/hooks");

const TYPES: EvaluationType[] = ["stt", "emotion", "retrieval", "grounding", "system"];

function neverRunDashboard(): EvaluationDashboard {
  return { types: TYPES.map((t) => ({ evaluation_type: t, status: "never_run", latest: null })) };
}

function runFixture(overrides: Partial<EvaluationRunSummary> = {}): EvaluationRunSummary {
  return {
    id: "11111111-1111-1111-1111-111111111111",
    evaluation_type: "stt",
    dataset_version: "stt-eval-v1",
    model_version: "faster-whisper:tiny",
    configuration: {},
    code_version: null,
    random_seed: 42,
    sample_count: 5,
    status: "completed",
    metrics: { n_samples: 5, avg_wer: 0.05, avg_cer: 0.0363 },
    errors: [],
    notes: null,
    started_at: "2026-09-06T00:00:00Z",
    completed_at: "2026-09-06T00:00:10Z",
    created_at: "2026-09-06T00:00:00Z",
    ...overrides,
  };
}

function mockDashboard(data: EvaluationDashboard) {
  vi.mocked(evaluationHooks.useEvaluationDashboard).mockReturnValue({
    data,
    isLoading: false,
    isError: false,
    refetch: vi.fn(),
  } as unknown as ReturnType<typeof evaluationHooks.useEvaluationDashboard>);
}

describe("EvaluationDashboardPage", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.mocked(evaluationHooks.useEvaluationRuns).mockReturnValue({
      data: [],
      isLoading: false,
    } as unknown as ReturnType<typeof evaluationHooks.useEvaluationRuns>);
  });

  it("shows 'Never run' for every type when nothing has been evaluated, never a fabricated metric", () => {
    mockDashboard(neverRunDashboard());
    renderWithProviders(<EvaluationDashboardPage />);

    expect(screen.getAllByText("Never run")).toHaveLength(5);
    expect(screen.getAllByText(/no real evaluation has been run yet/i)).toHaveLength(5);
  });

  it("renders a real headline metric for a type that has actually been evaluated", () => {
    const dashboard = neverRunDashboard();
    dashboard.types = dashboard.types.map((t) =>
      t.evaluation_type === "stt" ? { evaluation_type: "stt", status: "evaluated", latest: runFixture() } : t,
    );
    mockDashboard(dashboard);
    renderWithProviders(<EvaluationDashboardPage />);

    expect(screen.getByText("Evaluated")).toBeInTheDocument();
    expect(screen.getByText("0.05")).toBeInTheDocument(); // avg_wer headline
    expect(screen.getByText("Avg. WER")).toBeInTheDocument();
    expect(screen.getByText(/stt-eval-v1/)).toBeInTheDocument();
    // Every other type must still honestly report never_run alongside a real one.
    expect(screen.getAllByText("Never run")).toHaveLength(4);
  });

  it("renders an unavailable metric distinctly, never as a fabricated zero", () => {
    const dashboard = neverRunDashboard();
    dashboard.types = dashboard.types.map((t) =>
      t.evaluation_type === "grounding"
        ? {
            evaluation_type: "grounding" as const,
            status: "evaluated" as const,
            latest: runFixture({
              evaluation_type: "grounding",
              dataset_version: null,
              model_version: null,
              metrics: {
                n_generations: 3,
                citation_validity_rate: { status: "unavailable", reason: "No citations returned yet." },
              },
            }),
          }
        : t,
    );
    mockDashboard(dashboard);
    renderWithProviders(<EvaluationDashboardPage />);

    expect(screen.getByText("unavailable")).toBeInTheDocument();
  });
});
