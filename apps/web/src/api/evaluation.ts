import { apiFetch } from "@/api/client";
import type { EvaluationDashboard, EvaluationRunSummary, EvaluationType } from "@/types/api";

export function getEvaluationDashboard(): Promise<EvaluationDashboard> {
  return apiFetch<EvaluationDashboard>("/evaluations/dashboard");
}

export function getEvaluationRuns(evaluationType: EvaluationType): Promise<EvaluationRunSummary[]> {
  return apiFetch<EvaluationRunSummary[]>(
    `/evaluations/runs?evaluation_type=${encodeURIComponent(evaluationType)}`,
  );
}
