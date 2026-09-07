import { useQuery } from "@tanstack/react-query";

import * as evaluationApi from "@/api/evaluation";
import type { EvaluationType } from "@/types/api";

export function useEvaluationDashboard() {
  return useQuery({
    queryKey: ["evaluation", "dashboard"] as const,
    queryFn: evaluationApi.getEvaluationDashboard,
  });
}

export function useEvaluationRuns(evaluationType: EvaluationType, enabled: boolean) {
  return useQuery({
    queryKey: ["evaluation", "runs", evaluationType] as const,
    queryFn: () => evaluationApi.getEvaluationRuns(evaluationType),
    enabled,
  });
}
