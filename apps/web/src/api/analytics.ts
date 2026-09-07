import { apiFetch } from "@/api/client";
import type { AnalyticsDashboard } from "@/types/api";

export function getAnalyticsDashboard(): Promise<AnalyticsDashboard> {
  return apiFetch<AnalyticsDashboard>("/analytics/dashboard");
}
