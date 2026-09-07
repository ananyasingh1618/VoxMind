import { useQuery } from "@tanstack/react-query";

import * as analyticsApi from "@/api/analytics";

export function useAnalyticsDashboard() {
  return useQuery({
    queryKey: ["analytics", "dashboard"] as const,
    queryFn: analyticsApi.getAnalyticsDashboard,
  });
}
