import { useQuery } from "@tanstack/react-query";

import * as insightsApi from "@/api/insights";

export function useConversationInsights(conversationId: string | undefined) {
  return useQuery({
    queryKey: ["insights", conversationId ?? ""] as const,
    queryFn: () => insightsApi.getConversationInsights(conversationId as string),
    enabled: Boolean(conversationId),
  });
}
