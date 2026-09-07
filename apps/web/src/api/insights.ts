import { apiFetch } from "@/api/client";
import type { ConversationInsights } from "@/types/api";

export function getConversationInsights(conversationId: string): Promise<ConversationInsights> {
  return apiFetch<ConversationInsights>(`/conversations/${conversationId}/insights`);
}
