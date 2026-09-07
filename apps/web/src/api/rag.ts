import { apiFetch } from "@/api/client";
import type { AskResponse } from "@/types/api";

export function ask(conversationId: string, question: string): Promise<AskResponse> {
  return apiFetch<AskResponse>(`/conversations/${conversationId}/ask`, {
    method: "POST",
    body: { question },
  });
}
