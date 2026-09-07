import { apiFetch } from "@/api/client";
import type { IncongruenceSignal } from "@/types/api";

export function analyzeIncongruence(
  conversationId: string,
  messageId: string,
): Promise<IncongruenceSignal[]> {
  return apiFetch<IncongruenceSignal[]>(
    `/conversations/${conversationId}/messages/${messageId}/incongruence/analyze`,
    { method: "POST" },
  );
}

export function listIncongruence(
  conversationId: string,
  messageId: string,
): Promise<IncongruenceSignal[]> {
  return apiFetch<IncongruenceSignal[]>(
    `/conversations/${conversationId}/messages/${messageId}/incongruence`,
  );
}
