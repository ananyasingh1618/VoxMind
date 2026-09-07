import { apiFetch } from "@/api/client";
import type { NlpAnnotation } from "@/types/api";

export function processNlp(conversationId: string, messageId: string): Promise<NlpAnnotation> {
  return apiFetch<NlpAnnotation>(`/conversations/${conversationId}/messages/${messageId}/nlp/process`, {
    method: "POST",
  });
}

export function getNlp(conversationId: string, messageId: string): Promise<NlpAnnotation | null> {
  return apiFetch<NlpAnnotation | null>(`/conversations/${conversationId}/messages/${messageId}/nlp`);
}
